// Value-parameterized mirrors of test_kkt_system.cpp's dense-reference-comparison tests, run
// against every Settings::linear_solver backend. Same problem data, same dense reference --
// only Settings::linear_solver varies -- so any divergence between backends here points
// directly at a bug in the QdldlLdlt wrapper (permutation direction, sparsity-pattern
// assumptions, or the regularization-retry path), not at the algorithm itself.

#include "conicxx/kkt/kkt_system.h"

#include <gtest/gtest.h>

#include <Eigen/Cholesky>

#include "conicxx/cones/cone_set.h"
#include "test_helpers.h"

using namespace conicxx;
using namespace conicxx::detail;

namespace {

Mat denseReferenceK(const Mat& P, const Mat& A, const Mat& Hs, Scalar regP, Scalar regA) {
  const Index n = static_cast<Index>(P.rows());
  const Index m = static_cast<Index>(A.rows());
  Mat K = Mat::Zero(n + m, n + m);
  K.topLeftCorner(n, n) = P + regP * Mat::Identity(n, n);
  K.topRightCorner(n, m) = A.transpose();
  K.bottomLeftCorner(m, n) = A;
  K.bottomRightCorner(m, m) = -Hs - regA * Mat::Identity(m, m);
  return K;
}

class KktSystemBackends : public ::testing::TestWithParam<LinearSolverBackend> {};

}  // namespace

TEST_P(KktSystemBackends, SetupAndSolveMatchesDenseReference) {
  SparseMat P_upper = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {1, 1, 3.0}});
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});

  ConeSpec spec;
  spec.nonneg_dim = 2;
  ConeSet cones(spec);

  Settings settings;
  settings.linear_solver = GetParam();
  KktSystem kkt;
  ASSERT_TRUE(kkt.setup(P_upper, A, cones, settings));
  ASSERT_TRUE(kkt.isFactorized());

  Mat P(2, 2);
  P << 2, 0, 0, 3;
  Mat A_dense(2, 2);
  A_dense << 1, 0, 0, 1;
  Mat Hs = Mat::Identity(2, 2);
  Mat Kref = denseReferenceK(P, A_dense, Hs, settings.static_reg_P, settings.static_reg_A);

  Vec rhs(4);
  rhs << 1.0, 2.0, 3.0, 4.0;
  Vec x_ref = Kref.ldlt().solve(rhs);

  Vec x(4);
  Scalar relres = kkt.solve(rhs, x);
  EXPECT_LT(relres, 1e-10);
  testutil::expectVecNear(x, x_ref, 1e-8);
}

TEST_P(KktSystemBackends, UpdateScalingChangesOnlyHsBlock) {
  SparseMat P_upper = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {1, 1, 3.0}});
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  ConeSpec spec;
  spec.nonneg_dim = 2;
  ConeSet cones(spec);
  Settings settings;
  settings.linear_solver = GetParam();
  KktSystem kkt;
  ASSERT_TRUE(kkt.setup(P_upper, A, cones, settings));

  Vec s(2), z(2);
  s << 4.0, 9.0;
  z << 1.0, 1.0;
  cones.updateScaling(s, z);
  ASSERT_TRUE(kkt.updateScalingAndFactorize(cones));

  Mat P(2, 2);
  P << 2, 0, 0, 3;
  Mat A_dense = Mat::Identity(2, 2);
  Mat Hs = Mat::Zero(2, 2);
  Hs.diagonal() = (s.array() / z.array()).matrix();
  Mat Kref = denseReferenceK(P, A_dense, Hs, settings.static_reg_P, settings.static_reg_A);

  Vec rhs(4);
  rhs << 1.0, -1.0, 0.5, 2.0;
  Vec x_ref = Kref.ldlt().solve(rhs);

  Vec x(4);
  kkt.solve(rhs, x);
  testutil::expectVecNear(x, x_ref, 1e-8);
}

TEST_P(KktSystemBackends, HandlesRankDeficientEqualityBlockViaRegularization) {
  // Exercises the dynamic-regularization retry path (see KktSystem::factorizeWithRetry /
  // escalateAndResolve) through whichever backend is under test.
  SparseMat P_upper = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});

  ConeSpec spec;
  spec.zero_dim = 2;
  ConeSet cones(spec);
  Settings settings;
  settings.linear_solver = GetParam();
  KktSystem kkt;
  ASSERT_TRUE(kkt.setup(P_upper, A, cones, settings));

  Vec rhs(4);
  rhs << 1.0, 1.0, 0.0, 0.0;
  Vec x(4);
  Scalar relres = kkt.solve(rhs, x);
  EXPECT_TRUE(x.allFinite());
  EXPECT_LT(relres, 1e-6);
}

INSTANTIATE_TEST_SUITE_P(
    EigenAndQdldl, KktSystemBackends,
    ::testing::Values(LinearSolverBackend::Eigen, LinearSolverBackend::Qdldl,
                      LinearSolverBackend::RegularizedLdlt),
    [](const ::testing::TestParamInfo<LinearSolverBackend>& param_info) {
      switch (param_info.param) {
        case LinearSolverBackend::Qdldl:
          return "Qdldl";
        case LinearSolverBackend::RegularizedLdlt:
          return "RegularizedLdlt";
        default:
          return "Eigen";
      }
    });
