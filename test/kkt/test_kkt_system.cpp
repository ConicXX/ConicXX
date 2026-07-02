#include "conicxx/kkt/kkt_system.h"

#include <gtest/gtest.h>

#include <Eigen/Cholesky>

#include "conicxx/cones/cone_set.h"
#include "test_helpers.h"

using namespace conicxx;
using namespace conicxx::detail;

namespace {

// Builds the dense symmetric reference KKT matrix
//   [ P + regP*I     A'          ]
//   [ A              -Hs - regA*I]
// independently of KktSystem, for use as a ground-truth cross-check.
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

}  // namespace

TEST(KktSystem, SetupAndSolveMatchesDenseReference) {
  SparseMat P_upper = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {1, 1, 3.0}});
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});

  ConeSpec spec;
  spec.nonneg_dim = 2;
  ConeSet cones(spec);

  Settings settings;
  KktSystem kkt;
  ASSERT_TRUE(kkt.setup(P_upper, A, cones, settings));
  ASSERT_TRUE(kkt.isFactorized());

  Mat P(2, 2);
  P << 2, 0, 0, 3;
  Mat A_dense(2, 2);
  A_dense << 1, 0, 0, 1;
  Mat Hs = Mat::Identity(2, 2);  // NonnegativeCone starts at identity scaling
  Mat Kref = denseReferenceK(P, A_dense, Hs, settings.static_reg_P, settings.static_reg_A);

  Vec rhs(4);
  rhs << 1.0, 2.0, 3.0, 4.0;
  Vec x_ref = Kref.ldlt().solve(rhs);

  Vec x(4);
  Scalar relres = kkt.solve(rhs, x);
  EXPECT_LT(relres, 1e-10);
  testutil::expectVecNear(x, x_ref, 1e-8);
}

TEST(KktSystem, UpdateScalingChangesOnlyHsBlock) {
  SparseMat P_upper = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {1, 1, 3.0}});
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  ConeSpec spec;
  spec.nonneg_dim = 2;
  ConeSet cones(spec);
  Settings settings;
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
  Hs.diagonal() = (s.array() / z.array()).matrix();  // diag(4, 9)
  Mat Kref = denseReferenceK(P, A_dense, Hs, settings.static_reg_P, settings.static_reg_A);

  Vec rhs(4);
  rhs << 1.0, -1.0, 0.5, 2.0;
  Vec x_ref = Kref.ldlt().solve(rhs);

  Vec x(4);
  kkt.solve(rhs, x);
  testutil::expectVecNear(x, x_ref, 1e-8);
}

TEST(KktSystem, UpdateDataReproducesFreshSetup) {
  SparseMat P_upper1 = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {1, 1, 3.0}});
  SparseMat A1 = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  ConeSpec spec;
  spec.nonneg_dim = 2;

  // Path 1: fresh setup with the "new" data directly.
  SparseMat P_upper2 = testutil::makeSparse(2, 2, {{0, 0, 5.0}, {1, 1, 7.0}});
  SparseMat A2 = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {1, 1, 0.5}});

  ConeSet cones_fresh(spec);
  KktSystem kkt_fresh;
  ASSERT_TRUE(kkt_fresh.setup(P_upper2, A2, cones_fresh, Settings{}));

  // Path 2: setup with old data, then updateData() to the new numeric values
  // (same sparsity pattern), reusing the analyzed pattern.
  ConeSet cones_reused(spec);
  KktSystem kkt_reused;
  ASSERT_TRUE(kkt_reused.setup(P_upper1, A1, cones_reused, Settings{}));
  ASSERT_TRUE(kkt_reused.updateData(&P_upper2, &A2));

  Vec rhs(4);
  rhs << 0.3, -0.7, 1.1, 2.2;
  Vec x_fresh(4), x_reused(4);
  kkt_fresh.solve(rhs, x_fresh);
  kkt_reused.solve(rhs, x_reused);
  testutil::expectVecNear(x_fresh, x_reused, 1e-10);
}

TEST(KktSystem, UpdateDataRejectsPatternMismatch) {
  SparseMat P_upper = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {1, 1, 3.0}});
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  ConeSpec spec;
  spec.nonneg_dim = 2;
  ConeSet cones(spec);
  KktSystem kkt;
  ASSERT_TRUE(kkt.setup(P_upper, A, cones, Settings{}));

  // New P introduces an extra off-diagonal nonzero: different sparsity pattern.
  SparseMat P_upper_bad = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {0, 1, 0.5}, {1, 1, 3.0}});
  EXPECT_FALSE(kkt.updateData(&P_upper_bad, nullptr));
}

TEST(KktSystem, HandlesRankDeficientEqualityBlockViaRegularization) {
  // A zero-cone (equality) block whose rows are exactly a duplicate of each
  // other -- A is rank-deficient. Without regularization, an unpivoted LDLT
  // on the resulting quasi-definite-in-theory-but-degenerate-in-practice
  // system would hit an exact zero pivot; static+dynamic regularization
  // must keep the factorization well-defined and produce a finite result.
  SparseMat P_upper = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});

  ConeSpec spec;
  spec.zero_dim = 2;
  ConeSet cones(spec);
  KktSystem kkt;
  ASSERT_TRUE(kkt.setup(P_upper, A, cones, Settings{}));

  Vec rhs(4);
  rhs << 1.0, 1.0, 0.0, 0.0;
  Vec x(4);
  Scalar relres = kkt.solve(rhs, x);
  EXPECT_TRUE(x.allFinite());
  EXPECT_LT(relres, 1e-6);
}
