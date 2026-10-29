#include "conicxx/kkt_system.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "conicxx/cone_set.h"
#include "test_helpers.h"

using namespace conicxx;
using namespace conicxx::detail;

namespace {

// Dense [P A'; A -Hs] without regularization: solve() refines against the exact matrix, so the
// regularization must not show up in its result.
Mat denseReferenceK(const Mat& P, const Mat& A, const Mat& Hs) {
  const Index n = static_cast<Index>(P.rows());
  const Index m = static_cast<Index>(A.rows());
  Mat K = Mat::Zero(n + m, n + m);
  K.topLeftCorner(n, n) = P;
  K.topRightCorner(n, m) = A.transpose();
  K.bottomLeftCorner(m, n) = A;
  K.bottomRightCorner(m, m) = -Hs;
  return K;
}

}  // namespace

TEST(KktSystem, SetupAndSolveMatchesDenseReference) {
  SparseMat P_upper = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {1, 1, 3.0}});
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  ConeSpec spec;
  spec.nonneg_dim = 2;
  ConeSet cones(spec);

  KktSystem kkt;
  ASSERT_TRUE(kkt.setup(P_upper, A, cones, Settings{}));
  EXPECT_FALSE(kkt.isFactorized());  // setup() is symbolic only
  ASSERT_TRUE(kkt.updateScalingAndFactorize(cones));
  ASSERT_TRUE(kkt.isFactorized());

  Mat P(2, 2);
  P << 2, 0, 0, 3;
  const Mat Kref = denseReferenceK(P, Mat::Identity(2, 2), Mat::Identity(2, 2));
  Vec rhs(4);
  rhs << 1.0, 2.0, 3.0, 4.0;
  Vec x(4);
  EXPECT_LT(kkt.solve(rhs, x), 1e-12);
  testutil::expectVecNear(x, Kref.partialPivLu().solve(rhs), 1e-10);
}

TEST(KktSystem, UpdateScalingChangesOnlyHsBlock) {
  SparseMat P_upper = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {1, 1, 3.0}});
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  ConeSpec spec;
  spec.nonneg_dim = 2;
  ConeSet cones(spec);
  KktSystem kkt;
  ASSERT_TRUE(kkt.setup(P_upper, A, cones, Settings{}));

  Vec s(2), z(2);
  s << 4.0, 9.0;
  z << 1.0, 1.0;
  cones.updateScaling(s, z);
  ASSERT_TRUE(kkt.updateScalingAndFactorize(cones));

  Mat P(2, 2);
  P << 2, 0, 0, 3;
  Mat Hs = Mat::Zero(2, 2);
  Hs.diagonal() = (s.array() / z.array()).matrix();
  const Mat Kref = denseReferenceK(P, Mat::Identity(2, 2), Hs);
  Vec rhs(4);
  rhs << 1.0, -1.0, 0.5, 2.0;
  Vec x(4);
  kkt.solve(rhs, x);
  testutil::expectVecNear(x, Kref.partialPivLu().solve(rhs), 1e-10);
}

TEST(KktSystem, UpdateDataReproducesFreshSetupWithoutFactorizing) {
  ConeSpec spec;
  spec.nonneg_dim = 2;
  SparseMat P1 = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {1, 1, 3.0}});
  SparseMat A1 = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  SparseMat P2 = testutil::makeSparse(2, 2, {{0, 0, 5.0}, {1, 1, 7.0}});
  SparseMat A2 = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {1, 1, 0.5}});

  ConeSet cones_fresh(spec), cones_reused(spec);
  KktSystem fresh, reused;
  ASSERT_TRUE(fresh.setup(P2, A2, cones_fresh, Settings{}));
  ASSERT_TRUE(fresh.updateScalingAndFactorize(cones_fresh));
  ASSERT_TRUE(reused.setup(P1, A1, cones_reused, Settings{}));
  ASSERT_TRUE(reused.updateScalingAndFactorize(cones_reused));

  ASSERT_TRUE(reused.updateData(&P2, &A2));
  EXPECT_FALSE(reused.isFactorized());  // values only; the next IPM iteration factorizes
  EXPECT_EQ(reused.stats().factorizations, 1);
  ASSERT_TRUE(reused.updateScalingAndFactorize(cones_reused));

  Vec rhs(4);
  rhs << 0.3, -0.7, 1.1, 2.2;
  Vec x_fresh(4), x_reused(4);
  fresh.solve(rhs, x_fresh);
  reused.solve(rhs, x_reused);
  testutil::expectVecNear(x_fresh, x_reused, 1e-12);
}

TEST(KktSystem, UpdateDataRejectsPatternMismatch) {
  SparseMat P_upper = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {1, 1, 3.0}});
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  ConeSpec spec;
  spec.nonneg_dim = 2;
  ConeSet cones(spec);
  KktSystem kkt;
  ASSERT_TRUE(kkt.setup(P_upper, A, cones, Settings{}));
  SparseMat P_bad = testutil::makeSparse(2, 2, {{0, 0, 2.0}, {0, 1, 0.5}, {1, 1, 3.0}});
  EXPECT_FALSE(kkt.updateData(&P_bad, nullptr));
}

TEST(KktSystem, OneFactorizationPerUpdateWithEqualityRows) {
  // Zero-cone rows used to trigger a retry ladder of ~19 factorizations per update.
  const Index n = 6;
  std::vector<Triplet> p, a;
  for (Index i = 0; i < n; ++i) p.emplace_back(i, i, 1.0);
  for (Index i = 0; i < n; ++i) {  // banded equality rows
    a.emplace_back(i, i, 1.0);
    a.emplace_back(i, (i + 1) % n, 0.5);
  }
  for (Index c = 0; c < 2; ++c) {  // two 3-dim friction cones on the first variables
    a.emplace_back(n + 3 * c, 3 * c, -0.5);
    a.emplace_back(n + 3 * c + 1, 3 * c + 1, -1.0);
    a.emplace_back(n + 3 * c + 2, 3 * c + 2, -1.0);
  }
  SparseMat P_upper = testutil::makeSparse(n, n, p);
  SparseMat A = testutil::makeSparse(n + 6, n, a);
  ConeSpec spec;
  spec.zero_dim = n;
  spec.soc_dims = {3, 3};
  ConeSet cones(spec);
  KktSystem kkt;
  ASSERT_TRUE(kkt.setup(P_upper, A, cones, Settings{}));
  ASSERT_TRUE(kkt.updateScalingAndFactorize(cones));
  EXPECT_EQ(kkt.stats().factorizations, 1);
  EXPECT_EQ(kkt.ldlt().numPositivePivots(), n);
  Vec rhs = Vec::LinSpaced(kkt.dim(), -1.0, 1.0), x(kkt.dim());
  EXPECT_LT(kkt.solve(rhs, x), 1e-12);
  EXPECT_FALSE(kkt.equalityRankDeficient());
}

TEST(KktSystem, DuplicatedConsistentEqualityRowsAreSolved) {
  SparseMat P_upper = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});
  ConeSpec spec;
  spec.zero_dim = 2;
  ConeSet cones(spec);
  KktSystem kkt;
  ASSERT_TRUE(kkt.setup(P_upper, A, cones, Settings{}));
  ASSERT_TRUE(kkt.updateScalingAndFactorize(cones));

  Vec rhs(4), x(4);
  rhs << 1.0, 1.0, 0.5, 0.5;  // consistent: both equality rows ask for the same value
  const Scalar relres = kkt.solve(rhs, x);
  EXPECT_TRUE(x.allFinite());
  EXPECT_LT(relres, 1e-9);
  EXPECT_FALSE(kkt.equalityRankDeficient());
}

TEST(KktSystem, FlagsInconsistentEqualityRows) {
  SparseMat P_upper = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});
  ConeSpec spec;
  spec.zero_dim = 2;
  ConeSet cones(spec);
  KktSystem kkt;
  ASSERT_TRUE(kkt.setup(P_upper, A, cones, Settings{}));
  ASSERT_TRUE(kkt.updateScalingAndFactorize(cones));

  Vec rhs(4), x(4);
  rhs << 0.0, 0.0, 1.0, -1.0;  // the same row asked for two different values
  kkt.solve(rhs, x);
  EXPECT_TRUE(x.allFinite());
  EXPECT_TRUE(kkt.equalityRankDeficient());
}

TEST(KktSystem, NewtonDirectionSatisfiesLinearizedEqualitiesExactly) {
  // Equality rows carry static regularization in the factorization only; refinement against
  // the exact matrix must leave A_E dx = rhs_E satisfied to 1e-12.
  SparseMat P_upper = testutil::makeSparse(3, 3, {{0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}});
  SparseMat A = testutil::makeSparse(6, 3,
                                     {{0, 0, 1.0}, {0, 1, 1.0}, {1, 2, 1.0},      // zero cone
                                      {2, 0, -1.0},                               // orthant
                                      {3, 1, -1.0}, {4, 2, -1.0}, {5, 0, -1.0}});  // SOC
  const Index zero_dim = 2;
  ConeSpec spec;
  spec.zero_dim = zero_dim;
  spec.nonneg_dim = 1;
  spec.soc_dims = {3};
  ConeSet cones(spec);
  KktSystem kkt;
  ASSERT_TRUE(kkt.setup(P_upper, A, cones, Settings{}));
  ASSERT_TRUE(kkt.updateScalingAndFactorize(cones));

  Vec rhs(9);
  rhs << 0.3, -0.7, 1.1, 2.2, -0.4, 0.9, 1.0, -1.0, 0.5;
  Vec x(9);
  ASSERT_LT(kkt.solve(rhs, x), 1e-12);
  ASSERT_FALSE(kkt.equalityRankDeficient());
  const Vec Adx_E = Mat(A).topRows(zero_dim) * x.head(3);
  testutil::expectVecNear(Adx_E, rhs.segment(3, zero_dim), 1e-12);
}
