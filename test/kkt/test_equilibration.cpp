#include "conicxx/kkt/equilibration.h"

#include <gtest/gtest.h>

#include "conicxx/cones/cone_set.h"
#include "test_helpers.h"

using namespace conicxx;
using namespace conicxx::detail;

TEST(Equilibration, IdentityWhenDisabled) {
  SparseMat P = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  Vec q = Vec::Zero(2), b = Vec::Zero(2);
  ConeSpec spec;
  spec.nonneg_dim = 2;
  ConeSet cones(spec);

  Settings settings;
  settings.equilibrate = false;
  Equilibration eq;
  eq.compute(P, q, A, b, cones, settings);

  testutil::expectVecNear(eq.d(), Vec::Ones(2), 1e-14);
  testutil::expectVecNear(eq.e(), Vec::Ones(2), 1e-14);
  EXPECT_DOUBLE_EQ(eq.c(), 1.0);
}

TEST(Equilibration, ScaleUnscaleRoundtripPreservesKktConditions) {
  // A deliberately badly-scaled problem: P has widely different magnitudes,
  // A rows/cols span several orders of magnitude too.
  SparseMat P = testutil::makeSparse(3, 3, {{0, 0, 1e6}, {1, 1, 1e-4}, {2, 2, 50.0}});
  SparseMat A = testutil::makeSparse(4, 3,
                                      {{0, 0, 1e5},
                                       {1, 1, 1e-3},
                                       {2, 2, 20.0},
                                       {3, 0, 1e5},
                                       {3, 1, 1e-3},
                                       {3, 2, 20.0}});
  Vec q(3), b(4);
  q << 3.0, -2.0, 1.0;
  b << 1.0, 2.0, 3.0, 4.0;

  ConeSpec spec;
  spec.zero_dim = 1;
  spec.nonneg_dim = 3;
  ConeSet cones(spec);

  Settings settings;
  Equilibration eq;
  eq.compute(P, q, A, b, cones, settings);

  // d, e, c should all be strictly positive and finite.
  EXPECT_TRUE((eq.d().array() > 0).all());
  EXPECT_TRUE((eq.e().array() > 0).all());
  EXPECT_GT(eq.c(), 0.0);

  // Pick an arbitrary (x, z) and construct a consistent (s, q) so that the
  // *original* problem's KKT stationarity/feasibility equations hold
  // exactly: P x + A' z + q == 0  and  A x + s - b == 0.
  Vec x(3), z(4);
  x << 0.5, -1.5, 2.0;
  z << 0.2, -0.3, 0.1, 0.4;
  Mat Pdense = Mat(Mat(P).selfadjointView<Eigen::Upper>());
  Vec q_consistent = -(Pdense * x) - A.transpose() * z;
  Vec s = b - A * x;

  SparseMat P_scaled = P, A_scaled = A;
  Vec q_scaled = q_consistent, b_scaled = b;
  eq.scaleProblem(P_scaled, q_scaled, A_scaled, b_scaled);

  // x_hat = D^-1 x, s_hat = E s, z_hat = c E^-1 z (inverses of unscaleSolution).
  Vec x_hat = x.array() / eq.d().array();
  Vec s_hat = s.array() * eq.e().array();
  Vec z_hat = z.array() * eq.c() / eq.e().array();

  Mat P_scaled_dense = Mat(Mat(P_scaled).selfadjointView<Eigen::Upper>());
  Vec stationarity = P_scaled_dense * x_hat + A_scaled.transpose() * z_hat + q_scaled;
  Vec feasibility = A_scaled * x_hat + s_hat - b_scaled;

  testutil::expectVecNear(stationarity, Vec::Zero(3), 1e-8);
  testutil::expectVecNear(feasibility, Vec::Zero(4), 1e-8);

  // unscaleSolution() must be the exact inverse of the (x_hat,s_hat,z_hat)
  // construction above.
  Vec x_back = x_hat, s_back = s_hat, z_back = z_hat;
  eq.unscaleSolution(x_back, s_back, z_back);
  testutil::expectVecNear(x_back, x, 1e-10);
  testutil::expectVecNear(s_back, s, 1e-10);
  testutil::expectVecNear(z_back, z, 1e-10);
}

TEST(Equilibration, SecondOrderConeBlockGetsUniformScale) {
  SparseMat P = testutil::makeSparse(3, 3, {{0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}});
  // Rows of A within the SOC block have very different magnitudes.
  SparseMat A = testutil::makeSparse(3, 3, {{0, 0, 100.0}, {1, 1, 1.0}, {2, 2, 0.01}});
  Vec q = Vec::Zero(3), b = Vec::Zero(3);

  ConeSpec spec;
  spec.soc_dims = {3};
  ConeSet cones(spec);

  Settings settings;
  Equilibration eq;
  eq.compute(P, q, A, b, cones, settings);

  // All three e() entries (the single SOC block) must be identical.
  EXPECT_NEAR(eq.e()[0], eq.e()[1], 1e-12);
  EXPECT_NEAR(eq.e()[1], eq.e()[2], 1e-12);
}
