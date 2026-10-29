#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>

#include "conicxx/nonnegative_cone.h"
#include "conicxx/second_order_cone.h"
#include "conicxx/solver.h"
#include "test_helpers.h"

using namespace conicxx;

namespace {

// README example: min 0.5||x - (1, 1, 0.3)||^2 s.t. x0 + x1 = 1, x2 >= 0, (x2, x0, x1) in Q^3.
struct SmallSocp {
  SparseMat P = testutil::makeSparse(3, 3, {{0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}});
  Vec q = (Vec(3) << -1.0, -1.0, -0.3).finished();
  SparseMat A = testutil::makeSparse(
      5, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 2, -1.0}, {2, 2, -1.0}, {3, 0, -1.0}, {4, 1, -1.0}});
  Vec b = (Vec(5) << 1.0, 0.0, 0.0, 0.0, 0.0).finished();
  ConeSpec spec = [] {
    ConeSpec c;
    c.zero_dim = 1;
    c.nonneg_dim = 1;
    c.soc_dims = {3};
    return c;
  }();
};

void expectReadmeSolution(const Solution& sol) {
  ASSERT_EQ(sol.status, Status::Solved);
  EXPECT_NEAR(sol.x[0], 0.5, 1e-6);
  EXPECT_NEAR(sol.x[1], 0.5, 1e-6);
  EXPECT_NEAR(sol.x[2], std::sqrt(0.5), 1e-6);
}

}  // namespace

TEST(IpmSafeguards, SocCentralityMatchesNtScaling) {
  std::mt19937 rng(1);
  std::uniform_real_distribution<Scalar> U(-1.0, 1.0);
  SecondOrderCone cone(4);
  Vec s(4), z(4), lambda(4);
  for (int t = 0; t < 100; ++t) {
    for (Index i = 1; i < 4; ++i) {
      s[i] = U(rng);
      z[i] = U(rng);
    }
    s[0] = s.tail(3).norm() + 0.05 + std::abs(U(rng));
    z[0] = z.tail(3).norm() + 0.05 + std::abs(U(rng));
    cone.updateScaling(s, z);
    cone.applyW(z, lambda);
    const Scalar d = lambda[0] - lambda.tail(3).norm();
    EXPECT_NEAR(cone.centrality(s, z), d * d, 1e-10 * d * d) << "trial " << t;
  }
  s << 3.0, 0.0, 0.0, 0.0;  // central point: lambda o lambda = 9 e
  EXPECT_NEAR(cone.centrality(s, s), 9.0, 1e-13);
  z << 1.0, 1.0, 0.0, 0.0;  // boundary
  EXPECT_LE(cone.centrality(s, z), 0.0);
}

TEST(IpmSafeguards, NonnegativeCentralityIsSmallestProduct) {
  NonnegativeCone cone(3);
  Vec s(3), z(3);
  s << 1.0, 2.0, 3.0;
  z << 4.0, 0.25, 1.0;
  EXPECT_DOUBLE_EQ(cone.centrality(s, z), 0.5);
}

TEST(IpmSafeguards, SocInverseProductIsScaleInvariant) {
  SecondOrderCone cone(3);
  Vec u(3), w(3), x(3), back(3);
  u << 2.0, 1.0, 0.5;
  w << 0.3, -0.2, 0.7;
  for (Scalar scale : {1.0, 1e-8, 1e-12}) {
    const Vec us = scale * u, ws = scale * w;
    cone.inverseProduct(us, ws, x);
    cone.product(us, x, back);
    EXPECT_LT((back - ws).norm(), 1e-12 * ws.norm()) << "scale " << scale;
  }
}

TEST(IpmSafeguards, NoFactorizationAfterConvergenceAndNoColdStartWhenWarm) {
  SmallSocp p;
  Solver solver;
  ASSERT_TRUE(solver.setup(p.P, p.q, p.A, p.b, p.spec));
  const Solution cold = solver.solve();
  expectReadmeSolution(cold);
  // Cold start plus one factorization per iteration, none in the converged one.
  EXPECT_EQ(cold.info.kkt_factorizations, cold.info.iterations);
  EXPECT_GT(cold.info.solve_time_s, 0.0);
  EXPECT_GT(cold.info.setup_time_s, 0.0);

  Vec q = p.q;
  q[2] *= 1.001;
  ASSERT_TRUE(solver.updateData(nullptr, &q, nullptr, nullptr));
  const Solution warm = solver.solve();
  ASSERT_EQ(warm.status, Status::Solved);
  EXPECT_EQ(warm.info.kkt_factorizations, warm.info.iterations - 1);  // warm point used directly
}

TEST(IpmSafeguards, SetSettingsAfterSetupKeepsEquilibrationConsistent) {
  SmallSocp p;
  Solver solver;
  ASSERT_TRUE(solver.setup(p.P, p.q, p.A, p.b, p.spec));
  Settings s = solver.settings();
  s.equilibrate = false;             // setup-time field: takes effect at the next setup()
  s.regularization.static_P = 1e-7;  // runtime field: takes effect at the next solve()
  solver.setSettings(s);
  expectReadmeSolution(solver.solve());
}

TEST(IpmSafeguards, ValidateInputsRejectsNonFiniteData) {
  SmallSocp p;
  p.q[0] = std::numeric_limits<Scalar>::quiet_NaN();
  Solver solver;
  EXPECT_FALSE(solver.setup(p.P, p.q, p.A, p.b, p.spec));
  Settings s;
  s.validate_inputs = false;
  EXPECT_TRUE(solver.setup(p.P, p.q, p.A, p.b, p.spec, s));
}
