#include "conicxx/solver.h"

#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace conicxx;

TEST(Infeasible, PrimalInfeasibleContradictoryBounds) {
  // Single variable x with x >= 1 and x <= 0 simultaneously: no feasible x.
  const Index n = 1;
  SparseMat P(n, n);
  Vec q = Vec::Zero(1);

  // -x + s0 = -1, s0 >= 0  =>  x >= 1
  //  x + s1 =  0, s1 >= 0  =>  x <= 0
  SparseMat A = testutil::makeSparse(2, 1, {{0, 0, -1.0}, {1, 0, 1.0}});
  Vec b(2);
  b << -1.0, 0.0;

  ConeSpec spec;
  spec.nonneg_dim = 2;

  Solver solver;
  ASSERT_TRUE(solver.setup(P, q, A, b, spec));
  const Solution& sol = solver.solve();

  EXPECT_EQ(sol.status, Status::PrimalInfeasible) << "status=" << toString(sol.status);
}

TEST(Infeasible, DualInfeasibleUnboundedBelow) {
  // min -x  s.t. x >= 0: unbounded below as x -> infinity, so the dual is
  // infeasible.
  const Index n = 1;
  SparseMat P(n, n);
  Vec q(1);
  q << -1.0;

  SparseMat A = testutil::makeSparse(1, 1, {{0, 0, -1.0}});  // -x + s = 0, s >= 0
  Vec b = Vec::Zero(1);

  ConeSpec spec;
  spec.nonneg_dim = 1;

  Solver solver;
  ASSERT_TRUE(solver.setup(P, q, A, b, spec));
  const Solution& sol = solver.solve();

  EXPECT_EQ(sol.status, Status::DualInfeasible) << "status=" << toString(sol.status);
}
