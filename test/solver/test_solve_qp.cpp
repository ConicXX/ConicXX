#include "conicxx/solver.h"

#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace conicxx;

TEST(SolveQp, EqualityConstrainedClosedForm) {
  // min 0.5*(x0^2+x1^2)  s.t. x0+x1=1
  // Closed form (Lagrange multiplier): x0=x1=0.5.
  SparseMat P = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  Vec q = Vec::Zero(2);
  SparseMat A = testutil::makeSparse(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}});
  Vec b(1);
  b << 1.0;

  ConeSpec spec;
  spec.zero_dim = 1;

  Solver solver;
  ASSERT_TRUE(solver.setup(P, q, A, b, spec));
  const Solution& sol = solver.solve();

  ASSERT_TRUE(sol.ok()) << "status=" << toString(sol.status);
  Vec expected(2);
  expected << 0.5, 0.5;
  testutil::expectVecNear(sol.x, expected, 1e-6);
  EXPECT_NEAR(sol.objective, 0.25, 1e-6);

  Mat Pd = Mat(Mat(P).selfadjointView<Eigen::Upper>());
  Mat Ad = Mat(A);
  testutil::expectKktOptimal(Pd, q, Ad, b, spec, sol, 1e-6);
}

TEST(SolveQp, BoxConstraintActiveAtOneBound) {
  // min 0.5*(x0+1)^2 + 0.5*(x1-2)^2  s.t. x0>=0, x1>=0
  // Unconstrained optimum (-1,2) violates x0>=0; constrained optimum (0,2).
  SparseMat P = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  Vec q(2);
  q << 1.0, -2.0;
  // -x + s = 0, s >= 0  =>  s = x >= 0
  SparseMat A = testutil::makeSparse(2, 2, {{0, 0, -1.0}, {1, 1, -1.0}});
  Vec b = Vec::Zero(2);

  ConeSpec spec;
  spec.nonneg_dim = 2;

  Solver solver;
  ASSERT_TRUE(solver.setup(P, q, A, b, spec));
  const Solution& sol = solver.solve();

  ASSERT_TRUE(sol.ok()) << "status=" << toString(sol.status);
  Vec expected(2);
  expected << 0.0, 2.0;
  testutil::expectVecNear(sol.x, expected, 1e-6);
  // Solver reports 0.5*x'Px + q'x (no constant offset); at x=(0,2) that is
  // 0.5*(0+4) + (1*0 - 2*2) = 2 - 4 = -2 (the shifted-form objective
  // 0.5*(x0+1)^2+0.5*(x1-2)^2 evaluated there is 0.5, differing by the
  // dropped constant +2.5).
  EXPECT_NEAR(sol.objective, -2.0, 1e-6);

  Mat Pd = Mat(Mat(P).selfadjointView<Eigen::Upper>());
  Mat Ad = Mat(A);
  testutil::expectKktOptimal(Pd, q, Ad, b, spec, sol, 1e-6);
}
