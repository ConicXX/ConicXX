#include "conicxx/solver.h"

#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace conicxx;

TEST(SolveLp, SimpleVertexOptimum) {
  // min -x0 - 2 x1  s.t. x0+x1<=1, x0>=0, x1>=0
  // Optimum at the vertex (0,1) since x1 has the larger objective weight.
  const Index n = 2;
  SparseMat P(n, n);  // P == 0
  Vec q(2);
  q << -1.0, -2.0;

  SparseMat A = testutil::makeSparse(3, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, -1.0}, {2, 1, -1.0}});
  Vec b(3);
  b << 1.0, 0.0, 0.0;

  ConeSpec spec;
  spec.nonneg_dim = 3;

  Solver solver;
  ASSERT_TRUE(solver.setup(P, q, A, b, spec));
  const Solution& sol = solver.solve();

  ASSERT_TRUE(sol.ok()) << "status=" << toString(sol.status);
  Vec expected(2);
  expected << 0.0, 1.0;
  testutil::expectVecNear(sol.x, expected, 1e-6);
  EXPECT_NEAR(sol.objective, -2.0, 1e-6);

  Mat Pd = Mat::Zero(n, n);
  Mat Ad = Mat(A);
  testutil::expectKktOptimal(Pd, q, Ad, b, spec, sol, 1e-6);
}
