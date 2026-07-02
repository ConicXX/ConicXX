#include "conicxx/solver.h"

#include <gtest/gtest.h>

#include <cmath>

#include "test_helpers.h"

using namespace conicxx;

TEST(SolveSocp, MinimizeFirstCoordinateOverCone) {
  // min x0  s.t. x in Q^3 (x0 >= ||(x1,x2)||), x1 = 1.
  // Minimal x0 = ||(1, x2)|| minimized over x2 => x2=0, x0=1.
  const Index n = 3;
  SparseMat P(n, n);
  Vec q(3);
  q << 1.0, 0.0, 0.0;

  // Row 0: equality x1 = 1 (zero cone). Rows 1-3: -x + s = 0, s in Q^3.
  SparseMat A =
      testutil::makeSparse(4, 3, {{0, 1, 1.0}, {1, 0, -1.0}, {2, 1, -1.0}, {3, 2, -1.0}});
  Vec b(4);
  b << 1.0, 0.0, 0.0, 0.0;

  ConeSpec spec;
  spec.zero_dim = 1;
  spec.soc_dims = {3};

  Solver solver;
  ASSERT_TRUE(solver.setup(P, q, A, b, spec));
  const Solution& sol = solver.solve();

  ASSERT_TRUE(sol.ok()) << "status=" << toString(sol.status);
  Vec expected(3);
  expected << 1.0, 1.0, 0.0;
  testutil::expectVecNear(sol.x, expected, 1e-5);
  EXPECT_NEAR(sol.objective, 1.0, 1e-5);

  Mat Pd = Mat::Zero(n, n);
  Mat Ad = Mat(A);
  testutil::expectKktOptimal(Pd, q, Ad, b, spec, sol, 1e-5);
}
