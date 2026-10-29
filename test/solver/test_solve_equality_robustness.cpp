// Phase 2 accept criteria (CONICXX_AGENT_TASKS.md): "never perturb equality constraints" checked
// end to end through the public Solver API, plus the KktSystem-level regularization tests in
// test/kkt/test_kkt_system.cpp.

#include "conicxx/solver.h"

#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace conicxx;

TEST(SolveEqualityRobustness, ThreeEqualitiesOneOrthantOneSocSatisfyEqualitiesTightly) {
  // 4 variables. 3 equalities (zero cone) pin x1, x2 and link x3+x4; one nonneg row bounds
  // x4 >= 0; one SOC block (acting only on the already-pinned x1, x2, so always strictly
  // interior) exercises a mixed cone set alongside the equality block. Small P on x3/x4 keeps
  // the QP strictly convex and bounded.
  const Index n = 4;
  SparseMat P = testutil::makeSparse(n, n, {{2, 2, 0.1}, {3, 3, 0.1}});
  Vec q = Vec::Zero(n);

  // Rows 0-2 (zero cone): x1 = 1, x2 = -1, x3 + x4 = 0.5.
  // Row 3 (nonneg): s = x4 >= 0, via A row -e4, b = 0.
  // Rows 4-6 (SOC, dim 3): s0 = 2 (constant), s1 = -x1, s2 = -x2 -- always strictly interior
  // since x1, x2 are pinned to (1, -1) and sqrt(1^2+(-1)^2) = sqrt(2) < 2.
  SparseMat A = testutil::makeSparse(
      7, n, {{0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}, {2, 3, 1.0}, {3, 3, -1.0}, {5, 0, 1.0},
             {6, 1, 1.0}});
  Vec b(7);
  b << 1.0, -1.0, 0.5, 0.0, 2.0, 0.0, 0.0;

  ConeSpec spec;
  spec.zero_dim = 3;
  spec.nonneg_dim = 1;
  spec.soc_dims = {3};

  Solver solver;
  ASSERT_TRUE(solver.setup(P, q, A, b, spec));
  const Solution& sol = solver.solve();
  ASSERT_TRUE(sol.ok()) << "status=" << toString(sol.status);

  Mat Pd = Mat(Mat(P).selfadjointView<Eigen::Upper>()), Ad = Mat(A);
  testutil::expectKktOptimal(Pd, q, Ad, b, spec, sol, 1e-6);

  // The equality-block-specific check the task asks for: ||A_E x - b_E||_inf <= 1e-10, tighter
  // and scoped to just the rows that must never be perturbed (unlike expectKktOptimal's overall
  // 2-norm feasibility check above, which covers every row).
  const Mat A_E = Ad.topRows(spec.zero_dim);
  const Vec b_E = b.head(spec.zero_dim);
  const Scalar eq_residual_inf = (A_E * sol.x - b_E).cwiseAbs().maxCoeff();
  EXPECT_LE(eq_residual_inf, 1e-10);

  // x4 >= 0 should bind at exactly 0 is NOT required by this problem (P's quadratic term on x4
  // pulls the optimum to x3 = x4 = 0.25, strictly interior) -- just confirm the expected optimum
  // directly, since the equality/SOC rows fully pin x1, x2 and the P-regularized objective pins
  // x3, x4 uniquely.
  Vec expected(4);
  expected << 1.0, -1.0, 0.25, 0.25;
  testutil::expectVecNear(sol.x, expected, 1e-5);
}

TEST(SolveEqualityRobustness, DuplicatedEqualityRowNeverReportsWrongSolved) {
  // A's two equality rows are exact duplicates -- rank-deficient A_E, same shape as
  // KktSystem.HandlesRankDeficientEqualityBlockViaRegularization but exercised through the full
  // Solver/IPM loop and its public Info diagnostics, not just a single KKT solve.
  const Index n = 2;
  SparseMat P = testutil::makeSparse(n, n, {{0, 0, 1.0}, {1, 1, 1.0}});
  Vec q = Vec::Zero(n);
  SparseMat A = testutil::makeSparse(2, n, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});
  Vec b(2);
  b << 1.0, 1.0;

  ConeSpec spec;
  spec.zero_dim = 2;

  Solver solver;
  ASSERT_TRUE(solver.setup(P, q, A, b, spec));
  const Solution& sol = solver.solve();

  // Never a "wrong Solved": if it reports Solved, the equality residual must actually be tight.
  if (sol.status == Status::Solved) {
    const Mat Ad = Mat(A);
    const Scalar eq_residual_inf = (Ad * sol.x - b).cwiseAbs().maxCoeff();
    EXPECT_LE(eq_residual_inf, 1e-6);
  } else {
    // Otherwise it must be a clear, non-silent failure with the diagnostic flag set -- not some
    // other unrelated status.
    EXPECT_EQ(sol.status, Status::NumericalError) << "status=" << toString(sol.status);
    EXPECT_TRUE(sol.info.equality_rank_deficient);
  }
}
