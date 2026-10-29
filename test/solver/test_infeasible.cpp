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

  ASSERT_EQ(sol.status, Status::PrimalInfeasible) << "status=" << toString(sol.status);

  // T4.2 certificate check: z in K* (nonneg here, so z >= 0), b'z < 0, ||A'z||_inf small relative
  // to -b'z, and the returned certificate is normalized so -b'z == 1 exactly.
  Mat Ad = Mat(A);
  const Scalar bz = b.dot(sol.z);
  EXPECT_LT(bz, 0.0);
  EXPECT_NEAR(-bz, 1.0, 1e-6);
  EXPECT_GE(sol.z.minCoeff(), -1e-8) << "z not in the dual cone (nonneg)";
  const Scalar Atz_inf = (Ad.transpose() * sol.z).cwiseAbs().maxCoeff();
  EXPECT_LT(Atz_inf, 1e-4);
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

  ASSERT_EQ(sol.status, Status::DualInfeasible) << "status=" << toString(sol.status);

  // T4.2 certificate check: q'x < 0, ||Px||_inf and ||Ax+s||_inf small relative to -q'x, and the
  // returned certificate is normalized so -q'x == 1 exactly.
  Mat Pd = Mat::Zero(n, n);
  Mat Ad = Mat(A);
  const Scalar qx = q.dot(sol.x);
  EXPECT_LT(qx, 0.0);
  EXPECT_NEAR(-qx, 1.0, 1e-6);
  const Scalar Px_inf = (Pd * sol.x).cwiseAbs().maxCoeff();
  const Scalar Axs_inf = (Ad * sol.x + sol.s).cwiseAbs().maxCoeff();
  EXPECT_LT(Px_inf, 1e-4);
  EXPECT_LT(Axs_inf, 1e-4);
}

TEST(Infeasible, NearInfeasibleFrictionConeContradictoryEquality) {
  // A single 3D Coulomb friction cone (mu*fn >= ||ft||, which forces fn >= 0) combined with an
  // equality constraint pinning fn to a small *negative* value -- a direct, unambiguous but
  // numerically "near" (small-magnitude) contradiction with the cone's own fn >= 0 requirement.
  // Exercises T4.2 in the friction-cone domain this solver targets, not just a trivial 1-D LP.
  const Index n = 3;  // (fn, ft1, ft2)
  SparseMat P(n, n);
  Vec q = Vec::Zero(n);

  // Row 0 (zero cone): fn = -1e-3 (infeasible: the friction cone below requires fn >= 0).
  // Rows 1-3 (SOC): mu*fn >= ||ft||.
  SparseMat A = testutil::makeSparse(
      4, n, {{0, 0, 1.0}, {1, 0, -0.7}, {2, 1, -1.0}, {3, 2, -1.0}});
  Vec b(4);
  b << -1e-3, 0.0, 0.0, 0.0;

  ConeSpec spec;
  spec.zero_dim = 1;
  spec.soc_dims = {3};

  Solver solver;
  ASSERT_TRUE(solver.setup(P, q, A, b, spec));
  const Solution& sol = solver.solve();

  // A small-magnitude contradiction may legitimately be reported as AlmostPrimalInfeasible
  // (T4.3's reduced tolerances) rather than the full PrimalInfeasible -- either is an honest
  // answer; a Solved status here would be the "false certificate" the task's own accept
  // criteria explicitly rules out.
  EXPECT_TRUE(sol.status == Status::PrimalInfeasible || sol.status == Status::AlmostPrimalInfeasible)
      << "status=" << toString(sol.status);

  Mat Ad = Mat(A);
  const Scalar bz = b.dot(sol.z);
  EXPECT_LT(bz, 0.0);
}
