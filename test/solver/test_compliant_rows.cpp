// Phase 12: compliant zero-cone rows. The compact form (P, A_E with compliance C) must reproduce
// the augmented form with explicit variables pi (P_aug = blkdiag(P, C), rows [A_E, C]), whose
// solution has pi = -z_E.

#include "conicxx/solver.h"

#include <gtest/gtest.h>

#include <random>

#include "test_helpers.h"

using namespace conicxx;

namespace {

// Two "bodies" with diagonal mass, nE bilateral rows and three 3D friction cones.
struct Problem {
  Index n = 12, nE = 4, nC = 3;
  SparseMat M, AE, AC;
  Vec q, bE, bC;
  Vec compliance;
};

Problem makeProblem(unsigned seed, Scalar compliance_scale) {
  Problem p;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Scalar> u(-1.0, 1.0), pos(0.5, 2.0);
  std::vector<Triplet> m, e, c;
  for (Index i = 0; i < p.n; ++i) m.emplace_back(i, i, pos(rng));
  for (Index r = 0; r < p.nE; ++r) {
    for (Index j = 0; j < p.n; ++j) {
      if ((r + j) % 3 != 0) e.emplace_back(r, j, u(rng));
    }
  }
  for (Index r = 0; r < 3 * p.nC; ++r) {
    for (Index j = 0; j < p.n; ++j) {
      if ((2 * r + j) % 4 == 0) c.emplace_back(r, j, u(rng));
    }
  }
  p.M = testutil::makeSparse(p.n, p.n, m);
  p.AE = testutil::makeSparse(p.nE, p.n, e);
  p.AC = testutil::makeSparse(3 * p.nC, p.n, c);
  p.q = Vec::NullaryExpr(p.n, [&] { return 3.0 * u(rng); });
  p.bE = Vec::NullaryExpr(p.nE, [&] { return u(rng); });
  p.bC = Vec::Zero(3 * p.nC);
  for (Index k = 0; k < p.nC; ++k) p.bC[3 * k] = 0.1;  // x = 0 strictly feasible for the cones
  p.compliance = Vec::NullaryExpr(p.nE, [&] { return compliance_scale * pos(rng); });
  return p;
}

ConeSpec spec(const Problem& p) {
  ConeSpec s;
  s.zero_dim = p.nE;
  s.soc_dims.assign(static_cast<size_t>(p.nC), 3);
  return s;
}

// Tight tolerances: the comparisons below are about the formulation, not the stopping rule.
Settings tight() {
  Settings s;
  s.tol_feas = s.tol_gap_abs = s.tol_gap_rel = 1e-10;
  return s;
}

Vec stack(const Vec& a, const Vec& b) {
  Vec v(a.size() + b.size());
  v << a, b;
  return v;
}

SparseMat stackRows(const SparseMat& top, const SparseMat& bottom) {
  std::vector<Triplet> t;
  for (Index c = 0; c < top.outerSize(); ++c)
    for (SparseMat::InnerIterator it(top, c); it; ++it) t.emplace_back(it.row(), c, it.value());
  for (Index c = 0; c < bottom.outerSize(); ++c)
    for (SparseMat::InnerIterator it(bottom, c); it; ++it) t.emplace_back(top.rows() + it.row(), c, it.value());
  return testutil::makeSparse(top.rows() + bottom.rows(), top.cols(), t);
}

// Augmented form: x = (u, pi), P = blkdiag(M, C), rows [A_E C; A_C 0].
const Solution& solveAugmented(Solver& solver, const Problem& p) {
  const Index n = p.n + p.nE;
  std::vector<Triplet> P, A;
  for (Index i = 0; i < p.n; ++i) P.emplace_back(i, i, p.M.coeff(i, i));
  for (Index i = 0; i < p.nE; ++i) P.emplace_back(p.n + i, p.n + i, p.compliance[i]);
  for (Index c = 0; c < p.AE.outerSize(); ++c)
    for (SparseMat::InnerIterator it(p.AE, c); it; ++it) A.emplace_back(it.row(), c, it.value());
  for (Index i = 0; i < p.nE; ++i) A.emplace_back(i, p.n + i, p.compliance[i]);
  for (Index c = 0; c < p.AC.outerSize(); ++c)
    for (SparseMat::InnerIterator it(p.AC, c); it; ++it) A.emplace_back(p.nE + it.row(), c, it.value());
  const SparseMat Pm = testutil::makeSparse(n, n, P);
  const SparseMat Am = testutil::makeSparse(p.nE + 3 * p.nC, n, A);
  const Vec q = stack(p.q, Vec::Zero(p.nE));
  EXPECT_TRUE(solver.setup(Pm, q, Am, stack(p.bE, p.bC), spec(p), tight()));
  return solver.solve();
}

const Solution& solveCompact(Solver& solver, const Problem& p, const Vec* compliance) {
  EXPECT_TRUE(solver.setup(p.M, p.q, stackRows(p.AE, p.AC), stack(p.bE, p.bC), spec(p), tight(), compliance));
  return solver.solve();
}

// KKT residual of a compact solution (x, s, z) for the augmented problem with pi = -z_E:
// stationarity M x + q + A'z, compliant rows A_E x - C z_E - b_E, cone rows A_C x + s_C - b_C,
// complementarity s'z, relative to the data scale.
Scalar kktResidual(const Problem& p, const Solution& sol) {
  const Vec zE = sol.z.head(p.nE), zC = sol.z.tail(3 * p.nC);
  const Vec stat = p.M * sol.x + p.q + p.AE.transpose() * zE + p.AC.transpose() * zC;
  const Vec eq = p.AE * sol.x - p.compliance.cwiseProduct(zE) - p.bE;
  const Vec cone = p.AC * sol.x + sol.s.tail(3 * p.nC) - p.bC;
  return std::max({stat.lpNorm<Eigen::Infinity>(), eq.lpNorm<Eigen::Infinity>(),
                   cone.lpNorm<Eigen::Infinity>(), std::abs(sol.s.dot(sol.z))});
}

}  // namespace

TEST(CompliantRows, CompactEqualsAugmented) {
  for (unsigned seed = 1; seed <= 5; ++seed) {
    const Problem p = makeProblem(seed, 0.5);
    Solver aug, cmp;
    const Solution& sa = solveAugmented(aug, p);
    const Solution& sc = solveCompact(cmp, p, &p.compliance);
    ASSERT_TRUE(sa.ok()) << toString(sa.status);
    ASSERT_TRUE(sc.ok()) << toString(sc.status);
    EXPECT_LT(kktResidual(p, sc), 1e-8) << "seed " << seed;
    // Both are solutions of the same strictly convex problem; at degenerate (apex) contacts an
    // IPM iterate is only accurate to about sqrt(mu), hence the looser comparison.
    testutil::expectVecNear(sc.x, sa.x.head(p.n), 1e-4);
    testutil::expectVecNear(sc.z, sa.z, 1e-4);
    // pi = -z_E in the augmented solution.
    testutil::expectVecNear(sa.x.tail(p.nE), -sa.z.head(p.nE), 1e-4);
    EXPECT_NEAR(sc.objective, sa.objective, 1e-6 * std::max(1.0, std::abs(sa.objective)));
    EXPECT_LE(std::abs(sc.info.iterations - sa.info.iterations), 2) << "seed " << seed;
  }
}

TEST(CompliantRows, ZeroComplianceIsEquality) {
  const Problem p = makeProblem(7, 1.0);
  Solver plain, zero, tiny;
  const Solution& s0 = solveCompact(plain, p, nullptr);
  const Vec c0 = Vec::Zero(p.nE);
  const Solution& s1 = solveCompact(zero, p, &c0);
  const Vec c_tiny = Vec::Constant(p.nE, 1e-12);
  const Solution& s2 = solveCompact(tiny, p, &c_tiny);
  ASSERT_TRUE(s0.ok() && s1.ok() && s2.ok());
  testutil::expectVecNear(s1.x, s0.x, 1e-12);
  EXPECT_EQ(s1.info.iterations, s0.info.iterations);
  testutil::expectVecNear(s2.x, s0.x, 1e-4);
  // Equality rows are satisfied up to the compliance term.
  const Vec r = p.AE * s2.x - p.bE;
  EXPECT_LT(r.lpNorm<Eigen::Infinity>(), 1e-7);
}

TEST(CompliantRows, UpdateDataChangesCompliance) {
  const Problem soft = makeProblem(11, 1.0);
  Problem stiff = soft;
  stiff.compliance *= 1e-3;
  Solver updated, fresh;
  ASSERT_TRUE(solveCompact(updated, soft, &soft.compliance).ok());
  ASSERT_TRUE(updated.updateData(nullptr, nullptr, nullptr, nullptr, &stiff.compliance));
  const Solution& su = updated.solve();
  const Solution& sf = solveCompact(fresh, stiff, &stiff.compliance);
  ASSERT_TRUE(su.ok() && sf.ok());
  EXPECT_LT(kktResidual(stiff, su), 1e-8);
  testutil::expectVecNear(su.x, sf.x, 1e-4);
  // Negative or wrongly sized compliance is rejected.
  Vec bad = stiff.compliance;
  bad[0] = -1.0;
  EXPECT_FALSE(updated.updateData(nullptr, nullptr, nullptr, nullptr, &bad));
  const Vec wrong = Vec::Ones(soft.nE + 1);
  EXPECT_FALSE(updated.updateData(nullptr, nullptr, nullptr, nullptr, &wrong));
}
