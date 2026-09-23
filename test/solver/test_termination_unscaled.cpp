// T4.1 (CONICXX_AGENT_TASKS.md Phase 4): Info::primal_residual/dual_residual are computed
// internally from the *equilibrated* representation, algebraically undoing the Ruiz scaling
// (see docs/design.md "Termination and infeasibility" for the derivation) rather than by
// reconstructing an unscaled problem every iteration. This test is the actual safety net for
// that derivation: it reconstructs the unscaled residuals directly from the public API (the
// original P/A/q/b passed to setup(), and the returned Solution's x/s/z) on a deliberately
// badly-scaled problem (so Ruiz equilibration does real, non-trivial work -- d/e/c far from 1),
// and checks the two agree -- independent of whether the solve fully converged, so this isn't
// just "both are ~0 anyway".

#include "conicxx/solver.h"

#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace conicxx;

namespace {

// A badly-scaled QP: variables and constraints span ~1e-4 to ~1e5, forcing Ruiz equilibration to
// compute genuinely non-identity d/e/c (verified structurally already by
// Equilibration.SecondOrderConeBlockGetsUniformScale et al.; this test is about whether the
// *solver's* unscaling of the termination residuals, not the equilibration math itself, is
// correct).
struct BadlyScaledProblem {
  SparseMat P, A;
  Vec q, b;
  ConeSpec spec;
};

BadlyScaledProblem makeBadlyScaledQp() {
  BadlyScaledProblem p;
  const Index n = 3;
  p.P = testutil::makeSparse(n, n, {{0, 0, 1e4}, {1, 1, 1e-3}, {2, 2, 1.0}});
  p.q = Vec(n);
  p.q << -1e4, 5e-3, -0.5;

  // A box constraint x <= ub (nonneg slack) on each variable, wildly different scales.
  p.A = testutil::makeSparse(3, n, {{0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}});
  p.b = Vec(3);
  p.b << 2e5, 1e-4, 10.0;
  p.spec.nonneg_dim = 3;
  return p;
}

// ||v||_inf, or 0 for an empty vector.
Scalar infNorm(const Vec& v) { return v.size() == 0 ? 0.0 : v.cwiseAbs().maxCoeff(); }

}  // namespace

TEST(TerminationUnscaled, PrimalDualResidualMatchDirectUnscaledReconstruction) {
  BadlyScaledProblem p = makeBadlyScaledQp();
  Mat Pd = Mat(Mat(p.P).selfadjointView<Eigen::Upper>());
  Mat Ad = Mat(p.A);

  for (int max_iter : {3, 200}) {  // an early-stopped (unconverged) point, and a converged one
    Settings settings;
    settings.max_iter = max_iter;
    Solver solver;
    ASSERT_TRUE(solver.setup(p.P, p.q, p.A, p.b, p.spec, settings));
    const Solution& sol = solver.solve();
    ASSERT_TRUE(sol.x.allFinite() && sol.s.allFinite() && sol.z.allFinite())
        << "max_iter=" << max_iter << " status=" << toString(sol.status);

    // Direct reconstruction from the public API alone: original (unscaled) P/A/q/b and the
    // returned (already unscaled) x/s/z.
    const Vec Ax_s_minus_b = Ad * sol.x + sol.s - p.b;
    const Vec Px_Atz_q = Pd * sol.x + Ad.transpose() * sol.z + p.q;

    const Scalar normb = infNorm(p.b), normq = infNorm(p.q);
    const Scalar normx = infNorm(sol.x), norms = infNorm(sol.s), normz = infNorm(sol.z);

    const Scalar res_primal_direct =
        infNorm(Ax_s_minus_b) / std::max(Scalar(1.0), normb + normx + norms);
    const Scalar res_dual_direct =
        infNorm(Px_Atz_q) / std::max(Scalar(1.0), normq + normx + normz);

    // Loose-ish tolerance: this compares two independently-derived numbers (one via the
    // equilibrated-internals formula, one via direct reconstruction), not the same computation
    // twice, so residual floating-point noise from the different paths is expected, but any
    // *formula* bug (wrong scaling factor) would show up as an error many orders of magnitude
    // larger than this, not a borderline near-miss.
    EXPECT_NEAR(sol.info.primal_residual, res_primal_direct,
                1e-9 + 1e-6 * std::max(sol.info.primal_residual, res_primal_direct))
        << "max_iter=" << max_iter;
    EXPECT_NEAR(sol.info.dual_residual, res_dual_direct,
                1e-9 + 1e-6 * std::max(sol.info.dual_residual, res_dual_direct))
        << "max_iter=" << max_iter;
  }
}
