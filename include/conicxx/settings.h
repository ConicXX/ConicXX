#pragma once

#include <limits>

#include "conicxx/types.h"

namespace conicxx {

/// Which sparse LDL^T backend factorizes/solves the KKT system.
enum class LinearSolverBackend {
  Eigen,  ///< Eigen::SimplicialLDLT (always available)
  Qdldl,  ///< github.com/osqp/qdldl -- the backend QOCO/Clarabel use, if built with
          ///< CONICXX_WITH_QDLDL (falls back to Eigen with a one-time warning otherwise)
  RegularizedLdlt,  ///< Davis/ECOS-style true per-pivot dynamic regularization (a modified
                    ///< QDLDL numeric factorization loop), if built with CONICXX_WITH_QDLDL
                    ///< (falls back to Eigen with a one-time warning otherwise); corrects a
                    ///< bad pivot inline during elimination instead of KktSystem's outer
                    ///< refactorize-from-scratch retry loop
};

/// KKT regularization settings (Vanderbei quasi-definite construction), broken out per cone type
/// instead of per matrix side -- the whole point of Phase 2 is that equality (zero-cone) rows are
/// regularized differently (by default, not at all) from orthant/SOC rows, which the old flat
/// static_reg_P/static_reg_A pair couldn't express.
struct RegularizationSettings {
  // --- Static regularization: fixed amounts, added once to K_fact's diagonal (never K_exact's;
  // see KktSystem's K_exact/K_fact split). Signs are applied internally: + on the P (x) block,
  // - on the Hs (z) block, matching the KKT matrix's [P+.. A'; A -Hs-..] sign convention. ---
  Scalar static_P = 1e-8;       ///< (1,1) block diagonal (P)
  Scalar static_nonneg = 1e-8;  ///< nonnegative-orthant rows of the (2,2) block
  Scalar static_soc = 1e-8;     ///< second-order-cone rows of the (2,2) block
  Scalar static_zero = 0.0;     ///< equality (zero-cone) rows: exactly zero perturbation by default

  /// Additional static regularization, proportional to max|diag(K_exact)|, added on top of the
  /// fixed static_P/static_nonneg/static_soc amounts above (0 on zero rows, same as those). Keeps
  /// the fixed floor from becoming numerically irrelevant on a problem whose K is scaled far above
  /// or below O(1) (e.g. after equilibration failed to fully normalize an extreme instance).
  Scalar static_proportional = 2.2e-16 * 2.2e-16;

  /// Zero rows get 0 static regularization in K_exact (always) and static_zero in K_fact (above,
  /// 0 by default) -- but if that alone leaves K_fact's zero-row block unfactorizable (a genuinely
  /// singular pivot from the elimination order, not necessarily true rank deficiency), this is
  /// tried once, in K_fact only, as a preconditioner: refinement against K_exact removes its
  /// effect on the solution. If it's still not enough, KktSystem reports
  /// Info::equality_rank_deficient instead of silently accepting a bad factorization.
  Scalar static_zero_factor_only = 1e-10;

  // --- Dynamic regularization: an outer per-pivot-magnitude retry loop bumps these (10x per
  // attempt) only for the block a bad pivot belongs to, not uniformly across the whole matrix. ---
  Scalar dynamic_eps = 1e-13;    ///< pivot-magnitude threshold triggering a bump
  Scalar dynamic_delta = 2e-7;   ///< bumped pivot magnitude

  /// If false (default), a bad pivot on a zero (equality) row is never dynamically bumped -- it's
  /// treated as a rank-deficiency signal (see static_zero_factor_only above and
  /// Info::equality_rank_deficient) rather than a numerics problem to paper over. Set true to
  /// include zero rows in the same dynamic-regularization ladder as orthant/SOC rows instead.
  bool dynamic_on_zero_rows = false;
};

/// Solver configuration. A plain aggregate so it is cheap to copy and easy
/// to construct with designated-initializer-style usage.
struct Settings {
  // --- Termination tolerances (Phase 4, T4.1) ---
  // Evaluated on unequilibrated (real-units) quantities -- see docs/design.md "Termination and
  // infeasibility" for the exact formulas and how they're derived from the equilibrated
  // internal representation without reconstructing an unscaled problem every iteration.
  Scalar tol_feas = 1e-8;      ///< primal/dual feasibility residual tolerance (infinity norm)
  Scalar tol_gap_abs = 1e-8;   ///< absolute duality gap tolerance
  Scalar tol_gap_rel = 1e-8;   ///< relative duality gap tolerance

  /// kappa/tau must stay <= tol_ktratio.recip() * 1000 before infeasibility certificates are even
  /// considered (matches Clarabel's actual gate; `Solved` itself uses a fixed kappa/tau <= 1.0
  /// sanity bound, not tol_ktratio -- the task list's own paraphrase of this differs from
  /// Clarabel's real behavior, resolved in Clarabel's favor, see docs/design.md).
  Scalar tol_ktratio = 1e-6;

  Scalar tol_infeas_abs = 1e-7;  ///< absolute part of the infeasibility-certificate tolerance
  Scalar tol_infeas_rel = 1e-7;  ///< relative part (multiplies -b'z resp. -q'x)

  int max_iter = 200;
  Scalar time_limit = std::numeric_limits<Scalar>::infinity();  ///< seconds; Status::MaxTime

  /// "Almost" tolerances (T4.3): if the ordinary tolerances above aren't met but these looser
  /// ones are, the best iterate seen so far is returned with an Almost* status instead of
  /// MaxIterations/MaxTime/InsufficientProgress. Clarabel-like defaults.
  Scalar reduced_tol_feas = 1e-4;
  Scalar reduced_tol_gap_abs = 5e-5;
  Scalar reduced_tol_gap_rel = 5e-5;
  Scalar reduced_tol_infeas_abs = 5e-5;
  Scalar reduced_tol_infeas_rel = 5e-5;
  Scalar reduced_tol_ktratio = 1e-4;

  // --- KKT regularization (Vanderbei quasi-definite construction) ---
  RegularizationSettings regularization;

  // --- Iterative refinement on each KKT solve (against K_exact, the unregularized matrix -- see
  // KktSystem) ---
  int refine_max_iter = 10;
  Scalar refine_reltol = 1e-13;
  Scalar refine_abstol = 1e-12;
  Scalar refine_stop_ratio = 5;  ///< stop early if the residual doesn't shrink by this factor/step

  // --- Step length ---
  Scalar max_step_fraction = 0.99;  ///< fraction-to-boundary safety factor

  /// Centrality safeguard (Phase 3, T3.2): after a trial step, every cone block's
  /// min-eigenvalue(lambda o lambda) must be >= centrality_theta * mu (mu from *before* the
  /// step), or the step backtracks. lambda = W*z_trial, using the current iteration's already-
  /// computed NT scaling W. Catches steps that stay strictly interior (per cone.margin()) but
  /// land too close to the boundary to trust -- e.g. heading toward a cone's apex faster than mu
  /// is shrinking.
  Scalar centrality_theta = 1e-4;

  /// Backtracking factor for the step-length safeguard: alpha *= linesearch_backtrack each time
  /// a trial step fails the interior or centrality check.
  Scalar linesearch_backtrack = 0.8;

  /// If the safeguarded step length falls below this for two consecutive iterations, solve()
  /// returns Status::InsufficientProgress instead of continuing -- see the Status enum.
  Scalar min_terminate_step_length = 1e-4;

  // --- Equilibration (Ruiz scaling) ---
  bool equilibrate = true;
  int equilibrate_max_iter = 10;
  Scalar equilibrate_min_scale = 1e-4;
  Scalar equilibrate_max_scale = 1e4;

  // --- Warm start ---
  bool warm_start = true;  ///< reuse previous (x,s,z) to seed the next solve()

  // --- Diagnostics ---
  int verbose = 0;              ///< 0 = silent, 1 = summary, 2 = per-iteration
  bool record_timings = false;

  // --- Input validation ---
  bool validate_inputs = true;

  // --- Linear solver backend ---
  /// Defaults to Qdldl (github.com/osqp/qdldl, the same backend QOCO/Clarabel use) when built
  /// with CONICXX_WITH_QDLDL; falls back to Eigen with a one-time warning otherwise.
  LinearSolverBackend linear_solver = LinearSolverBackend::Qdldl;
};

}  // namespace conicxx
