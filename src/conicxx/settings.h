#pragma once

#include <limits>

#include "conicxx/types.h"

namespace conicxx {

/// KKT regularization, see docs/design.md "KKT factorization". Static regularization is applied
/// to the pivots of the LDL^T factorization only; iterative refinement runs against the exact KKT
/// matrix, so it does not perturb the Newton directions, equality rows included.
struct RegularizationSettings {
  Scalar static_P = 1e-8;       ///< P (x) rows, added to the pivot
  Scalar static_nonneg = 1e-8;  ///< nonnegative-orthant rows, subtracted from the pivot
  Scalar static_soc = 1e-8;     ///< second-order-cone rows, subtracted from the pivot
  Scalar static_zero = 1e-8;    ///< equality (zero-cone) rows, subtracted from the pivot
  /// Added on every row, multiplied by max|diag(K)|, so the regularization keeps its relative
  /// size on badly scaled problems.
  Scalar static_proportional = 2.2e-16 * 2.2e-16;

  /// Dynamic regularization during the factorization: a pivot with D*sign < dynamic_eps is
  /// replaced by sign*dynamic_delta (sign = +1 for P rows, -1 for cone rows).
  bool dynamic_enable = true;
  Scalar dynamic_eps = 1e-13;
  Scalar dynamic_delta = 2e-7;
};

/// Solver configuration (plain aggregate).
struct Settings {
  // --- Termination, evaluated on unscaled quantities (docs/design.md "Termination") ---
  Scalar tol_feas = 1e-8;     ///< primal/dual residual tolerance (infinity norm)
  Scalar tol_gap_abs = 1e-8;  ///< absolute duality gap tolerance
  Scalar tol_gap_rel = 1e-8;  ///< relative duality gap tolerance
  /// Infeasibility certificates are checked only once kappa/tau > 1000 / tol_ktratio (as in
  /// Clarabel); Solved requires kappa/tau <= 1.
  Scalar tol_ktratio = 1e-6;
  Scalar tol_infeas_abs = 1e-7;  ///< absolute part of the certificate tolerance
  Scalar tol_infeas_rel = 1e-7;  ///< relative part (multiplies -b'z resp. -q'x)

  int max_iter = 200;
  Scalar time_limit = std::numeric_limits<Scalar>::infinity();  ///< seconds, Status::MaxTime

  /// On MaxIterations/MaxTime/InsufficientProgress the best iterate is reported with an Almost*
  /// status if it meets these reduced tolerances.
  Scalar reduced_tol_feas = 1e-4;
  Scalar reduced_tol_gap_abs = 5e-5;
  Scalar reduced_tol_gap_rel = 5e-5;
  Scalar reduced_tol_infeas_abs = 5e-5;
  Scalar reduced_tol_infeas_rel = 5e-5;
  Scalar reduced_tol_ktratio = 1e-4;

  RegularizationSettings regularization;

  // --- Iterative refinement of every KKT solve, against the exact KKT matrix ---
  int refine_max_iter = 10;
  Scalar refine_reltol = 1e-13;  ///< relative to ||rhs||_inf
  Scalar refine_abstol = 1e-12;
  Scalar refine_stop_ratio = 5;  ///< stop when a step shrinks the residual by less than this
  /// Refine the affine (predictor) direction too. It only sets the centering parameter and the
  /// second-order correction, so it is solved together with the constant right-hand side and,
  /// by default, not refined.
  bool refine_affine = false;

  // --- Step length ---
  Scalar max_step_fraction = 0.99;  ///< fraction to the boundary
  /// Centrality safeguard: a trial step is accepted only if every cone block and tau*kappa have
  /// centrality >= centrality_theta * mu_trial, measured exactly at the trial point (smallest
  /// eigenvalue of lambda o lambda for its NT scaling point); otherwise alpha *= linesearch_backtrack.
  Scalar centrality_theta = 1e-4;
  Scalar linesearch_backtrack = 0.8;
  /// Two consecutive steps shorter than this end the solve with InsufficientProgress.
  Scalar min_terminate_step_length = 1e-4;

  // --- Ruiz equilibration ---
  bool equilibrate = true;
  int equilibrate_max_iter = 10;
  Scalar equilibrate_min_scale = 1e-4;
  Scalar equilibrate_max_scale = 1e4;

  // --- Warm start ---
  bool warm_start = true;  ///< seed the next solve() with the previous solution
  /// Weight of the previous solution in the warm start: (x, s, z) = lambda * previous +
  /// (1 - lambda) * (0, e, e), tau = 1, kappa = 1 - lambda (Skajaa, Andersen, Ye 2013).
  Scalar warm_lambda = 0.99;

  // --- Diagnostics ---
  int verbose = 0;              ///< 0 silent, 1 summary, 2 one line per iteration (stdout)
  bool validate_inputs = true;  ///< setup()/updateData() reject non-finite data
};

}  // namespace conicxx
