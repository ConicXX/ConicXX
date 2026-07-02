#pragma once

#include "conicxx/types.h"

namespace conicxx {

/// Solver configuration. A plain aggregate so it is cheap to copy and easy
/// to construct with designated-initializer-style usage.
struct Settings {
  // --- Termination tolerances ---
  Scalar tol_feas = 1e-8;      ///< tolerance on scaled primal/dual residual norms
  Scalar tol_gap_abs = 1e-8;   ///< absolute duality gap tolerance
  Scalar tol_gap_rel = 1e-8;   ///< relative duality gap tolerance
  Scalar tol_infeas = 1e-7;    ///< tolerance for infeasibility certificate detection
  int max_iter = 200;

  // --- KKT regularization (Vanderbei quasi-definite construction) ---
  Scalar static_reg_P = 1e-8;   ///< added once to (1,1) block diagonal (P)
  Scalar static_reg_A = 1e-8;   ///< added once to (2,2) block diagonal (-Hs)
  Scalar dynamic_reg_eps = 1e-14;  ///< pivot-magnitude threshold triggering a bump
  Scalar dynamic_reg_delta = 1e-7; ///< bumped pivot magnitude

  // --- Iterative refinement on each KKT solve ---
  int refine_max_iter = 3;
  Scalar refine_tol = 1e-12;

  // --- Step length ---
  Scalar max_step_fraction = 0.99;  ///< fraction-to-boundary safety factor

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
};

}  // namespace conicxx
