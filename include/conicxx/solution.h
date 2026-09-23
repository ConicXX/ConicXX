#pragma once

#include <string>

#include "conicxx/types.h"

namespace conicxx {

enum class Status {
  Unsolved,
  Solved,
  PrimalInfeasible,
  DualInfeasible,
  MaxIterations,
  NumericalError,
  /// The step-length safeguard (see Settings::centrality_theta/min_terminate_step_length)
  /// backtracked to a step below min_terminate_step_length for two consecutive iterations --
  /// returned instead of continuing to grind on iterations that aren't making real progress, or
  /// silently accepting a step too close to the boundary to trust. Reports the current iterate
  /// (full best-iterate tracking across the whole run is Phase 4/T4.3's job, not implemented
  /// here -- see CONICXX_AGENT_TASKS.md).
  InsufficientProgress,
};

const char* toString(Status status);

struct Info {
  int iterations = 0;
  Scalar primal_residual = 0;
  Scalar dual_residual = 0;
  Scalar duality_gap = 0;
  Scalar mu = 0;
  Scalar setup_time_s = 0;
  Scalar solve_time_s = 0;

  /// Relative residual of the last KKT solve's iterative refinement (against K_exact -- see
  /// KktSystem), i.e. how well the *unregularized* system was actually solved.
  Scalar kkt_refinement_residual = 0;

  /// Set when a zero-cone (equality) pivot stayed bad (wrong sign or too small) even after
  /// KktSystem's one-shot factor-only preconditioner (Settings::regularization.static_zero_factor_only),
  /// with dynamic_on_zero_rows left at its default (false) -- a signal of redundant/hyperstatic
  /// equality constraints (rank-deficient A restricted to the zero-cone rows), not corrected or
  /// perturbed away silently.
  bool equality_rank_deficient = false;
};

struct Solution {
  Status status = Status::Unsolved;
  Vec x;  ///< primal variables
  Vec s;  ///< slack variables, s in K
  Vec z;  ///< dual variables, z in K*
  Scalar objective = 0;
  Info info;

  bool ok() const { return status == Status::Solved; }
};

}  // namespace conicxx
