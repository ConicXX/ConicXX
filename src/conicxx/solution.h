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
  /// Two consecutive steps below Settings::min_terminate_step_length, or residuals growing
  /// (Clarabel's progress check); best iterate reported.
  InsufficientProgress,
  /// The best iterate meets the reduced tolerances after MaxIterations/MaxTime/
  /// InsufficientProgress; the best iterate is reported.
  AlmostSolved,
  AlmostPrimalInfeasible,
  AlmostDualInfeasible,
  /// Settings::time_limit reached; best iterate reported.
  MaxTime,
};

const char* toString(Status status);

struct Info {
  int iterations = 0;
  Scalar primal_residual = 0;  ///< unscaled, relative, infinity norm
  Scalar dual_residual = 0;    ///< unscaled, relative, infinity norm
  Scalar duality_gap = 0;
  Scalar mu = 0;
  Scalar setup_time_s = 0;  ///< duration of the last setup()
  Scalar solve_time_s = 0;  ///< duration of this solve()

  /// ||rhs - K x||_inf / ||rhs||_inf of the last KKT solve, against the exact KKT matrix.
  Scalar kkt_refinement_residual = 0;
  /// Some KKT solve of this solve() left the equality (zero-cone) rows unsatisfied: redundant or
  /// inconsistent equality constraints.
  bool equality_rank_deficient = false;
  int kkt_factorizations = 0;      ///< numeric LDL^T factorizations in this solve()
  int kkt_dynamic_reg_pivots = 0;  ///< pivots replaced by dynamic regularization, summed
  int kkt_refinement_steps = 0;    ///< iterative-refinement steps, summed over all KKT solves
  /// The solve started from the warm start (otherwise from the cold start).
  bool warm_start_used = false;
  /// Merit of the warm starting point and of the cold start it was compared with (the cached one
  /// of an earlier solve() since setup() if no cold start was computed); 0 if not evaluated.
  Scalar warm_start_merit = 0;
  Scalar cold_start_merit = 0;

  /// max(primal_residual, dual_residual, min(absolute gap, relative gap)) of the reported iterate.
  Scalar merit = 0;
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
