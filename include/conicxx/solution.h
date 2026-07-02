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
