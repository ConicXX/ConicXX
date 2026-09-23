#include "conicxx/solution.h"

namespace conicxx {

const char* toString(Status status) {
  switch (status) {
    case Status::Unsolved:
      return "Unsolved";
    case Status::Solved:
      return "Solved";
    case Status::PrimalInfeasible:
      return "PrimalInfeasible";
    case Status::DualInfeasible:
      return "DualInfeasible";
    case Status::MaxIterations:
      return "MaxIterations";
    case Status::NumericalError:
      return "NumericalError";
    case Status::InsufficientProgress:
      return "InsufficientProgress";
    case Status::AlmostSolved:
      return "AlmostSolved";
    case Status::AlmostPrimalInfeasible:
      return "AlmostPrimalInfeasible";
    case Status::AlmostDualInfeasible:
      return "AlmostDualInfeasible";
    case Status::MaxTime:
      return "MaxTime";
  }
  return "Unknown";
}

}  // namespace conicxx
