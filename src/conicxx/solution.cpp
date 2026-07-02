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
  }
  return "Unknown";
}

}  // namespace conicxx
