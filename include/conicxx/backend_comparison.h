#pragma once

#include <vector>

#include "conicxx/cone_spec.h"
#include "conicxx/settings.h"
#include "conicxx/solution.h"
#include "conicxx/types.h"

namespace conicxx {

struct BackendComparisonResult {
  LinearSolverBackend backend = LinearSolverBackend::Eigen;
  Status status = Status::Unsolved;
  Info info;
  double wall_time_ms = 0;  ///< setup() + solve(), wall-clock (Info::*_time_s is not yet wired up)
};

/// Runs the same problem through all of Settings::linear_solver's backends (Eigen, Qdldl,
/// RegularizedLdlt), each from a fresh Solver instance with otherwise-identical settings, and
/// returns one result per backend in that order. Meant for comparing backends on a specific
/// hard problem instance -- e.g. capture (P, q, A, b, cone_spec) at whichever simulation
/// timestep is failing to converge (or converging slowly) and pass them here directly, the
/// same objects you'd otherwise hand to Solver::setup(). `settings.linear_solver` is
/// overridden per run; every other field (tolerances, regularization, max_iter, ...) is shared
/// across all three so the comparison isolates the backend choice.
std::vector<BackendComparisonResult> compareLinearSolverBackends(
    const SparseMat& P, const Vec& q, const SparseMat& A, const Vec& b,
    const ConeSpec& cone_spec, Settings settings = Settings{});

/// Prints compareLinearSolverBackends()'s results as an aligned table to stdout.
void printBackendComparison(const std::vector<BackendComparisonResult>& results);

}  // namespace conicxx
