#include "conicxx/backend_comparison.h"

#include <chrono>
#include <cstdio>
#include <string>

#include "conicxx/solver.h"

namespace conicxx {

namespace {
const char* backendName(LinearSolverBackend backend) {
  switch (backend) {
    case LinearSolverBackend::Eigen:
      return "Eigen";
    case LinearSolverBackend::Qdldl:
      return "Qdldl";
    case LinearSolverBackend::RegularizedLdlt:
      return "RegularizedLdlt";
  }
  return "?";
}
}  // namespace

std::vector<BackendComparisonResult> compareLinearSolverBackends(
    const SparseMat& P, const Vec& q, const SparseMat& A, const Vec& b, const ConeSpec& cone_spec,
    Settings settings) {
  std::vector<BackendComparisonResult> results;
  for (LinearSolverBackend backend : {LinearSolverBackend::Eigen, LinearSolverBackend::Qdldl,
                                      LinearSolverBackend::RegularizedLdlt}) {
    Settings backend_settings = settings;
    backend_settings.linear_solver = backend;

    BackendComparisonResult r;
    r.backend = backend;

    Solver solver;
    const auto t0 = std::chrono::steady_clock::now();
    if (!solver.setup(P, q, A, b, cone_spec, backend_settings)) {
      r.status = Status::NumericalError;
      results.push_back(r);
      continue;
    }
    const Solution& sol = solver.solve();
    const auto t1 = std::chrono::steady_clock::now();

    r.status = sol.status;
    r.info = sol.info;
    r.wall_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    results.push_back(r);
  }
  return results;
}

void printBackendComparison(const std::vector<BackendComparisonResult>& results) {
  std::printf("%-16s %-16s %6s %10s %12s %12s %12s\n", "Backend", "Status", "Iters", "Time(ms)",
              "PrimalRes", "DualRes", "Gap");
  std::printf("%s\n", std::string(90, '-').c_str());
  for (const auto& r : results) {
    std::printf("%-16s %-16s %6d %10.3f %12.3e %12.3e %12.3e\n", backendName(r.backend),
                toString(r.status), r.info.iterations, r.wall_time_ms, r.info.primal_residual,
                r.info.dual_residual, r.info.duality_gap);
  }
}

}  // namespace conicxx
