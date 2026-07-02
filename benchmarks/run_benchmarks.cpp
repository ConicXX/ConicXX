// ConicXX benchmark suite.
//
// Runs a fixed, deterministic set of problems drawn from the literature
// (ECOS's portfolio-optimization example, QOCO's group-lasso example) plus a
// domain-specific multi-contact friction-cone benchmark, and reports
// per-instance iteration counts, solve times, and residuals in a table
// styled after the reporting used in the QOCO/Clarabel papers (Problem |
// Size | Trial | Iterations | Runtime | ...).
//
// Because every instance is generated from a fixed seed (see
// problem_generators.cpp), the iteration counts reported here are exactly
// reproducible: if a future change to the solver alters them (or, worse,
// causes a previously-solvable instance to stop converging), that shows up
// as a diff in this program's output -- run it before/after a change and
// compare. `RunAllBenchmarks` also returns a nonzero process exit code if
// any instance fails to reach Status::Solved, which is a hard regression
// signal suitable for wiring into CI (this binary is also registered as a
// ctest test, see benchmarks/CMakeLists.txt).

#include <conicxx/solver.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

#include "problem_generators.h"

using namespace conicxx;
using namespace conicxx::bench;

namespace {

struct Result {
  std::string family, size_label;
  int trial = 0;
  Status status = Status::Unsolved;
  int iterations = 0;
  double solve_ms = 0.0;
  double primal_res = 0.0, dual_res = 0.0, gap = 0.0;

  double msPerIter() const { return solve_ms / static_cast<double>(std::max(iterations, 1)); }
};

Result runOne(const BenchProblem& p, int trial) {
  Result r;
  r.family = p.family;
  r.size_label = p.size_label;
  r.trial = trial;

  Solver solver;
  Settings settings;  // library defaults throughout, deliberately not tuned
                       // per-problem, so this also exercises the default
                       // configuration a user would start from.

  const auto t0 = std::chrono::steady_clock::now();
  if (!solver.setup(p.P, p.q, p.A, p.b, p.cone_spec, settings)) {
    r.status = Status::NumericalError;
    return r;
  }
  const Solution& sol = solver.solve();
  const auto t1 = std::chrono::steady_clock::now();

  r.status = sol.status;
  r.iterations = sol.info.iterations;
  r.solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  r.primal_res = sol.info.primal_residual;
  r.dual_res = sol.info.dual_residual;
  r.gap = sol.info.duality_gap;
  return r;
}

void printHeader() {
  std::printf("%-16s %-24s %5s %-15s %6s %10s %12s %10s %10s %10s\n", "Family", "Size", "Trial",
              "Status", "Iters", "Time(ms)", "Time/Iter(ms)", "PrimalRes", "DualRes", "Gap");
  std::printf("%s\n", std::string(130, '-').c_str());
}

void printRow(const Result& r) {
  std::printf("%-16s %-24s %5d %-15s %6d %10.3f %12.4f %10.2e %10.2e %10.2e\n", r.family.c_str(),
              r.size_label.c_str(), r.trial, toString(r.status), r.iterations, r.solve_ms,
              r.msPerIter(), r.primal_res, r.dual_res, r.gap);
}

void printSummary(const std::vector<Result>& group) {
  std::vector<int> iters;
  double sum_ms = 0.0;
  double worst_primal = 0.0, worst_dual = 0.0, worst_gap = 0.0;
  bool all_solved = true;
  for (const Result& r : group) {
    iters.push_back(r.iterations);
    sum_ms += r.solve_ms;
    worst_primal = std::max(worst_primal, r.primal_res);
    worst_dual = std::max(worst_dual, r.dual_res);
    worst_gap = std::max(worst_gap, std::abs(r.gap));
    all_solved = all_solved && (r.status == Status::Solved);
  }
  const auto [min_it, max_it] = std::minmax_element(iters.begin(), iters.end());
  const double mean_it =
      std::accumulate(iters.begin(), iters.end(), 0.0) / static_cast<double>(iters.size());

  std::printf("  -> %s: iters[min=%d,mean=%.1f,max=%d] mean_time=%.3fms worst[primal=%.2e "
              "dual=%.2e gap=%.2e]\n",
              all_solved ? "OK" : "**FAILED TO CONVERGE**", *min_it, mean_it, *max_it,
              sum_ms / static_cast<double>(group.size()), worst_primal, worst_dual, worst_gap);
}

/// Runs `trials` repetitions of `make(rng)` for each entry, printing rows
/// and a per-size summary. Appends every trial's convergence outcome to
/// `all_ok`.
template <typename MakeFn>
void runFamily(const std::string& family, const std::vector<Index>& sizes, int trials,
              MakeFn make, bool& all_ok) {
  for (Index size : sizes) {
    std::vector<Result> group;
    for (int trial = 0; trial < trials; ++trial) {
      std::mt19937 rng = makeRng(family, size, trial);
      const BenchProblem p = make(size, rng);
      Result r = runOne(p, trial);
      printRow(r);
      std::fflush(stdout);
      group.push_back(r);
      all_ok = all_ok && (r.status == Status::Solved);
    }
    printSummary(group);
  }
}

}  // namespace

int main() {
  printHeader();
  bool all_ok = true;
  const int kTrials = 3;

  runFamily(
      "RandomQP", {10, 50, 200}, kTrials, [](Index n, std::mt19937& rng) { return makeRandomQp(n, rng); },
      all_ok);

  runFamily(
      "RandomLP", {10, 50, 200}, kTrials, [](Index n, std::mt19937& rng) { return makeRandomLp(n, rng); },
      all_ok);

  runFamily(
      "RandomSOCP", {5, 20, 50}, kTrials,
      [](Index num_blocks, std::mt19937& rng) { return makeRandomSocp(num_blocks, 4, rng); }, all_ok);

  // Portfolio: from Domahidi/Chu/Boyd (ECOS, 2013 ECC), eq. (17). "Size" is
  // the asset count; factor count scales with it, mirroring the ECOS
  // Fig. 1 benchmark table (#assets, #factors).
  runFamily(
      "Portfolio", {20, 60, 120}, kTrials,
      [](Index num_assets, std::mt19937& rng) {
        return makePortfolio(num_assets, std::max<Index>(3, num_assets / 8), rng);
      },
      all_ok);

  // Group lasso: from Chari et al., "QOCO" (arXiv:2503.12658), sec. C.3.
  runFamily(
      "GroupLasso", {10, 30, 60}, kTrials,
      [](Index num_groups, std::mt19937& rng) {
        return makeGroupLasso(num_groups, 5, 8 * num_groups, rng);
      },
      all_ok);

  // Domain-specific: multi-contact Coulomb friction, the target application.
  runFamily(
      "FrictionChain", {4, 16, 64}, kTrials,
      [](Index num_contacts, std::mt19937& rng) { return makeFrictionChain(num_contacts, 0.7, rng); },
      all_ok);

  // Arbitrary-size scaling benchmark: sized after a real per-timestep
  // multibody friction-contact problem reported against this solver
  // (zero=4944, soc_blocks=66). 66 is included exactly as one data point;
  // the smaller/larger points show the scaling trend of KKT factorization
  // cost as the problem grows. Fewer trials than the other families since
  // instances get large -- keeps total runtime reasonable.
  runFamily(
      "FrictionChainXL", {8, 33, 66, 264}, /*trials=*/2,
      [](Index num_contacts, std::mt19937& rng) { return makeFrictionChainXL(num_contacts, 0.7, rng); },
      all_ok);

  std::printf("\n%s\n", all_ok ? "ALL BENCHMARKS CONVERGED (Status::Solved)."
                                : "REGRESSION: at least one benchmark instance did not "
                                  "converge -- see **FAILED TO CONVERGE** rows above.");
  return all_ok ? 0 : 1;
}
