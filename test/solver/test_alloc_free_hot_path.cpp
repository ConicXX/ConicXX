#include "conicxx/solver.h"

#include <atomic>
#include <gtest/gtest.h>

#include "test_helpers.h"

// T7.1's accept criterion: after the first solve() call has sized every scratch buffer, a
// second solve() on the same Solver must not touch the heap at all. ASan's malloc/free hooks
// (https://github.com/google/sanitizers/wiki/AddressSanitizerMallocHook) see every allocation
// the process makes, including ones inside Eigen/QDLDL that never go through operator new
// directly (Eigen's own allocator generally calls std::malloc, not `new`, so overriding
// operator new here would silently miss those and pass even with real per-iteration
// allocations still present) -- this only compiles/runs in an ASan build, matching how this
// project already verifies every phase (see CONICXX_AGENT_TASKS.md's "clean under
// -fsanitize=address,undefined" verification note on every prior phase).
#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
#define CONICXX_ALLOC_HOOK_TEST_ENABLED 1
#endif

#ifdef CONICXX_ALLOC_HOOK_TEST_ENABLED

namespace {
std::atomic<bool> g_tracking{false};
std::atomic<long> g_alloc_count{0};
}  // namespace

extern "C" void __sanitizer_malloc_hook(const volatile void*, size_t) {
  if (g_tracking.load(std::memory_order_relaxed)) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
}
extern "C" void __sanitizer_free_hook(const volatile void*) {}

using namespace conicxx;

// A friction-cone SOC projection, the same shape as
// SolveFrictionQpSoc.ProjectsDesiredForceOntoFrictionCone -- takes several genuine IPM
// iterations (affine + combined step, KKT solve, metrics, step-length safeguard all exercised
// at least once), not a trivial single-iteration solve.
TEST(AllocFreeHotPath, SolveAllocatesNothingOnHeapAfterFirstCall) {
  const Index n = 3;
  SparseMat P = testutil::makeSparse(n, n, {{0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}});

  Vec f_des(3);
  f_des << 1.0, 3.0, 4.0;
  Vec q = -f_des;

  SparseMat A = testutil::makeSparse(3, 3, {{0, 0, -1.0}, {1, 1, -1.0}, {2, 2, -1.0}});
  Vec b = Vec::Zero(3);

  ConeSpec spec;
  spec.soc_dims = {3};

  Solver solver;
  Settings settings;
  // Force every solve() call to run the same full iteration sequence from the same cold start,
  // rather than a warm-started (and possibly 1-iteration) one -- the point is to exercise
  // computeAffineStep()/computeCombinedStep()/safeguardedStepLength() repeatedly, not just
  // whatever a warm start happens to need.
  settings.warm_start = false;
  ASSERT_TRUE(solver.setup(P, q, A, b, spec, settings));

  // First call: sizing every scratch buffer (kkt_rhs_, kkt_sol_, the KKT backends'
  // value_scatter_/solve_scratch_, etc.) is allowed to touch the heap.
  const Solution& sol1 = solver.solve();
  ASSERT_TRUE(sol1.ok()) << "status=" << toString(sol1.status);
  ASSERT_GT(sol1.info.iterations, 1) << "problem converges in too few iterations to exercise the "
                                        "hot path meaningfully -- pick a harder instance";

  g_alloc_count.store(0, std::memory_order_relaxed);
  g_tracking.store(true, std::memory_order_relaxed);
  const Solution& sol2 = solver.solve();
  g_tracking.store(false, std::memory_order_relaxed);

  ASSERT_TRUE(sol2.ok()) << "status=" << toString(sol2.status);
  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0)
      << "solve() allocated on the heap after the first call (T7.1 regression)";
}

#endif  // CONICXX_ALLOC_HOOK_TEST_ENABLED
