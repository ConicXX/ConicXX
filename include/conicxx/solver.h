#pragma once

#include <memory>

#include "conicxx/cone_spec.h"
#include "conicxx/settings.h"
#include "conicxx/solution.h"
#include "conicxx/types.h"

namespace conicxx {

namespace detail {
class SolverImpl;
}

/// Top-level interior-point solver for
///
///   min  0.5 x'Px + q'x
///   s.t. Ax + s = b,  s in K = Zero x Nonnegative x SecondOrderCone x ...
///
/// P must be symmetric positive semidefinite, supplied as an upper-triangular
/// sparse matrix. Typical usage in a per-timestep simulation loop:
///
///   conicxx::Solver solver;
///   solver.setup(P, q, A, b, cone_spec, settings);   // once: builds KKT sparsity pattern
///   while (simulating) {
///     fillProblemData(P, q, A, b);                    // same sparsity pattern
///     if (!solver.updateData(&P, &q, &A, &b))          // cheap: values only
///       solver.setup(P, q, A, b, cone_spec, settings); // pattern changed: full rebuild
///     const conicxx::Solution& sol = solver.solve();   // auto warm-started per Settings
///   }
class Solver {
 public:
  Solver();
  ~Solver();
  Solver(Solver&&) noexcept;
  Solver& operator=(Solver&&) noexcept;
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  /// One-time (or sparsity-pattern-change) structural setup: builds the KKT
  /// sparsity pattern, allocates workspace, runs `analyzePattern()` once.
  /// Returns false on invalid input (mismatched dimensions, invalid cone
  /// spec, etc.) -- never throws.
  bool setup(const SparseMat& P, const Vec& q, const SparseMat& A, const Vec& b,
             const ConeSpec& cone_spec, const Settings& settings = Settings{});

  /// Cheap numeric-only update: overwrites P/q/A/b values in place, reusing
  /// the sparsity pattern from the last setup(). Pass nullptr for any
  /// argument that is unchanged. Returns false if a provided matrix's
  /// nonzero pattern does not match what setup() built -- the caller must
  /// then call setup() again (e.g. a multibody contact was made/broken).
  bool updateData(const SparseMat* P, const Vec* q, const SparseMat* A, const Vec* b);

  /// Seed the next solve() with a specific (x, s, z) iterate, given in
  /// original problem units. Must be called after setup().
  void setWarmStart(const Vec& x, const Vec& s, const Vec& z);

  void setSettings(const Settings& settings);
  const Settings& settings() const;

  /// Runs the homogeneous self-dual embedding interior-point method to
  /// convergence or failure. Always returns a Solution with a definite
  /// Status -- never throws, never hangs past max_iter.
  const Solution& solve();

  const Solution& solution() const;

 private:
  std::unique_ptr<detail::SolverImpl> impl_;
};

}  // namespace conicxx
