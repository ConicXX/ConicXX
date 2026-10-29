#pragma once

#include <memory>

#include "conicxx/cone_spec.h"
#include "conicxx/convert.h"
#include "conicxx/settings.h"
#include "conicxx/solution.h"
#include "conicxx/types.h"

namespace conicxx {

namespace detail {
class SolverImpl;
}

/// Top-level interior-point solver for
///
///   min  0.5 x'Px + q'x + 0.5 z_E'C z_E
///   s.t. Ax - C z_E + s = b,  s in K = Zero x Nonnegative x SecondOrderCone x ...
///
/// P must be symmetric positive semidefinite, supplied as an upper-triangular
/// sparse matrix. C = diag(c) >= 0 is an optional compliance of the zero-cone (equality) rows,
/// with z_E their multipliers: c_i = 0 is an equality, c_i > 0 a compliant row. It is the
/// elimination of extra primal variables pi = -z_E with objective 0.5 pi'C pi and rows
/// A_E x + C pi = b_E (docs/design.md, "Compliant rows"). Typical usage in a per-timestep simulation loop:
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
  ///
  /// zero_compliance: c (size cone_spec.zero_dim, entries >= 0); nullptr means c = 0.
  bool setup(const SparseMat& P, const Vec& q, const SparseMat& A, const Vec& b,
             const ConeSpec& cone_spec, const Settings& settings = Settings{},
             const Vec* zero_compliance = nullptr);

  /// Convenience overloads accepting P and/or A as dense matrices (handy for
  /// small problems or quick prototyping, e.g. writing them out with `<<`).
  /// Each dense argument is converted via conicxx::toSparse()/
  /// toSparseUpperTriangular() -- which emits a one-time warning to stderr,
  /// since a dense matrix carries no sparsity for the solver to exploit --
  /// then delegates to the canonical sparse setup() above. For large or
  /// genuinely sparse problems, construct SparseMat directly instead.
  ///
  /// There is deliberately no dense overload of updateData(): that is the
  /// per-timestep hot path this library is designed around, and accepting
  /// dense input there would defeat the sparsity-pattern-reuse it exists
  /// for (a dense conversion of changing numeric data can't be relied on to
  /// keep a stable derived sparsity pattern the way a real sparse matrix
  /// does). Convert once with toSparse()/toSparseUpperTriangular() and reuse
  /// the resulting SparseMat across updateData() calls, mutating only its
  /// values.
  bool setup(const Mat& P, const Vec& q, const SparseMat& A, const Vec& b,
             const ConeSpec& cone_spec, const Settings& settings = Settings{});
  bool setup(const SparseMat& P, const Vec& q, const Mat& A, const Vec& b,
             const ConeSpec& cone_spec, const Settings& settings = Settings{});
  bool setup(const Mat& P, const Vec& q, const Mat& A, const Vec& b, const ConeSpec& cone_spec,
             const Settings& settings = Settings{});

  /// Cheap numeric-only update: overwrites P/q/A/b values in place, reusing
  /// the sparsity pattern from the last setup(). Pass nullptr for any
  /// argument that is unchanged. Returns false if a provided matrix's
  /// nonzero pattern does not match what setup() built -- the caller must
  /// then call setup() again (e.g. a multibody contact was made/broken).
  /// zero_compliance: new c (nullptr = unchanged); rejected (false) if invalid.
  bool updateData(const SparseMat* P, const Vec* q, const SparseMat* A, const Vec* b,
                  const Vec* zero_compliance = nullptr);

  /// Seed the next solve() with a specific (x, s, z) iterate, given in
  /// original problem units. Must be called after setup(). s and z need not lie in the cones:
  /// they are projected onto them before the warm start is formed (Settings::warm_lambda). Used
  /// only if Settings::warm_start is set and its merit is not worse than the cold start's.
  /// Ignored if a dimension does not match or a value is not finite.
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
