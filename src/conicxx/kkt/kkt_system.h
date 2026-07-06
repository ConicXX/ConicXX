#pragma once

#include <Eigen/SparseCholesky>
#include <vector>

#include "conicxx/cones/cone_set.h"
#include "conicxx/kkt/sparsity_map.h"
#include "conicxx/settings.h"
#include "conicxx/types.h"

#ifdef CONICXX_HAVE_QDLDL
#include "conicxx/kkt/qdldl_ldlt.h"
#include "conicxx/kkt/regularized_ldlt.h"
#endif

namespace conicxx::detail {

/// Assembles and factorizes the augmented KKT system
///
///   K = [ P + epsP*I        A'           ]
///       [ A                 -Hs - epsA*I ]
///
/// Only the strict lower triangle (plus diagonal) is ever built, matching
/// Eigen::SimplicialLDLT<..., Eigen::Lower>'s expectations -- this halves
/// the number of physical nonzeros relative to building both triangles.
/// Within the Hs block, only SecondOrder cone blocks get a full dense
/// lower-triangle of slots (their NT scaling is genuinely dense); Zero and
/// Nonnegative blocks -- whose Hs is zero resp. diagonal -- get diagonal
/// slots only, since registering the full triangle there would explicitly
/// store an always-zero off-diagonal block in K's sparsity pattern, which
/// is catastrophic for fill-in/factorization cost when such a block is
/// large (e.g. a big equality/Zero-cone block).
///
/// `setup()` performs the one-time triplet-based assembly and a single
/// `analyzePattern()` call. Two update paths reuse that pattern without any
/// re-triangulation:
///   - `updateData()` overwrites P/A numeric values in place (the common
///     per-timestep path when the sparsity pattern hasn't changed).
///   - `updateScalingAndFactorize()` overwrites the Hs (NT-scaling) block
///     values (the per-IPM-iteration path) and refactorizes.
///
/// Backend: factorizes/solves via Eigen::SimplicialLDLT, QdldlLdlt
/// (github.com/osqp/qdldl, the backend QOCO/Clarabel use), or RegularizedLdlt (a modified QDLDL
/// numeric factorization loop with true per-pivot dynamic regularization, Davis/ECOS-style),
/// selected once in setup() from Settings::linear_solver and dispatched through
/// backendAnalyzePattern()/backendFactorize()/backendInfo()/backendVectorD()/
/// backendSolve() -- everything else in this class is written against those
/// five calls and does not know which concrete backend is active.
///
/// Regularization: Eigen::SimplicialLDLT and QdldlLdlt cannot intercept individual pivots
/// mid-factorization, so for those two backends "dynamic regularization" is implemented as an
/// outer retry loop: factorize with the current regularization, inspect the resulting pivots
/// (D from LDL^T) for magnitude, and if any are too small, bump the regularization and
/// refactorize from scratch (up to a few times). RegularizedLdlt instead corrects a bad pivot
/// in place as soon as it is computed, so this retry loop is (for that backend) effectively
/// always satisfied on the first attempt -- it stays in place uniformly across all three
/// backends rather than being special-cased away, since it is harmless overhead when it never
/// fires. The *unregularized* (well, only statically-regularized) matrix is kept
/// separately so iterative refinement converges to the solution of the
/// intended system, not the dynamically over-regularized one used only to
/// obtain a stable factorization.
class KktSystem {
 public:
  bool setup(const SparseMat& P_upper, const SparseMat& A, const ConeSet& cones,
             const Settings& settings);

  /// Overwrite P/A numeric values (nullptr = unchanged) and refactorize.
  /// Returns false if the provided matrix's sparsity pattern does not match
  /// what setup() built -- the caller must then call setup() again.
  bool updateData(const SparseMat* P_upper, const SparseMat* A);

  /// Overwrite the Hs (NT scaling) block from the current cone scaling and
  /// refactorize. Called once per IPM iteration.
  bool updateScalingAndFactorize(const ConeSet& cones);

  /// Solve K * x = rhs using the current factorization plus iterative
  /// refinement against the (statically-regularized only) system matrix.
  /// Returns the achieved relative residual norm. If the solve produces a
  /// non-finite result (the accepted factorization was fine by its pivot
  /// check but still numerically inadequate for this particular rhs -- see
  /// escalateAndResolve()), transparently bumps regularization further and
  /// retries in place before giving up.
  Scalar solve(const Vec& rhs, Vec& x_out);

  Index n() const { return n_; }
  Index m() const { return m_; }
  Index dim() const { return n_ + m_; }
  bool isFactorized() const { return factorized_; }

 private:
  bool factorizeWithRetry();
  void writeRegularizedDiagonal(SparseMat& K, Scalar extra_p_reg, Scalar extra_a_reg) const;

  /// Factorizes K_ (bumped by extra_p/extra_a if either is nonzero) and
  /// reports whether the result is usable: Eigen::Success and every |D|
  /// pivot above dynamic_reg_eps_. Shared by factorizeWithRetry() (initial
  /// factorization) and escalateAndResolve() (post-solve escalation).
  bool tryFactorizeWithBump(Scalar extra_p, Scalar extra_a);

  /// Called by solve() when backendSolve() returns a non-finite result even
  /// though factorizeWithRetry() accepted the factorization (its per-pivot
  /// magnitude check can pass while the pivot *spread* -- min vs. max |D| --
  /// is still large enough to blow up for a particular rhs, e.g. a
  /// near-degenerate cone block whose NT scaling makes one KKT diagonal
  /// entry orders of magnitude larger than another that's already at the
  /// static-regularization floor). Bumps regularization far more
  /// aggressively than factorizeWithRetry() (whose small, pivot-floor-only
  /// bumps can be satisfied without denting the actual pivot spread) and
  /// refactorizes, re-solving rhs after each bump until the result is
  /// finite. On success, writes the finite result to x_out and returns
  /// true; returns false (x_out left at its last, still non-finite value)
  /// if attempts are exhausted.
  bool escalateAndResolve(const Vec& rhs, Vec& x_out);

  // Backend dispatch: the rest of the class (regularization retry/escalation, iterative
  // refinement) is written entirely in terms of these five calls and knows nothing about which
  // concrete backend (Eigen::SimplicialLDLT or QdldlLdlt) is active.
  void backendAnalyzePattern(const SparseMat& K);
  void backendFactorize(const SparseMat& K);
  Eigen::ComputationInfo backendInfo() const;
  Vec backendVectorD() const;
  Vec backendSolve(const Vec& rhs) const;

  Index n_ = 0, m_ = 0;
  Scalar static_reg_p_ = 0, static_reg_a_ = 0;
  Scalar dynamic_reg_eps_ = 0, dynamic_reg_delta_ = 0;
  int refine_max_iter_ = 3;
  Scalar refine_tol_ = 1e-12;

  SparsityMap sparsity_;
  SparseMat K_;  // statically-regularized "true" matrix (no dynamic bump)
  Eigen::SimplicialLDLT<SparseMat, Eigen::Lower> ldlt_eigen_;
#ifdef CONICXX_HAVE_QDLDL
  QdldlLdlt ldlt_qdldl_;
  RegularizedLdlt ldlt_reg_;
#endif
  bool use_qdldl_ = false;
  bool use_regularized_ = false;
  bool factorized_ = false;
  // Regularization bump used to obtain the *current* ldlt_ factorization
  // (0 unless factorizeWithRetry() needed to bump past the statically-
  // regularized K_ to clear its pivot-magnitude check). escalateAndResolve()
  // continues bumping from here rather than restarting at 0.
  Scalar last_extra_p_ = 0, last_extra_a_ = 0;

  // Slot bookkeeping, in terms of (row, col) pairs kept alongside for
  // robust value lookup via SparseMatrix::coeff() on updateData.
  std::vector<Index> p_diag_slots_;                              // size n
  std::vector<std::tuple<Index, Index, Index>> p_offdiag_slots_;  // (slot, row, col), row<col in P_upper
  std::vector<std::tuple<Index, Index, Index>> a_slots_;          // (slot, row_in_A, col_in_A)

  struct HsBlockSlots {
    Index z_offset = 0;  // offset of this cone block within z-space [0, m)
    Index dim = 0;
    // (slot, a, b) for a >= b (lower-triangular within the block, incl. diag)
    std::vector<std::tuple<Index, Index, Index>> slots;
  };
  std::vector<HsBlockSlots> hs_blocks_;

  // Structural fingerprints from setup(), used to validate updateData() inputs.
  std::vector<Index> p_outer_ref_, p_inner_ref_;
  std::vector<Index> a_outer_ref_, a_inner_ref_;
};

}  // namespace conicxx::detail
