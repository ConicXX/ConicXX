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
/// large (e.g. a big equality/Zero-cone block). The numeric values themselves
/// come from each ConeBase's own writeHsLowerTriangle() (see cone_base.h) --
/// no dense dim() x dim() Hs matrix is ever materialized for Zero/Nonnegative,
/// which is what made a large block catastrophic for memory too, not just
/// fill-in.
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
/// Regularization (see Settings::RegularizationSettings): two matrices share one sparsity
/// pattern (built once, never copied in the per-iteration hot path -- only the O(n+m) diagonal
/// slots are ever rewritten there, not the O(nnz) off-diagonal ones):
///   - `K_exact_`: the true, unregularized system. P's diagonal and Hs's off-diagonal entries are
///     exact; the zero-cone (equality) rows of the (2,2) block are exactly 0 unless the caller
///     explicitly opts into `static_zero != 0`. Iterative refinement in solve() always targets
///     this matrix, so any regularization baked into the *factorized* matrix below is compensated
///     rather than silently perturbing the answer.
///   - `K_fact_`: `K_exact_` plus per-cone-type static regularization on the diagonal only (see
///     `rebuildKFactDiagonal()`). This is the only matrix ever factorized.
/// Eigen::SimplicialLDLT and QdldlLdlt cannot intercept individual pivots mid-factorization, so
/// for those two backends "dynamic regularization" is an outer retry loop: factorize `K_fact_`,
/// inspect the resulting pivots (D from LDL^T) for magnitude, and if any are too small, bump the
/// per-block diagonal (P/nonneg/SOC only, in place -- zero rows are excluded by default, see
/// `Settings::RegularizationSettings::dynamic_on_zero_rows`) and refactorize (up to a few times).
/// RegularizedLdlt instead corrects a bad pivot in place as soon as it is computed (except on a
/// zero row, which it never corrects, for the same "never perturb equalities" reason), so this
/// retry loop is satisfied on the first attempt in the common case -- but its own info() still
/// reports failure when too large a fraction of pivots needed correcting, or when *any*
/// (uncorrectable) zero-row pivot was bad, so this retry loop remains a real backstop for it too.
/// If the per-block ladder exhausts without success and a zero-cone block exists, one more
/// factorization is tried with `static_zero_factor_only` added to the zero rows only (a
/// preconditioner, not a perturbation -- refinement against `K_exact_` removes its effect); if
/// that still fails, `Info::equality_rank_deficient` is set instead of accepting a bad
/// factorization or silently perturbing the equality rows further.
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

  /// Solve K * x = rhs using the current factorization plus iterative refinement against
  /// K_exact_ (the unregularized system -- see the class comment). Returns the achieved relative
  /// residual norm (also cached, see lastRefinementResidual()). If the solve produces a
  /// non-finite result (the accepted factorization was fine by its pivot check but still
  /// numerically inadequate for this particular rhs -- see escalateAndResolve()), transparently
  /// bumps regularization further and retries in place before giving up.
  Scalar solve(const Vec& rhs, Vec& x_out);

  Index n() const { return n_; }
  Index m() const { return m_; }
  Index dim() const { return n_ + m_; }
  bool isFactorized() const { return factorized_; }

  /// Relative residual (against K_exact_) of the last solve()'s iterative refinement.
  Scalar lastRefinementResidual() const { return last_refinement_residual_; }

  /// Set by the most recent factorizeWithRetry() (called from updateData()/
  /// updateScalingAndFactorize()) if a zero-cone (equality) pivot stayed bad even after the
  /// one-shot static_zero_factor_only fallback -- see the class comment and Settings.
  bool equalityRankDeficient() const { return equality_rank_deficient_; }

 private:
  /// Writes P/A numeric values into K_exact_ (always) and K_fact_'s off-diagonal entries (never
  /// regularized) -- the value-writing part of updateData(), factored out so setup() can use it
  /// without updateData()'s own factorizeWithRetry() call (see setup()'s comment for why).
  /// Returns false on a sparsity-pattern mismatch, same as updateData().
  bool writePAValues(const SparseMat* P_upper, const SparseMat* A);

  bool factorizeWithRetry();

  /// Rewrites K_fact_'s diagonal entries (P block: +static_P plus the proportional term; Hs
  /// block: -static_{nonneg,soc} plus the proportional term, or -static_zero with no
  /// proportional term for zero rows) from K_exact_'s current raw diagonal values. This is the
  /// single source of truth for K_fact_'s diagonal -- called after every K_exact_ diagonal write
  /// (from updateData() and updateScalingAndFactorize()) and resets any dynamic bump baked into
  /// K_fact_ back to the pure-static baseline (a full O(n+m) rewrite is simpler and just as cheap
  /// as incremental rollback bookkeeping, and only touches the diagonal, never the O(nnz)
  /// off-diagonal entries).
  void rebuildKFactDiagonal();

  /// Adds `sign * delta` to K_fact_'s value at each slot in `slots` (a no-op if delta == 0).
  /// Used both by the dynamic-regularization ladder (factorizeWithRetry()/escalateAndResolve())
  /// and the T2.5 zero-row factor-only fallback.
  void bumpDiagSlots(const std::vector<Index>& slots, Scalar delta, Scalar sign);

  /// Factorizes the current K_fact_ and reports whether the result is usable: Eigen::Success and
  /// every |D| pivot above dynamic_eps_. Shared by factorizeWithRetry() (initial factorization)
  /// and escalateAndResolve() (post-solve escalation).
  bool tryFactorizeCurrentKFact();

  /// Called by solve() when backendSolve() returns a non-finite result even
  /// though factorizeWithRetry() accepted the factorization (its per-pivot
  /// magnitude check can pass while the pivot *spread* -- min vs. max |D| --
  /// is still large enough to blow up for a particular rhs, e.g. a
  /// near-degenerate cone block whose NT scaling makes one KKT diagonal
  /// entry orders of magnitude larger than another that's already at the
  /// static-regularization floor). Bumps the P/nonneg/SOC diagonal (never zero rows) far more
  /// aggressively than factorizeWithRetry() and refactorizes, re-solving rhs after each bump
  /// until the result is finite. On success, writes the finite result to x_out and returns true;
  /// returns false (x_out left at its last, still non-finite value) if attempts are exhausted.
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

  // --- Regularization settings (Settings::RegularizationSettings, unpacked) ---
  Scalar static_P_ = 0, static_nonneg_ = 0, static_soc_ = 0, static_zero_ = 0;
  Scalar static_proportional_ = 0, static_zero_factor_only_ = 0;
  Scalar dynamic_eps_ = 0, dynamic_delta_ = 0;
  bool dynamic_on_zero_rows_ = false;

  int refine_max_iter_ = 10;
  Scalar refine_reltol_ = 1e-13, refine_abstol_ = 1e-12, refine_stop_ratio_ = 5;
  Scalar last_refinement_residual_ = 0;
  bool equality_rank_deficient_ = false;

  SparsityMap sparsity_;
  SparseMat K_exact_;  // unregularized "true" matrix; iterative refinement always targets this
  SparseMat K_fact_;   // K_exact_ + per-block static (+ dynamic, transiently) regularization
  Eigen::SimplicialLDLT<SparseMat, Eigen::Lower> ldlt_eigen_;
#ifdef CONICXX_HAVE_QDLDL
  QdldlLdlt ldlt_qdldl_;
  RegularizedLdlt ldlt_reg_;
#endif
  bool use_qdldl_ = false;
  bool use_regularized_ = false;
  bool factorized_ = false;
  // Dynamic bump currently baked into K_fact_ (0 right after rebuildKFactDiagonal()).
  // escalateAndResolve() continues bumping from here rather than restarting at 0.
  Scalar last_extra_p_ = 0, last_extra_nonneg_ = 0, last_extra_soc_ = 0, last_extra_zero_ = 0;

  // Slot bookkeeping, in terms of (row, col) pairs kept alongside for
  // robust value lookup via SparseMatrix::coeff() on updateData.
  std::vector<Index> p_diag_slots_;                              // size n
  std::vector<std::tuple<Index, Index, Index>> p_offdiag_slots_;  // (slot, row, col), row<col in P_upper
  std::vector<std::tuple<Index, Index, Index>> a_slots_;          // (slot, row_in_A, col_in_A)

  // Diagonal slots bucketed by cone type, for rebuildKFactDiagonal()'s per-type static
  // regularization and the dynamic-regularization ladder's per-block bumping. Disjoint from each
  // other and from p_diag_slots_ (P and Hs occupy different rows of K).
  std::vector<Index> nonneg_diag_slots_, soc_diag_slots_, zero_diag_slots_;
  // Concatenation of p_diag_slots_ + {nonneg,soc,zero}_diag_slots_, for the max|diag(K_exact_)|
  // scan in rebuildKFactDiagonal()'s proportional term.
  std::vector<Index> all_diag_slots_;

  struct HsBlockSlots {
    ConeType type = ConeType::Zero;
    // Flat slot ids, in the same row-major lower-triangle order (a = 0..dim-1, b = 0..a) that
    // ConeBase::writeHsLowerTriangle() writes values in -- diagonal-only for Zero/Nonnegative.
    std::vector<Index> slots;
    // Parallel to slots: 1 where (a == b), i.e. this entry is on the diagonal.
    std::vector<unsigned char> is_diag;
  };
  std::vector<HsBlockSlots> hs_blocks_;
  // Scratch buffer for one block's Hs entries, sized once in setup() to the largest block's
  // numHsEntries() and reused (via .head()) across blocks/iterations to avoid a per-iteration
  // heap allocation in updateScalingAndFactorize().
  Vec hs_entries_scratch_;

  // Original-row membership in a zero-cone block (size n_+m_), passed to RegularizedLdlt so it
  // never dynamically corrects an equality pivot -- see regularized_ldlt.h.
  std::vector<unsigned char> is_zero_row_;

  // Structural fingerprints from setup(), used to validate updateData() inputs.
  std::vector<Index> p_outer_ref_, p_inner_ref_;
  std::vector<Index> a_outer_ref_, a_inner_ref_;
};

}  // namespace conicxx::detail
