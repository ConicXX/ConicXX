#pragma once

#include <Eigen/SparseCholesky>
#include <vector>

#include "conicxx/cones/cone_set.h"
#include "conicxx/kkt/sparsity_map.h"
#include "conicxx/settings.h"
#include "conicxx/types.h"

namespace conicxx::detail {

/// Assembles and factorizes the augmented KKT system
///
///   K = [ P + epsP*I        A'           ]
///       [ A                 -Hs - epsA*I ]
///
/// Only the strict lower triangle (plus diagonal) is ever built, matching
/// Eigen::SimplicialLDLT<..., Eigen::Lower>'s expectations -- this halves
/// the number of physical nonzeros relative to building both triangles.
///
/// `setup()` performs the one-time triplet-based assembly and a single
/// `analyzePattern()` call. Two update paths reuse that pattern without any
/// re-triangulation:
///   - `updateData()` overwrites P/A numeric values in place (the common
///     per-timestep path when the sparsity pattern hasn't changed).
///   - `updateScalingAndFactorize()` overwrites the Hs (NT-scaling) block
///     values (the per-IPM-iteration path) and refactorizes.
///
/// Regularization: Eigen's SimplicialLDLT does not support intercepting
/// individual pivots mid-factorization (unlike e.g. a custom Davis-style
/// sparse LDL used by ECOS/CVXGEN), so "dynamic regularization" here is
/// implemented as an outer retry loop: factorize with the current
/// regularization, inspect the resulting pivots (D from LDL^T) for
/// magnitude, and if any are too small, bump the regularization and
/// refactorize from scratch (up to a few times). The *unregularized* (well,
/// only statically-regularized) matrix is kept separately so iterative
/// refinement converges to the solution of the intended system, not the
/// dynamically over-regularized one used only to obtain a stable
/// factorization.
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
  /// Returns the achieved relative residual norm.
  Scalar solve(const Vec& rhs, Vec& x_out) const;

  Index n() const { return n_; }
  Index m() const { return m_; }
  Index dim() const { return n_ + m_; }
  bool isFactorized() const { return factorized_; }

 private:
  bool factorizeWithRetry();
  void writeRegularizedDiagonal(SparseMat& K, Scalar extra_p_reg, Scalar extra_a_reg) const;

  Index n_ = 0, m_ = 0;
  Scalar static_reg_p_ = 0, static_reg_a_ = 0;
  Scalar dynamic_reg_eps_ = 0, dynamic_reg_delta_ = 0;
  int refine_max_iter_ = 3;
  Scalar refine_tol_ = 1e-12;

  SparsityMap sparsity_;
  SparseMat K_;  // statically-regularized "true" matrix (no dynamic bump)
  Eigen::SimplicialLDLT<SparseMat, Eigen::Lower> ldlt_;
  bool factorized_ = false;

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
