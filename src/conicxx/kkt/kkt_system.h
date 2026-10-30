#pragma once

#include <utility>
#include <vector>

#include "conicxx/cones/cone_set.h"
#include "conicxx/kkt/quasidefinite_ldlt.h"
#include "conicxx/settings.h"
#include "conicxx/kkt/sparsity_map.h"
#include "conicxx/types.h"

namespace conicxx::detail {

/// The quasidefinite KKT system of one IPM iteration,
///
///   K = [ P   A'  ]      lower triangle stored, exact values,
///       [ A  -Hs  ]      Hs = blockdiag(C on zero cones, W'W on orthant and SOC blocks),
///
/// C the (diagonal, >= 0) compliance of the zero-cone rows, 0 for equalities.
///
/// Regularization is applied to the pivots of the factorization only and removed again by
/// iterative refinement against K (see QuasiDefiniteLdlt and docs/design.md, "KKT
/// factorization"). One numeric factorization per call of updateScalingAndFactorize().
///   setup()                      symbolic analysis and P/A values, no factorization
///   updateData()                 new P/A values with the same pattern, no factorization
///   updateScalingAndFactorize()  Hs from the cones, then one factorization
class KktSystem {
 public:
  /// Counters accumulated since the last resetStats().
  struct Stats {
    int factorizations = 0;
    int dynamic_reg_pivots = 0;
    int refinement_steps = 0;
    bool equality_rank_deficient = false;
  };

  bool setup(const SparseMat& P_upper, const SparseMat& A, const ConeSet& cones,
             const Settings& settings);

  /// Overwrites P and/or A (nullptr = unchanged). Returns false, and changes nothing, if a
  /// sparsity pattern differs from the one given to setup().
  bool updateData(const SparseMat* P_upper, const SparseMat* A);

  /// True if P_upper and A (nullptr = not checked) have the sparsity patterns given to setup().
  bool samePatterns(const SparseMat* P_upper, const SparseMat* A) const;

  /// Compliance of the zero-cone rows (size = zero-cone dimension, or empty for none); takes
  /// effect at the next updateScalingAndFactorize().
  void setZeroCompliance(const Vec& compliance) { compliance_ = compliance; }

  /// Writes Hs from the current cone scalings and factorizes K.
  bool updateScalingAndFactorize(const ConeSet& cones);

  /// Exact-Coulomb Newton matrix: as updateScalingAndFactorize(), but the (2,2) block of every SOC
  /// block bi in coulomb_blocks becomes -(Hs + e_0 r'), r = row0.segment(offset, dim) of that
  /// block, i.e. only its first row changes. K is then unsymmetric and factorized by
  /// QuasiDefiniteLdlt::factorizeUnsymmetric(); solve() and solvePair() solve with it.
  bool updateScalingAndFactorize(const ConeSet& cones, const std::vector<Index>& coulomb_blocks,
                                 const Vec& row0);

  /// Applies regularization and refinement settings; no new setup() needed.
  void updateSettings(const Settings& settings);

  /// Solves K x = rhs; returns ||rhs - K x||_inf / ||rhs||_inf (0 if refinement is off).
  Scalar solve(const Vec& rhs, Vec& x_out);

  /// Solves K x1 = rhs1 (refined, as solve()) and K x2 = rhs2 (refined only if refine2) in one
  /// pass over the factor; returns the relative residual of x1.
  Scalar solvePair(const Vec& rhs1, Vec& x1, const Vec& rhs2, Vec& x2, bool refine2);

  Index n() const { return n_; }
  Index m() const { return m_; }
  Index dim() const { return n_ + m_; }
  bool isFactorized() const { return factorized_; }
  Scalar lastRefinementResidual() const { return last_refinement_residual_; }
  /// A solve left the equality (zero-cone) rows unsatisfied: redundant or inconsistent
  /// equality constraints. Accumulated like the other Stats.
  bool equalityRankDeficient() const { return stats_.equality_rank_deficient; }
  const Stats& stats() const { return stats_; }
  void resetStats() { stats_ = Stats{}; }
  const QuasiDefiniteLdlt& ldlt() const { return ldlt_; }

 private:
  bool writePAValues(const SparseMat* P_upper, const SparseMat* A);
  /// Writes -Hs (and the zero-cone compliance) into K and sets the static regularization.
  void writeScaling(const ConeSet& cones);
  Scalar recordSolve(const Vec& rhs, Scalar residual);

  Index n_ = 0, m_ = 0;
  bool factorized_ = false;
  Scalar last_refinement_residual_ = 0;
  Stats stats_;

  SparsityMap sparsity_;
  SparseMat K_;
  QuasiDefiniteLdlt ldlt_;

  // Static regularization: reg_ = reg_base_ + static_proportional * max|diag(K)|, per row.
  Vec reg_base_, reg_;
  Scalar reg_proportional_ = 0;
  std::vector<unsigned char> row_kind_;  // per row: 0 P, 1 orthant, 2 SOC, 3 zero cone
  std::vector<Index> diag_slots_;       // K slot of the diagonal entry of every row
  std::vector<Index> zero_row_pivots_;  // pivot positions of the zero-cone rows

  // Input value index -> K slot maps for updateData().
  std::vector<Index> p_diag_slots_, p_diag_value_idx_;  // value index -1: entry not stored
  std::vector<std::pair<Index, Index>> p_offdiag_slots_, a_slots_;  // (K slot, value index)
  std::vector<std::vector<Index>> hs_slots_;  // per cone block, writeHsLowerTriangle() order
  Vec hs_scratch_;
  Vec compliance_;  // zero-cone compliance, empty if none
  // Values of K's transposed entries for the unsymmetric (Coulomb) matrix, in K_'s value order.
  std::vector<Scalar> upper_values_;

  // Sparsity patterns of P and A at setup().
  std::vector<Index> p_outer_ref_, p_inner_ref_, a_outer_ref_, a_inner_ref_;
};

}  // namespace conicxx::detail
