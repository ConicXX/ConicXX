#pragma once

#include <vector>

#include "conicxx/types.h"

namespace conicxx::detail {

/// Sparse LDL^T factorization of quasidefinite KKT matrices.
///
/// Up-looking factorization with a fill-reducing AMD ordering (the QDLDL algorithm), extended
/// the way Clarabel and QOCO extend QDLDL for interior-point methods:
///   - expected pivot signs: a pivot with D*sign < dynamic_eps is replaced by
///     sign*dynamic_delta, so the inertia is always the expected one;
///   - static regularization per row, applied to the pivots only and never stored in the
///     matrix copy used for refinement;
///   - iterative refinement against that exact copy, keeping the best iterate.
/// analyzePattern() computes the ordering, the elimination tree and the row structure of L
/// once; factorize() is a single numeric pass without allocations or elimination-tree walks.
/// See docs/design.md, section "KKT factorization".
class QuasiDefiniteLdlt {
 public:
  struct Params {
    bool dynamic_enable = true;
    Scalar dynamic_eps = 1e-13;
    Scalar dynamic_delta = 2e-7;
    int refine_max_iter = 10;
    Scalar refine_reltol = 1e-13;
    Scalar refine_abstol = 1e-12;
    Scalar refine_stop_ratio = 5;  ///< stop when ||r_old|| / ||r_new|| < refine_stop_ratio
    /// Supernodal numeric factorization (dense blocks for columns with nested patterns) where it
    /// pays, i.e. unless L is very sparse; read by analyzePattern(). Same pivots, regularization
    /// and L as the simplicial one, up to roundoff.
    bool supernodal = true;
  };
  Params params;

  /// Symbolic phase. K_lower: lower triangle (incl. diagonal) of a symmetric matrix, compressed
  /// column-major; entries above the diagonal are ignored, missing diagonal entries are zero.
  /// signs: expected pivot sign of every row in the original ordering (+1 or -1).
  /// perm: pivot -> original row; nullptr (default) computes an AMD ordering.
  bool analyzePattern(const SparseMat& K_lower, const std::vector<signed char>& signs,
                      const std::vector<Index>* perm = nullptr);

  /// Static regularization per row in the original ordering: the pivot of row i starts from
  /// K_ii + sign_i * reg_i in factorize(). Zero after analyzePattern().
  void setStaticRegularization(const Vec& reg);

  /// Numeric phase; K_lower must have the analyzed pattern. Returns false only for a zero or
  /// non-finite pivot (non-finite data, or dynamic regularization disabled).
  bool factorize(const SparseMat& K_lower);

  /// Solves K x = b for the exact K of the last factorize(), with iterative refinement, and
  /// returns ||b - K x||_inf, or -1 if refinement is off (Params::refine_max_iter = 0) and no
  /// residual was computed. Resizes x if needed; allocation-free otherwise.
  Scalar solve(const Vec& b, Vec& x);

  /// Solves K x1 = b1 and K x2 = b2 in one pass over L. x1 is refined as in solve(); x2 only if
  /// refine2 (otherwise no residual is computed for it). Returns ||b1 - K x1||_inf; lastResidual()
  /// and lastRefineSteps() refer to x1 (plus the steps for x2).
  Scalar solvePair(const Vec& b1, Vec& x1, const Vec& b2, Vec& x2, bool refine2);

  Index rows() const { return n_; }
  /// The last analyzePattern() chose the supernodal factorization.
  bool isSupernodal() const { return supernodal_; }
  Index nonZerosL() const { return n_ > 0 ? Lp_[static_cast<size_t>(n_)] : 0; }
  /// Pivots replaced by dynamic regularization in the last factorize().
  Index numDynamicRegularized() const { return num_dyn_reg_; }
  /// Positive pivots of the last factorize(); always the number of +1 signs.
  Index numPositivePivots() const { return num_positive_; }
  int lastRefineSteps() const { return last_refine_steps_; }
  /// b - K x of the last solve() in pivot order: row i of K is entry iperm()[i].
  const Vec& lastResidual() const { return r_; }
  const std::vector<Index>& perm() const { return perm_; }    ///< pivot -> original row
  const std::vector<Index>& iperm() const { return iperm_; }  ///< original row -> pivot

 private:
  void buildPermutedUpper(const SparseMat& K_lower);
  /// Elimination tree and column counts of L for the current C.
  void eliminationTree(std::vector<Index>& etree, std::vector<Index>& Lnz) const;
  void analyzeSupernodes(const std::vector<Index>& etree);
  bool factorizeSimplicial();
  bool factorizeSupernodal();
  /// Applies dynamic regularization to pivot k and stores it; false for a zero/non-finite pivot.
  bool setPivot(Index k, Scalar d);
  void solveInPlace(Vec& x) const;
  void solvePairInPlace(Vec& y) const;  // y interleaved: y[2k], y[2k+1]
  Scalar refine(Vec& xp, Vec& r, int& steps);
  Scalar residual(const Vec& x, Vec& r) const;

  Index n_ = 0;
  bool analyzed_ = false;
  std::vector<Index> perm_, iperm_;
  // C = upper triangle of P K P^T with exact values, and the position in C of each entry of K.
  std::vector<Index> Cp_, Ci_, KtoC_;
  std::vector<Scalar> Cx_;
  // L in compressed columns, and its row structure: row k holds (column Rj_[q], position
  // Rpos_[q] in Li_/Lx_) for q in [Rp_[k], Rp_[k+1]), columns ascending.
  std::vector<Index> Lp_, Li_, Rp_, Rj_, Rpos_;
  std::vector<Scalar> Lx_, D_, Dinv_, sign_, reg_, y_;
  Vec bp_, xp_, xt_, r_, rt_;  // solve() and refinement workspaces, pivot order
  Vec xp2_, r2_, y2_;  // second right-hand side of solvePair(); y2_ interleaved

  // Supernodal factorization: supernode s holds columns [sn_ptr_[s], sn_ptr_[s+1]), whose patterns
  // are the trailing rows of the row list sn_rows_[sn_rows_ptr_[s] ...], stored as a dense
  // column-major block at sn_x_[sn_x_ptr_[s]]. Lower entries of C by column (lo_*) for assembly.
  bool supernodal_ = false;
  std::vector<Index> sn_ptr_, sn_of_, sn_rows_ptr_, sn_rows_, sn_x_ptr_;
  std::vector<Index> lo_p_, lo_r_, lo_pos_, diag_pos_;
  std::vector<Scalar> sn_x_;
  std::vector<Index> map_, head_, next_, dpos_;  // numeric workspaces
  std::vector<Scalar> upd_, wbuf_;
  Index max_rows_ = 0, max_width_ = 0;
  Index num_dyn_reg_ = 0, num_positive_ = 0;
  int last_refine_steps_ = 0;
};

}  // namespace conicxx::detail
