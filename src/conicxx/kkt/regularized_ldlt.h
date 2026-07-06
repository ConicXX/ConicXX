#pragma once

#include <Eigen/Core>
#include <vector>

#include "conicxx/types.h"

namespace conicxx::detail {

/// A sparse LDL^T backend with true per-pivot dynamic regularization, in the style of the
/// "Davis-style" regularized LDL used by ECOS/CVXGEN (Domahidi/Chu/Boyd): each diagonal pivot
/// is checked for the correct sign and minimum magnitude *as it is computed*, and replaced
/// in place with a signed floor value before it is used in the Schur-complement update for
/// any later pivot -- unlike KktSystem's outer factorizeWithRetry()/escalateAndResolve() retry
/// loop (used by the Eigen/QdldlLdlt backends), which can only react to a bad pivot by
/// refactorizing the *entire* matrix from scratch with a larger uniform bump.
///
/// This is a derivative of QDLDL's (github.com/osqp/qdldl, Apache-2.0) numeric factorization
/// loop: the symbolic analysis (elimination tree/Lnz via QDLDL_etree) and the triangular solves
/// (QDLDL_solve) are used unmodified from the linked qdldl library -- neither involves pivot
/// values, so there is nothing to regularize there. Only QDLDL_factor's numeric loop is
/// reimplemented here (see regularized_ldlt.cpp), with the per-pivot check/correction inserted
/// at the point each pivot is finalized. This is *not* a port of ECOS's own GPLv3-licensed
/// ldl.c -- only the published regularization methodology is reused, applied fresh to QDLDL's
/// (permissively licensed) elimination-tree-based algorithm.
///
/// Expects the KKT system's Vanderbei quasi-definite convention (see Settings' "KKT
/// regularization" section): rows/cols [0, nx) (the P block) must stay positive, rows/cols
/// [nx, n) (the -Hs block) must stay negative. `analyzePattern()` records which permuted
/// position each expectation applies to, since AMD reorders rows/cols.
class RegularizedLdlt {
 public:
  /// One-time: computes the AMD ordering, the elimination tree/L sparsity pattern, and the
  /// per-permuted-position expected pivot sign, from K_lower's structure and the P/-Hs block
  /// split at `nx`. Must be called again if K_lower's sparsity pattern changes.
  void analyzePattern(const SparseMat& K_lower, Index nx, Scalar dynamic_reg_eps,
                      Scalar dynamic_reg_delta);

  /// Refreshes the permuted-upper mirror of K_lower's values and runs the regularized numeric
  /// factorization. K_lower must have the same sparsity pattern passed to the last
  /// analyzePattern() call. Always succeeds (info() == Eigen::Success): a pivot that is too
  /// small or has the wrong sign is corrected in place rather than aborting the factorization.
  void factorize(const SparseMat& K_lower);

  /// Solves K_lower * x = rhs using the current factorization.
  Vec solve(const Vec& rhs) const;

  Eigen::ComputationInfo info() const { return info_; }

  /// D from the LDL^T factorization, in the *permuted* variable order -- see QdldlLdlt::vectorD
  /// for why callers don't need it undone (only used for aggregate min/max-magnitude checks).
  Vec vectorD() const { return Eigen::Map<const Vec>(D_.data(), n_); }

 private:
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index> perm_;
  SparseMat A_upper_;  // permuted, upper-triangular mirror of K_lower; values refreshed every factorize()

  std::vector<Index> etree_, Lnz_, Lp_, Li_, iwork_;
  std::vector<Scalar> Lx_, D_, Dinv_, fwork_;
  std::vector<unsigned char> bwork_;

  // expected_sign_[k] is +1 if permuted position k belongs to the P block (rows/cols < nx in
  // the original, unpermuted ordering) or -1 if it belongs to the -Hs block -- computed once in
  // analyzePattern() from the AMD permutation, since factorize() operates entirely in permuted
  // coordinates.
  std::vector<Scalar> expected_sign_;

  Index n_ = 0;
  Scalar dynamic_reg_eps_ = 0, dynamic_reg_delta_ = 0;
  Eigen::ComputationInfo info_ = Eigen::NumericalIssue;
};

}  // namespace conicxx::detail
