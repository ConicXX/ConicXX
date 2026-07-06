#include "conicxx/kkt/qdldl_ldlt.h"

#include <Eigen/OrderingMethods>
#include <algorithm>

#include <qdldl.h>

namespace conicxx::detail {

namespace {
// Forced off via CMake (QDLDL_LONG/QDLDL_FLOAT CACHE FORCE) so these hold exactly -- lets every
// QDLDL_* call below pass conicxx's own Index*/Scalar* buffers directly, with no copies/casts.
static_assert(sizeof(QDLDL_int) == sizeof(Index), "QDLDL_int must match conicxx::Index (build with QDLDL_LONG=OFF)");
static_assert(sizeof(QDLDL_float) == sizeof(Scalar), "QDLDL_float must match conicxx::Scalar (build with QDLDL_FLOAT=OFF)");
}  // namespace

void QdldlLdlt::analyzePattern(const SparseMat& K_lower) {
  n_ = static_cast<Index>(K_lower.rows());

  // AMDOrdering's output is the *inverse* permutation -- verified directly against
  // Eigen::SimplicialLDLT::permutationP() on a real matrix, which produced a wildly different
  // (114x more fill-in) elimination order when this inversion was missing. Eigen's own
  // SimplicialCholeskyBase::ordering() (SimplicialCholesky.h) has the same "note that ordering
  // methods compute the inverse permutation" comment and does exactly this inversion before
  // using the result for anything.
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index> perm_inv;
  Eigen::AMDOrdering<Index> amd;
  amd(K_lower.selfadjointView<Eigen::Lower>(), perm_inv);
  perm_ = perm_inv.inverse();

  A_upper_.resize(n_, n_);
  A_upper_.selfadjointView<Eigen::Upper>() = K_lower.selfadjointView<Eigen::Lower>().twistedBy(perm_);
  A_upper_.makeCompressed();

  std::vector<Index> work(static_cast<size_t>(n_));
  etree_.assign(static_cast<size_t>(n_), 0);
  Lnz_.assign(static_cast<size_t>(n_), 0);
  const Index Ltotal = QDLDL_etree(n_, A_upper_.outerIndexPtr(), A_upper_.innerIndexPtr(),
                                    work.data(), Lnz_.data(), etree_.data());
  info_ = (Ltotal >= 0) ? Eigen::Success : Eigen::NumericalIssue;

  const size_t LtotalSafe = static_cast<size_t>(std::max(Ltotal, Index(0)));
  Lp_.assign(static_cast<size_t>(n_) + 1, 0);
  Li_.assign(LtotalSafe, 0);
  Lx_.assign(LtotalSafe, Scalar(0));
  D_.assign(static_cast<size_t>(n_), Scalar(0));
  Dinv_.assign(static_cast<size_t>(n_), Scalar(0));
  fwork_.assign(static_cast<size_t>(n_), Scalar(0));
  bwork_.assign(static_cast<size_t>(n_), 0);
  iwork_.assign(3 * static_cast<size_t>(n_), 0);
}

void QdldlLdlt::factorize(const SparseMat& K_lower) {
  // Precondition: analyzePattern(K_lower) has already been called with the same sparsity
  // pattern (matches Eigen::SimplicialLDLT's own analyzePattern-once/factorize-many contract).
  A_upper_.selfadjointView<Eigen::Upper>() = K_lower.selfadjointView<Eigen::Lower>().twistedBy(perm_);
  A_upper_.makeCompressed();

  const Index posD = QDLDL_factor(n_, A_upper_.outerIndexPtr(), A_upper_.innerIndexPtr(),
                                   A_upper_.valuePtr(), Lp_.data(), Li_.data(), Lx_.data(), D_.data(),
                                   Dinv_.data(), Lnz_.data(), etree_.data(), bwork_.data(),
                                   iwork_.data(), fwork_.data());
  info_ = (posD >= 0) ? Eigen::Success : Eigen::NumericalIssue;
}

Vec QdldlLdlt::solve(const Vec& rhs) const {
  Vec x = perm_ * rhs;
  QDLDL_solve(n_, Lp_.data(), Li_.data(), Lx_.data(), Dinv_.data(), x.data());
  return perm_.inverse() * x;
}

}  // namespace conicxx::detail
