#include "conicxx/kkt/qdldl_ldlt.h"

#include <Eigen/OrderingMethods>
#include <algorithm>
#include <cmath>

// QDLDL_factor/QDLDL_etree's argument lists are specific to the pinned v0.1.9 (see
// CMakeLists.txt's FetchContent GIT_TAG) -- an older or newer vendored qdldl pulled in ahead of
// conicxx's own fetch (e.g. via find_package(qdldl) resolving to a sibling dependency's vendored
// copy first) can have a different signature and fail to compile here with a confusing "too
// few/many arguments" error rather than a version-mismatch one.
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

  // Precompute value_scatter_ (T7.3): re-run twistedBy() once more here (still trusted, still
  // correct -- this is analyzePattern(), not the hot path) with each of K_lower's stored values
  // replaced by a distinct sentinel (its own 1-based slot index), then read back, for each
  // entry that lands in A_upper_, which sentinel (hence which original slot) it came from. This
  // discovers the value_scatter_ map *empirically* from twistedBy()'s own (trusted) behavior
  // instead of re-deriving the permuted-symmetric-matrix index math by hand -- this codebase has
  // hit the "which direction does this permutation go" bug class more than once (see
  // RegularizedLdlt's and this class's own AMD-inverse-permutation history), and this sidesteps
  // it entirely rather than risking a repeat. Every stored (lower) entry of K_lower maps to
  // exactly one stored (upper) entry of A_upper_ (twistedBy() only reflects the matrix, it
  // doesn't change how many physical nonzeros represent a given symmetric pair), so this is a
  // total, one-to-one map, verified directly in test_regularized_ldlt.cpp-style tests comparing
  // scatter-based factorize() against a fresh twistedBy() on independently-randomized values.
  {
    const Index nnzK = K_lower.nonZeros();
    SparseMat K_sentinel = K_lower;
    for (Index k = 0; k < nnzK; ++k) {
      K_sentinel.valuePtr()[k] = static_cast<Scalar>(k + 1);
    }
    SparseMat A_sentinel(n_, n_);
    A_sentinel.selfadjointView<Eigen::Upper>() =
        K_sentinel.selfadjointView<Eigen::Lower>().twistedBy(perm_);
    A_sentinel.makeCompressed();

    value_scatter_.assign(static_cast<size_t>(nnzK), -1);
    for (Index i = 0; i < A_sentinel.nonZeros(); ++i) {
      const Index k = static_cast<Index>(std::llround(A_sentinel.valuePtr()[i])) - 1;
      value_scatter_[static_cast<size_t>(k)] = i;
    }
  }

  solve_scratch_.resize(n_);

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
  // Scatter values directly via the precomputed map (T7.3) -- no twistedBy() call, no allocation.
  const Index nnzK = K_lower.nonZeros();
  const Scalar* Kx = K_lower.valuePtr();
  Scalar* Ax = A_upper_.valuePtr();
  for (Index k = 0; k < nnzK; ++k) {
    Ax[value_scatter_[static_cast<size_t>(k)]] = Kx[k];
  }

  const Index posD = QDLDL_factor(n_, A_upper_.outerIndexPtr(), A_upper_.innerIndexPtr(),
                                   A_upper_.valuePtr(), Lp_.data(), Li_.data(), Lx_.data(), D_.data(),
                                   Dinv_.data(), Lnz_.data(), etree_.data(), bwork_.data(),
                                   iwork_.data(), fwork_.data());
  info_ = (posD >= 0) ? Eigen::Success : Eigen::NumericalIssue;
}

void QdldlLdlt::solve(const Vec& rhs, Eigen::Ref<Vec> out) const {
  solve_scratch_ = perm_ * rhs;
  QDLDL_solve(n_, Lp_.data(), Li_.data(), Lx_.data(), Dinv_.data(), solve_scratch_.data());
  out = perm_.inverse() * solve_scratch_;
}

}  // namespace conicxx::detail
