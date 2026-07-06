// Portions of this file (the numeric factorization loop in factorize(), before the per-pivot
// regularization insertion) are a derivative of QDLDL's QDLDL_factor
// (https://github.com/osqp/qdldl, src/qdldl.c):
//
//   Copyright 2018, Paul Goulart, Bartolomeo Stellato, Goran Banjac, Ian McInerney,
//   The OSQP developers
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
// This file has been modified from the original: the elimination-tree/nonzero-pattern
// bookkeeping is unchanged, but the per-pivot regularization check/correction in
// regularizePivot() and its call sites are new, and the "abort on exact-zero pivot" behavior
// of the original has been replaced by that correction.

#include "conicxx/kkt/regularized_ldlt.h"

#include <Eigen/OrderingMethods>
#include <cmath>

// QDLDL_etree/QDLDL_solve are used here unmodified (see regularized_ldlt.h), but their
// signatures are specific to the pinned v0.1.9 (CMakeLists.txt's FetchContent GIT_TAG) -- an
// older or newer vendored qdldl copy pulled in by a different dependency (e.g. via
// find_package(qdldl) picking up a sibling project's vendored version first) can have a
// different QDLDL_factor/QDLDL_etree argument list and fail to compile here with a confusing
// "too few/many arguments" error rather than a version-mismatch one.
#include <qdldl.h>

namespace conicxx::detail {

namespace {
constexpr Index kUnknown = -1;
constexpr unsigned char kUnused = 0;
constexpr unsigned char kUsed = 1;

static_assert(sizeof(QDLDL_int) == sizeof(Index),
             "QDLDL_int must match conicxx::Index (build with QDLDL_LONG=OFF)");
static_assert(sizeof(QDLDL_float) == sizeof(Scalar),
             "QDLDL_float must match conicxx::Scalar (build with QDLDL_FLOAT=OFF)");
}  // namespace

void RegularizedLdlt::analyzePattern(const SparseMat& K_lower, Index nx, Scalar dynamic_reg_eps,
                                     Scalar dynamic_reg_delta) {
  n_ = static_cast<Index>(K_lower.rows());
  dynamic_reg_eps_ = dynamic_reg_eps;
  dynamic_reg_delta_ = dynamic_reg_delta;

  // AMDOrdering's output is the *inverse* permutation -- see QdldlLdlt::analyzePattern for the
  // verification against Eigen::SimplicialLDLT that motivated always inverting it before use.
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index> perm_inv;
  Eigen::AMDOrdering<Index> amd;
  amd(K_lower.selfadjointView<Eigen::Lower>(), perm_inv);
  perm_ = perm_inv.inverse();

  // perm_inv.indices()(k) is the *original* (unpermuted) row/col that ends up at permuted
  // position k -- verified directly against SparseSelfAdjointView::twistedBy(perm_)'s actual
  // index mapping (permute_symm_to_symm), not assumed from the general permutation-matrix
  // convention, since getting this backwards would silently apply each pivot's regularization
  // to the wrong sign.
  expected_sign_.assign(static_cast<size_t>(n_), Scalar(1));
  for (Index k = 0; k < n_; ++k) {
    const Index orig = perm_inv.indices()(k);
    expected_sign_[static_cast<size_t>(k)] = (orig < nx) ? Scalar(1) : Scalar(-1);
  }

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
  bwork_.assign(static_cast<size_t>(n_), kUnused);
  iwork_.assign(3 * static_cast<size_t>(n_), 0);
}

namespace {
// Fraction of pivots that may be corrected in a single factorize() call before it's reported as
// a failure (Eigen::NumericalIssue) rather than Eigen::Success -- see RegularizedLdlt::info().
// A KKT matrix legitimately needs a handful of pivots nudged near convergence (a cone sitting
// close to its boundary is normal, not pathological); but if a large fraction need correcting at
// once, that's evidence the *pattern* of ill-conditioning is beyond what a per-pivot nudge can
// fix cleanly, and KktSystem's uniform whole-matrix bump (factorizeWithRetry()) should get a
// chance to react instead, the same way it would for the Eigen/QdldlLdlt backends. Not yet
// calibrated against a real hard instance -- see the CardilloMPI domino-scene report this was
// added for.
constexpr Scalar kMaxRegularizedFraction = 0.05;

// Davis/ECOS-style per-pivot check: `Dk` is acceptable iff it already has the expected sign
// (for this permuted position) with magnitude at least `eps`. `Dk * expected_sign < eps` catches
// both cases uniformly -- a wrong-signed pivot makes the product negative, a correctly-signed but
// too-small one makes it a small positive number below eps.
//
// On failure, `expected_sign * delta` is *added* to `Dk`, not substituted for it: the naturally-
// computed value (the raw diagonal entry minus the Schur-complement contributions from every
// already-eliminated column) still carries real information about how far off this specific
// pivot is, and folding the correction on top of it -- rather than discarding it for an
// absolute constant common to every corrected pivot regardless of the surrounding problem's
// scale -- keeps pivots that are merely borderline much closer to their natural value. A prior
// version of this function replaced Dk outright; on a real large-scale problem (13,600+
// near-degenerate SOC blocks with widely varying natural pivot scales) that made the IPM need
// 15x+ more iterations than the Eigen/QdldlLdlt backends on the identical problem, despite
// matching them exactly on smaller, less-degenerate problems -- see the CardilloMPI report this
// fixes for the full diagnosis. The additive correction can (rarely, for a pivot far from the
// expected sign/magnitude, which shouldn't occur for a well-formed quasi-definite KKT matrix)
// still fail the check; the pre-existing hard-floor behavior remains as a fallback purely to
// guarantee Dinv stays finite.
//
// Applied *before* `Dk` is used to compute Dinv and consumed by any later column's
// Schur-complement update -- this is what makes it a true inline correction rather than a
// post-hoc rescaling of an already-inconsistent factorization.
inline bool regularizePivot(Scalar& Dk, Scalar expected_sign, Scalar eps, Scalar delta) {
  if (Dk * expected_sign >= eps) return false;
  Dk += expected_sign * delta;
  if (Dk * expected_sign < eps) {
    Dk = expected_sign * delta;
  }
  return true;
}
}  // namespace

void RegularizedLdlt::factorize(const SparseMat& K_lower) {
  // Precondition: analyzePattern(K_lower, ...) has already been called with the same sparsity
  // pattern (matches Eigen::SimplicialLDLT's/QdldlLdlt's own analyzePattern-once/factorize-many
  // contract).
  A_upper_.selfadjointView<Eigen::Upper>() = K_lower.selfadjointView<Eigen::Lower>().twistedBy(perm_);
  A_upper_.makeCompressed();

  const Index n = n_;
  const Index* Ap = A_upper_.outerIndexPtr();
  const Index* Ai = A_upper_.innerIndexPtr();
  const Scalar* Ax = A_upper_.valuePtr();

  Index* Lp = Lp_.data();
  Index* Li = Li_.data();
  Scalar* Lx = Lx_.data();
  Scalar* D = D_.data();
  Scalar* Dinv = Dinv_.data();

  Index* yIdx = iwork_.data();
  Index* elimBuffer = iwork_.data() + n;
  Index* LNextSpaceInCol = iwork_.data() + 2 * n;
  Scalar* yVals = fwork_.data();
  unsigned char* yMarkers = bwork_.data();

  Lp[0] = 0;
  for (Index i = 0; i < n; ++i) {
    Lp[i + 1] = Lp[i] + Lnz_[static_cast<size_t>(i)];
    yMarkers[i] = kUnused;
    yVals[i] = Scalar(0);
    D[i] = Scalar(0);
    LNextSpaceInCol[i] = Lp[i];
  }

  Index num_regularized = 0;

  // First pivot: column 0 of an upper-triangular CSC matrix has exactly one entry (the
  // diagonal), so Ax[0] is D[0] directly -- no elimination to perform yet.
  D[0] = Ax[0];
  if (regularizePivot(D[0], expected_sign_[0], dynamic_reg_eps_, dynamic_reg_delta_)) {
    ++num_regularized;
  }
  Dinv[0] = Scalar(1) / D[0];

  for (Index k = 1; k < n; ++k) {
    Index nnzY = 0;
    const Index colEnd = Ap[k + 1];

    for (Index i = Ap[k]; i < colEnd; ++i) {
      const Index bidx = Ai[i];

      if (bidx == k) {
        D[k] = Ax[i];
        continue;
      }

      yVals[bidx] = Ax[i];

      Index nextIdx = bidx;
      if (yMarkers[nextIdx] == kUnused) {
        yMarkers[nextIdx] = kUsed;
        elimBuffer[0] = nextIdx;
        Index nnzE = 1;

        nextIdx = etree_[static_cast<size_t>(bidx)];
        while (nextIdx != kUnknown && nextIdx < k) {
          if (yMarkers[nextIdx] == kUsed) break;
          yMarkers[nextIdx] = kUsed;
          elimBuffer[nnzE] = nextIdx;
          ++nnzE;
          nextIdx = etree_[static_cast<size_t>(nextIdx)];
        }

        while (nnzE) {
          yIdx[nnzY++] = elimBuffer[--nnzE];
        }
      }
    }

    for (Index i = nnzY - 1; i >= 0; --i) {
      const Index cidx = yIdx[i];
      const Index colNext = LNextSpaceInCol[cidx];
      const Scalar yVals_cidx = yVals[cidx];

      for (Index j = Lp[cidx]; j < colNext; ++j) {
        yVals[Li[j]] -= Lx[j] * yVals_cidx;
      }

      Li[colNext] = k;
      Lx[colNext] = yVals_cidx * Dinv[cidx];
      D[k] -= yVals_cidx * Lx[colNext];
      LNextSpaceInCol[cidx]++;

      yVals[cidx] = Scalar(0);
      yMarkers[cidx] = kUnused;
    }

    if (regularizePivot(D[k], expected_sign_[static_cast<size_t>(k)], dynamic_reg_eps_,
                        dynamic_reg_delta_)) {
      ++num_regularized;
    }
    Dinv[k] = Scalar(1) / D[k];
  }

  num_regularized_pivots_ = num_regularized;
  // Every pivot clears dynamic_reg_eps_ with the correct sign (unlike upstream QDLDL_factor,
  // which aborts instead), but if too large a fraction needed correcting, don't silently accept
  // a factorization built from that many independently-doctored pivots -- report failure so
  // KktSystem's factorizeWithRetry() gets a chance to fall back to its uniform whole-matrix bump
  // instead, the same backstop the Eigen/QdldlLdlt backends get. See kMaxRegularizedFraction.
  info_ = (static_cast<Scalar>(num_regularized) > kMaxRegularizedFraction * static_cast<Scalar>(n))
              ? Eigen::NumericalIssue
              : Eigen::Success;
}

Vec RegularizedLdlt::solve(const Vec& rhs) const {
  Vec x = perm_ * rhs;
  QDLDL_solve(n_, Lp_.data(), Li_.data(), Lx_.data(), Dinv_.data(), x.data());
  return perm_.inverse() * x;
}

}  // namespace conicxx::detail
