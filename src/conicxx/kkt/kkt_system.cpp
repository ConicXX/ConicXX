#include "conicxx/kkt/kkt_system.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

namespace conicxx::detail {

namespace {
bool sameSparsityPattern(const SparseMat& mat, const std::vector<Index>& outer_ref,
                          const std::vector<Index>& inner_ref) {
  if (mat.outerSize() + 1 != static_cast<Index>(outer_ref.size())) return false;
  if (!std::equal(mat.outerIndexPtr(), mat.outerIndexPtr() + mat.outerSize() + 1,
                   outer_ref.begin())) {
    return false;
  }
  if (mat.nonZeros() != static_cast<Index>(inner_ref.size())) return false;
  return std::equal(mat.innerIndexPtr(), mat.innerIndexPtr() + mat.nonZeros(), inner_ref.begin());
}
}  // namespace

bool KktSystem::setup(const SparseMat& P_upper, const SparseMat& A, const ConeSet& cones,
                      const Settings& settings) {
  n_ = static_cast<Index>(P_upper.rows());
  m_ = cones.totalDim();

  const auto& reg = settings.regularization;
  static_P_ = reg.static_P;
  static_nonneg_ = reg.static_nonneg;
  static_soc_ = reg.static_soc;
  static_zero_ = reg.static_zero;
  static_proportional_ = reg.static_proportional;
  static_zero_factor_only_ = reg.static_zero_factor_only;
  dynamic_eps_ = reg.dynamic_eps;
  dynamic_delta_ = reg.dynamic_delta;
  dynamic_on_zero_rows_ = reg.dynamic_on_zero_rows;

  refine_max_iter_ = settings.refine_max_iter;
  refine_reltol_ = settings.refine_reltol;
  refine_abstol_ = settings.refine_abstol;
  refine_stop_ratio_ = std::max(settings.refine_stop_ratio, Scalar(1));
  last_refinement_residual_ = 0;
  equality_rank_deficient_ = false;
  factorized_ = false;

  use_qdldl_ = settings.linear_solver == LinearSolverBackend::Qdldl;
  use_regularized_ = settings.linear_solver == LinearSolverBackend::RegularizedLdlt;
#ifndef CONICXX_HAVE_QDLDL
  if (use_qdldl_ || use_regularized_) {
    std::fprintf(stderr,
                 "[conicxx] warning: Settings::linear_solver requested %s, but conicxx was "
                 "built without it (CONICXX_WITH_QDLDL=OFF) -- falling back to Eigen\n",
                 use_regularized_ ? "RegularizedLdlt" : "Qdldl");
    use_qdldl_ = false;
    use_regularized_ = false;
  }
#endif

  sparsity_ = SparsityMap();
  p_diag_slots_.clear();
  p_diag_slots_.reserve(static_cast<size_t>(n_));
  p_offdiag_slots_.clear();
  a_slots_.clear();
  hs_blocks_.clear();
  nonneg_diag_slots_.clear();
  soc_diag_slots_.clear();
  zero_diag_slots_.clear();

  for (Index i = 0; i < n_; ++i) {
    p_diag_slots_.push_back(sparsity_.addEntry(i, i));
  }
  p_diag_value_idx_.assign(static_cast<size_t>(n_), -1);

  // k tracks the running index into P_upper.valuePtr(): InnerIterator visits a compressed
  // SparseMatrix's stored entries in exactly that order (increasing per column, columns in
  // order), so incrementing once per step keeps k equal to the valuePtr() position throughout.
  {
    Index k = 0;
    for (Index c = 0; c < P_upper.outerSize(); ++c) {
      for (SparseMat::InnerIterator it(P_upper, c); it; ++it, ++k) {
        const Index r = static_cast<Index>(it.row());
        if (r > c) continue;  // below diagonal; input assumed upper-triangular
        if (r == c) {
          p_diag_value_idx_[static_cast<size_t>(c)] = k;
          continue;
        }
        const Index slot = sparsity_.addEntry(c, r);
        p_offdiag_slots_.emplace_back(slot, k);
      }
    }
  }

  {
    Index k = 0;
    for (Index c = 0; c < A.outerSize(); ++c) {
      for (SparseMat::InnerIterator it(A, c); it; ++it, ++k) {
        const Index r = static_cast<Index>(it.row());
        const Index slot = sparsity_.addEntry(n_ + r, c);
        a_slots_.emplace_back(slot, k);
      }
    }
  }

  Index max_hs_entries = 0;
  for (Index bi = 0; bi < cones.numBlocks(); ++bi) {
    const ConeBase& blk = cones.block(bi);
    const Index off = cones.blockOffset(bi);
    const Index d = blk.dim();
    HsBlockSlots hbs;
    hbs.type = blk.type();
    if (blk.type() == ConeType::SecondOrder) {
      // The NT scaling block Hs = W^T W is genuinely dense for SOC blocks.
      for (Index a = 0; a < d; ++a) {
        for (Index b = 0; b <= a; ++b) {
          hbs.slots.push_back(sparsity_.addEntry(n_ + off + a, n_ + off + b));
          hbs.is_diag.push_back(a == b ? 1 : 0);
        }
      }
    } else {
      // Zero/Nonnegative cones: Hs is (at most) diagonal, so only the
      // diagonal needs a slot in K (it also carries static/dynamic
      // regularization). Registering the full triangle here would
      // explicitly store an always-zero-valued dense block in K's
      // sparsity pattern -- for a large equality (Zero-cone) or
      // nonnegative block this causes catastrophic fill-in during
      // symbolic/numeric factorization for a block that mathematically
      // contributes nothing off-diagonal.
      for (Index a = 0; a < d; ++a) {
        hbs.slots.push_back(sparsity_.addEntry(n_ + off + a, n_ + off + a));
        hbs.is_diag.push_back(1);
      }
    }
    assert(static_cast<Index>(hbs.slots.size()) == blk.numHsEntries());
    max_hs_entries = std::max(max_hs_entries, static_cast<Index>(hbs.slots.size()));

    std::vector<Index>* bucket = nullptr;
    switch (hbs.type) {
      case ConeType::Zero: bucket = &zero_diag_slots_; break;
      case ConeType::Nonnegative: bucket = &nonneg_diag_slots_; break;
      case ConeType::SecondOrder: bucket = &soc_diag_slots_; break;
    }
    for (size_t i = 0; i < hbs.slots.size(); ++i) {
      if (hbs.is_diag[i]) bucket->push_back(hbs.slots[i]);
    }
    hs_blocks_.push_back(std::move(hbs));
  }
  hs_entries_scratch_.resize(max_hs_entries);

  all_diag_slots_ = p_diag_slots_;
  all_diag_slots_.insert(all_diag_slots_.end(), nonneg_diag_slots_.begin(), nonneg_diag_slots_.end());
  all_diag_slots_.insert(all_diag_slots_.end(), soc_diag_slots_.begin(), soc_diag_slots_.end());
  all_diag_slots_.insert(all_diag_slots_.end(), zero_diag_slots_.begin(), zero_diag_slots_.end());

  is_zero_row_.assign(static_cast<size_t>(n_ + m_), 0);
  for (Index bi = 0; bi < cones.numBlocks(); ++bi) {
    const ConeBase& blk = cones.block(bi);
    if (blk.type() != ConeType::Zero) continue;
    const Index off = cones.blockOffset(bi);
    for (Index a = 0; a < blk.dim(); ++a) {
      is_zero_row_[static_cast<size_t>(n_ + off + a)] = 1;
    }
  }

  K_exact_ = sparsity_.finalize(n_ + m_, n_ + m_);
  K_fact_ = K_exact_;  // same structure; values diverge via writes/rebuildKFactDiagonal() below
  backendAnalyzePattern(K_fact_);

  refine_r_.resize(n_ + m_);
  refine_dx_.resize(n_ + m_);

  p_outer_ref_.assign(P_upper.outerIndexPtr(), P_upper.outerIndexPtr() + P_upper.outerSize() + 1);
  p_inner_ref_.assign(P_upper.innerIndexPtr(), P_upper.innerIndexPtr() + P_upper.nonZeros());
  a_outer_ref_.assign(A.outerIndexPtr(), A.outerIndexPtr() + A.outerSize() + 1);
  a_inner_ref_.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());

  // Write the initial P/A values directly rather than going through updateData() (which would
  // also factorize): at this point in setup(), the Hs block hasn't been written yet (that's
  // updateScalingAndFactorize() below), so K_fact_'s (2,2) block is transiently just whatever
  // sparsity_.finalize() zero-initialized it to -- for a zero-cone row (0 static regularization
  // by design) that transient state can genuinely fail to factorize even with the T2.5
  // factor-only fallback, for problems where the *final*, properly-scaled system factorizes
  // fine. Factorizing only once, after Hs is populated, avoids gating setup() on a factorization
  // of a matrix state the solver never actually uses.
  if (!writePAValues(&P_upper, &A)) return false;
  return updateScalingAndFactorize(cones);
}

bool KktSystem::writePAValues(const SparseMat* P_upper, const SparseMat* A) {
  if (P_upper) {
    if (!sameSparsityPattern(*P_upper, p_outer_ref_, p_inner_ref_)) return false;
    const Scalar* Px = P_upper->valuePtr();
    for (Index i = 0; i < n_; ++i) {
      const Index k = p_diag_value_idx_[static_cast<size_t>(i)];
      sparsity_.setValue(K_exact_, p_diag_slots_[static_cast<size_t>(i)], k >= 0 ? Px[k] : Scalar(0));
    }
    for (const auto& [slot, k] : p_offdiag_slots_) {
      const Scalar v = Px[k];
      sparsity_.setValue(K_exact_, slot, v);
      sparsity_.setValue(K_fact_, slot, v);  // off-diagonal P entries are never regularized
    }
  }
  if (A) {
    if (!sameSparsityPattern(*A, a_outer_ref_, a_inner_ref_)) return false;
    const Scalar* Ax = A->valuePtr();
    for (const auto& [slot, k] : a_slots_) {
      const Scalar v = Ax[k];
      sparsity_.setValue(K_exact_, slot, v);
      sparsity_.setValue(K_fact_, slot, v);  // A block is never regularized
    }
  }
  return true;
}

bool KktSystem::updateData(const SparseMat* P_upper, const SparseMat* A) {
  if (!writePAValues(P_upper, A)) return false;
  rebuildKFactDiagonal();
  return factorizeWithRetry();
}

bool KktSystem::updateScalingAndFactorize(const ConeSet& cones) {
  for (Index bi = 0; bi < cones.numBlocks(); ++bi) {
    const ConeBase& blk = cones.block(bi);
    HsBlockSlots& hbs = hs_blocks_[static_cast<size_t>(bi)];
    const Index ne = static_cast<Index>(hbs.slots.size());
    Eigen::Ref<Vec> entries = hs_entries_scratch_.head(ne);
    blk.writeHsLowerTriangle(entries);
    for (Index i = 0; i < ne; ++i) {
      const Scalar v = -entries[i];
      const Index slot = hbs.slots[static_cast<size_t>(i)];
      sparsity_.setValue(K_exact_, slot, v);
      // Diagonal entries are rewritten in bulk by rebuildKFactDiagonal() below (it needs the
      // full current diagonal for the proportional term first); off-diagonal ones (SOC only) are
      // never regularized, so write them straight through here.
      if (!hbs.is_diag[static_cast<size_t>(i)]) sparsity_.setValue(K_fact_, slot, v);
    }
  }
  rebuildKFactDiagonal();
  return factorizeWithRetry();
}

void KktSystem::rebuildKFactDiagonal() {
  Scalar max_abs_diag = 0.0;
  for (Index slot : all_diag_slots_) {
    max_abs_diag = std::max(max_abs_diag, std::abs(sparsity_.getValue(K_exact_, slot)));
  }
  const Scalar prop = static_proportional_ * max_abs_diag;

  for (Index slot : p_diag_slots_) {
    sparsity_.setValue(K_fact_, slot, sparsity_.getValue(K_exact_, slot) + static_P_ + prop);
  }
  for (Index slot : nonneg_diag_slots_) {
    sparsity_.setValue(K_fact_, slot, sparsity_.getValue(K_exact_, slot) - static_nonneg_ - prop);
  }
  for (Index slot : soc_diag_slots_) {
    sparsity_.setValue(K_fact_, slot, sparsity_.getValue(K_exact_, slot) - static_soc_ - prop);
  }
  for (Index slot : zero_diag_slots_) {
    // No proportional term on zero (equality) rows -- see Settings::RegularizationSettings.
    sparsity_.setValue(K_fact_, slot, sparsity_.getValue(K_exact_, slot) - static_zero_);
  }
  last_extra_p_ = last_extra_nonneg_ = last_extra_soc_ = last_extra_zero_ = 0.0;
}

void KktSystem::bumpDiagSlots(const std::vector<Index>& slots, Scalar delta, Scalar sign) {
  if (delta == 0.0) return;
  for (Index slot : slots) {
    sparsity_.setValue(K_fact_, slot, sparsity_.getValue(K_fact_, slot) + sign * delta);
  }
}

void KktSystem::backendAnalyzePattern(const SparseMat& K) {
#ifdef CONICXX_HAVE_QDLDL
  if (use_regularized_) {
    ldlt_reg_.analyzePattern(K, is_zero_row_, n_, dynamic_eps_, dynamic_delta_);
    return;
  }
  if (use_qdldl_) {
    ldlt_qdldl_.analyzePattern(K);
    return;
  }
#endif
  ldlt_eigen_.analyzePattern(K);
}

void KktSystem::backendFactorize(const SparseMat& K) {
#ifdef CONICXX_HAVE_QDLDL
  if (use_regularized_) {
    ldlt_reg_.factorize(K);
    return;
  }
  if (use_qdldl_) {
    ldlt_qdldl_.factorize(K);
    return;
  }
#endif
  ldlt_eigen_.factorize(K);
}

Eigen::ComputationInfo KktSystem::backendInfo() const {
#ifdef CONICXX_HAVE_QDLDL
  if (use_regularized_) return ldlt_reg_.info();
  if (use_qdldl_) return ldlt_qdldl_.info();
#endif
  return ldlt_eigen_.info();
}

Scalar KktSystem::backendMinAbsD() const {
#ifdef CONICXX_HAVE_QDLDL
  if (use_regularized_) return ldlt_reg_.minAbsD();
  if (use_qdldl_) return ldlt_qdldl_.minAbsD();
#endif
  // Eigen::SimplicialLDLT::vectorD() returns a const reference to its own internal storage (no
  // copy), so this doesn't materialize an owned Vec either.
  return ldlt_eigen_.vectorD().array().abs().minCoeff();
}

void KktSystem::backendSolve(const Vec& rhs, Eigen::Ref<Vec> out) const {
#ifdef CONICXX_HAVE_QDLDL
  if (use_regularized_) {
    ldlt_reg_.solve(rhs, out);
    return;
  }
  if (use_qdldl_) {
    ldlt_qdldl_.solve(rhs, out);
    return;
  }
#endif
  out = ldlt_eigen_.solve(rhs);
}

bool KktSystem::tryFactorizeCurrentKFact() {
  // Pattern is fixed after setup()'s one-time analyzePattern() call, so only factorize() (numeric
  // refactorization) is needed here -- never compute(), which would redo the fill-reducing
  // symbolic analysis from scratch on every IPM iteration.
  backendFactorize(K_fact_);
  if (backendInfo() != Eigen::Success) return false;
  const Scalar minAbsD = dim() > 0 ? backendMinAbsD() : Scalar(1.0);
  return std::isfinite(minAbsD) && minAbsD > dynamic_eps_;
}

bool KktSystem::factorizeWithRetry() {
  equality_rank_deficient_ = false;
  // rebuildKFactDiagonal() (called by both callers just before this) already reset K_fact_ to
  // the pure-static baseline, so the ladder starts fresh from extra = 0 here.
  Scalar extra_p = 0.0, extra_nonneg = 0.0, extra_soc = 0.0, extra_zero = 0.0;
  // 18 attempts of x10 (capping out around 1e-3 above dynamic_delta_'s ~1e-10 floor) was
  // calibrated for a pivot just barely underflowing dynamic_eps_ -- verified against a real
  // large multi-contact problem to be far too weak for a badly-scaled cone block that needs
  // several more orders of magnitude of regularization to clear the same check. Growing further
  // (still x10, so small-bump cases still resolve in a couple of attempts) costs nothing extra
  // when the first attempt already succeeds.
  constexpr int kMaxRetries = 18;
  for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
    if (tryFactorizeCurrentKFact()) {
      factorized_ = true;
      last_extra_p_ = extra_p;
      last_extra_nonneg_ = extra_nonneg;
      last_extra_soc_ = extra_soc;
      last_extra_zero_ = extra_zero;
      return true;
    }
    const Scalar new_extra_p = std::max(extra_p * 10.0, dynamic_delta_);
    const Scalar new_extra_nonneg = std::max(extra_nonneg * 10.0, dynamic_delta_);
    const Scalar new_extra_soc = std::max(extra_soc * 10.0, dynamic_delta_);
    bumpDiagSlots(p_diag_slots_, new_extra_p - extra_p, +1.0);
    bumpDiagSlots(nonneg_diag_slots_, new_extra_nonneg - extra_nonneg, -1.0);
    bumpDiagSlots(soc_diag_slots_, new_extra_soc - extra_soc, -1.0);
    extra_p = new_extra_p;
    extra_nonneg = new_extra_nonneg;
    extra_soc = new_extra_soc;
    if (dynamic_on_zero_rows_) {
      const Scalar new_extra_zero = std::max(extra_zero * 10.0, dynamic_delta_);
      bumpDiagSlots(zero_diag_slots_, new_extra_zero - extra_zero, -1.0);
      extra_zero = new_extra_zero;
    }
  }

  // The P/nonneg/soc ladder above exhausted without success -- restore K_fact_'s diagonal to the
  // pure-static baseline before trying anything else, via a full rebuild (re-read from
  // K_exact_), NOT by subtracting the accumulated bump back out. Leaving the bump in place (as a
  // prior version of this function did) means the zero-row fallback below would factorize a
  // K_fact_ whose P/SOC diagonal has already been pushed to ~extra*10^18, an essentially
  // different matrix from the real problem. Subtracting it back out is *also* unsafe once extra
  // has grown that large: `static_P (1e-8) + extra (~2e10)` rounds to exactly 2e10 in double
  // precision (1e-8 is ~5e-19 relative to 2e10, far below machine epsilon), so the original
  // baseline is not just imprecise but completely and unrecoverably lost to the addition itself
  // -- subtracting extra back out then gives exactly 0, not 1e-8 (caught via a real repro: a
  // single zero-cone row with no duplication at all, where this exact cancellation left the
  // P-block diagonal at 0 during the zero-row fallback, corrupting a perfectly well-posed
  // problem). A full re-read from K_exact_ sidesteps the whole problem instead of trying to make
  // the arithmetic reversible.
  rebuildKFactDiagonal();

  // The ladder above never touches zero rows unless dynamic_on_zero_rows_ is set. If it's still
  // failing and there are zero-cone rows, try the T2.5 factor-only delta on those rows alone,
  // against the clean static baseline -- a preconditioner, not a problem perturbation: refinement
  // against K_exact_ removes its effect from the returned solution. Escalate it (x10 per attempt,
  // same shape as the P/nonneg/soc ladder above) rather than trying it once at its floor value:
  // Settings::regularization.static_zero_factor_only's default (1e-10) is calibrated to be
  // negligible after refinement, but a zero row's Schur complement is eliminated against
  // whatever scale the *other* blocks happen to be at (e.g. an O(1) SOC Hs block), so a single
  // fixed attempt that far below that scale routinely isn't enough to make the (2,2) block
  // negative definite -- required for Eigen::SimplicialLDLT's unpivoted factorization to succeed
  // at all, regardless of whether the underlying problem is actually rank-deficient.
  if (!dynamic_on_zero_rows_ && static_zero_factor_only_ != 0.0 && !zero_diag_slots_.empty()) {
    Scalar zero_delta = 0.0;
    constexpr int kMaxZeroFactorAttempts = 10;
    for (int attempt = 0; attempt < kMaxZeroFactorAttempts; ++attempt) {
      const Scalar new_zero_delta = std::max(zero_delta * 10.0, static_zero_factor_only_);
      bumpDiagSlots(zero_diag_slots_, new_zero_delta - zero_delta, -1.0);
      zero_delta = new_zero_delta;
      if (tryFactorizeCurrentKFact()) {
        factorized_ = true;
        last_extra_p_ = last_extra_nonneg_ = last_extra_soc_ = 0.0;
        last_extra_zero_ = zero_delta;
        return true;
      }
    }
    // Even escalating the factor-only preconditioner couldn't produce a usable factorization --
    // this is the rank-deficiency signal, not a numerics problem to paper over further.
    equality_rank_deficient_ = true;
  }

  factorized_ = false;
  return false;
}

bool KktSystem::escalateAndResolve(const Vec& rhs, Vec& x_out) {
  // factorizeWithRetry()'s per-pivot |D| >= dynamic_eps_ check can be satisfied by a bump far too
  // small to matter -- e.g. a P-diagonal pivot already sitting exactly at the static-
  // regularization floor clears that check trivially, even while a *different* pivot has grown
  // many orders of magnitude larger (from a near-degenerate cone's NT scaling), leaving the
  // min/max pivot spread enormous and the actual solve for a specific rhs prone to overflow
  // despite the "successful" factorization. The x10-per-attempt ladder used there is calibrated
  // to nudge a small pivot just past dynamic_eps_, which is far too slow to close a spread of
  // several orders of magnitude within a handful of attempts -- so here we grow far more
  // aggressively (x1000/attempt) and use the one signal that actually reflects whether it
  // worked: does backendSolve(rhs) come back finite, not just whether some abstract
  // pivot-magnitude heuristic passed. Zero rows are never touched here either, same reasoning as
  // factorizeWithRetry().
  constexpr int kMaxEscalations = 10;
  Scalar extra_p = last_extra_p_, extra_nonneg = last_extra_nonneg_, extra_soc = last_extra_soc_;
  for (int attempt = 0; attempt < kMaxEscalations; ++attempt) {
    const Scalar new_extra_p = std::max(extra_p * 1000.0, dynamic_delta_);
    const Scalar new_extra_nonneg = std::max(extra_nonneg * 1000.0, dynamic_delta_);
    const Scalar new_extra_soc = std::max(extra_soc * 1000.0, dynamic_delta_);
    bumpDiagSlots(p_diag_slots_, new_extra_p - extra_p, +1.0);
    bumpDiagSlots(nonneg_diag_slots_, new_extra_nonneg - extra_nonneg, -1.0);
    bumpDiagSlots(soc_diag_slots_, new_extra_soc - extra_soc, -1.0);
    extra_p = new_extra_p;
    extra_nonneg = new_extra_nonneg;
    extra_soc = new_extra_soc;
    if (!tryFactorizeCurrentKFact()) continue;
    backendSolve(rhs, x_out);
    if (x_out.allFinite()) {
      last_extra_p_ = extra_p;
      last_extra_nonneg_ = extra_nonneg;
      last_extra_soc_ = extra_soc;
      return true;
    }
  }
  return false;
}

Scalar KktSystem::solve(const Vec& rhs, Vec& x_out) {
  // backendSolve() writes into x_out via Eigen::Ref, which -- unlike a plain Vec assignment --
  // does not resize a mismatched target, so a caller passing a not-yet-sized (or stale-sized)
  // x_out here would be writing out of bounds. Callers on the per-iteration hot path pre-size
  // their buffer once and never hit this; this guards the (cheap, no-op once sized) general case.
  if (x_out.size() != n_ + m_) x_out.resize(n_ + m_);
  backendSolve(rhs, x_out);
  if (!x_out.allFinite()) escalateAndResolve(rhs, x_out);

  const Scalar rhsNorm = std::max(rhs.norm(), Scalar(1e-30));
  const Scalar tol = refine_abstol_ + refine_reltol_ * rhsNorm;

  refine_r_.noalias() = rhs - K_exact_.selfadjointView<Eigen::Lower>() * x_out;
  Scalar resNorm = refine_r_.norm();

  for (int it = 0; it < refine_max_iter_ && resNorm > tol; ++it) {
    backendSolve(refine_r_, refine_dx_);
    x_out += refine_dx_;
    refine_r_.noalias() = rhs - K_exact_.selfadjointView<Eigen::Lower>() * x_out;
    const Scalar newResNorm = refine_r_.norm();
    const bool shrinking_enough = newResNorm <= resNorm / refine_stop_ratio_;
    resNorm = newResNorm;
    if (!shrinking_enough) break;  // stagnating -- more iterations won't help
  }

  // factorizeWithRetry() only flags equality_rank_deficient_ when even the T2.5 factor-only
  // delta failed to factorize at all -- but a factorization that nominally succeeds with that
  // delta can still be so ill-conditioned (the delta broke an exact tie between near-duplicate
  // equality rows by a mere static_zero_factor_only) that refinement against the true K_exact_
  // stalls well above tolerance, same symptom, just caught one level later. Both are the
  // practical definition of "rank-deficient in floating point," so both set the flag -- silently
  // returning a stalled-residual solve as if it were a clean solution is exactly the "wrong
  // Solved" this is meant to prevent. Never cleared here (only factorizeWithRetry() clears it, so
  // any solve() call within one factorization's lifetime can still flag it).
  if (last_extra_zero_ != 0.0 && resNorm > tol) {
    equality_rank_deficient_ = true;
  }

  last_refinement_residual_ = resNorm / rhsNorm;
  return last_refinement_residual_;
}

}  // namespace conicxx::detail
