// The elimination tree and the up-looking numeric factorization follow QDLDL
// (https://github.com/osqp/qdldl, Apache-2.0); see NOTICE.
#include "conicxx/kkt/quasidefinite_ldlt.h"

#include <Eigen/OrderingMethods>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace conicxx::detail {

namespace {
// Performance parameters of the supernodal factorization; results do not depend on them. Updates
// from supernodes narrower than kMinProductRank use scalar loops over the contiguous dense columns,
// wider ones a matrix product (a product with a smaller inner dimension costs more than it saves).
// The supernodal factorization is used only if the columns of L hold at least kMinMeanColumn
// entries on average; on sparser factors its per-supernode overhead outweighs the contiguous
// access (docs/design.md, "Supernodal factorization").
constexpr Index kMinProductRank = 4;
constexpr Index kMinMeanColumn = 16;
}  // namespace

bool QuasiDefiniteLdlt::analyzePattern(const SparseMat& K, const std::vector<signed char>& signs,
                                       const std::vector<Index>* perm) {
  analyzed_ = false;
  const Index n = static_cast<Index>(K.cols());
  if (K.rows() != K.cols() || static_cast<Index>(signs.size()) != n || !K.isCompressed()) {
    return false;
  }
  n_ = n;

  // Fill-reducing ordering. Eigen's ordering methods return the inverse permutation, whose
  // indices map pivot position -> original row.
  perm_.resize(static_cast<size_t>(n));
  if (perm) {
    if (static_cast<Index>(perm->size()) != n) return false;
    perm_ = *perm;
  } else if (n > 0) {
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index> pinv;
    Eigen::AMDOrdering<Index> amd;
    amd(K.selfadjointView<Eigen::Lower>(), pinv);
    std::copy(pinv.indices().data(), pinv.indices().data() + n, perm_.begin());
  }
  iperm_.assign(static_cast<size_t>(n), -1);
  for (Index k = 0; k < n; ++k) {
    const Index row = perm_[k];
    if (row < 0 || row >= n || iperm_[row] != -1) return false;  // not a permutation
    iperm_[row] = k;
  }

  buildPermutedUpper(K);
  std::vector<Index> etree, Lnz;
  eliminationTree(etree, Lnz);

  Index nnzL_estimate = 0;
  for (Index j = 0; j < n; ++j) nnzL_estimate += Lnz[j];
  supernodal_ = params.supernodal && n > 0 && nnzL_estimate >= kMinMeanColumn * n;
  if (supernodal_) {
    // Postorder the elimination tree (same fill) so that the columns of a supernode are contiguous.
    std::vector<Index> head(n, -1), next(n, -1), stack, post;
    post.reserve(n);
    for (Index j = n - 1; j >= 0; --j) {
      if (etree[j] != -1) {
        next[j] = head[etree[j]];
        head[etree[j]] = j;
      }
    }
    for (Index root = 0; root < n; ++root) {
      if (etree[root] != -1) continue;
      stack.push_back(root);
      while (!stack.empty()) {
        const Index j = stack.back();
        if (head[j] != -1) {  // descend into the next unvisited child
          const Index child = head[j];
          head[j] = next[child];
          stack.push_back(child);
        } else {
          stack.pop_back();
          post.push_back(j);
        }
      }
    }
    std::vector<Index> perm_post(n);
    for (Index k = 0; k < n; ++k) perm_post[k] = perm_[post[k]];
    perm_ = perm_post;
    for (Index k = 0; k < n; ++k) iperm_[perm_[k]] = k;
    buildPermutedUpper(K);
    eliminationTree(etree, Lnz);
  }

  Lp_.assign(n + 1, 0);
  for (Index j = 0; j < n; ++j) Lp_[j + 1] = Lp_[j] + Lnz[j];
  const Index nnzL = Lp_[n];
  Li_.resize(nnzL);
  Lx_.assign(nnzL, Scalar(0));

  // Row k of L is the reach of column k of C in the elimination tree. Filling rows in the order
  // k = 0, 1, ... leaves every column of L sorted by row.
  std::vector<Index> fill(Lp_.begin(), Lp_.end() - 1);
  std::vector<Index> mark(n, -1);
  for (Index k = 0; k < n; ++k) {
    mark[k] = k;
    for (Index p = Cp_[k]; p < Cp_[k + 1]; ++p) {
      for (Index i = Ci_[p]; mark[i] != k; i = etree[i]) {
        mark[i] = k;
        Li_[fill[i]++] = k;
      }
    }
  }

  // Row structure (transposed pattern) with ascending columns, a valid elimination order.
  Rp_.assign(n + 1, 0);
  for (Index p = 0; p < nnzL; ++p) ++Rp_[Li_[p] + 1];
  for (Index k = 0; k < n; ++k) Rp_[k + 1] += Rp_[k];
  Rj_.resize(nnzL);
  Rpos_.resize(nnzL);
  std::vector<Index> rfill(Rp_.begin(), Rp_.end() - 1);
  for (Index j = 0; j < n; ++j) {
    for (Index p = Lp_[j]; p < Lp_[j + 1]; ++p) {
      const Index q = rfill[Li_[p]]++;
      Rj_[q] = j;
      Rpos_[q] = p;
    }
  }

  if (supernodal_) analyzeSupernodes(etree);

  sign_.resize(n);
  for (Index k = 0; k < n; ++k) sign_[k] = signs[perm_[k]] >= 0 ? Scalar(1) : Scalar(-1);
  reg_.assign(n, Scalar(0));
  D_.assign(n, Scalar(0));
  Dinv_.assign(n, Scalar(0));
  y_.assign(n, Scalar(0));
  bp_.setZero(n);
  xp_.setZero(n);
  xt_.setZero(n);
  r_.setZero(n);
  rt_.setZero(n);
  xp2_.setZero(n);
  r2_.setZero(n);
  y2_.setZero(2 * n);
  num_dyn_reg_ = num_positive_ = 0;
  last_refine_steps_ = 0;
  analyzed_ = true;
  return true;
}

void QuasiDefiniteLdlt::eliminationTree(std::vector<Index>& etree, std::vector<Index>& Lnz) const {
  const Index n = n_;
  etree.assign(n, -1);
  Lnz.assign(n, 0);
  std::vector<Index> mark(n, -1);
  for (Index j = 0; j < n; ++j) {
    mark[j] = j;
    for (Index p = Cp_[j]; p < Cp_[j + 1]; ++p) {
      for (Index i = Ci_[p]; mark[i] != j; i = etree[i]) {
        if (etree[i] == -1) etree[i] = j;
        ++Lnz[i];
        mark[i] = j;
      }
    }
  }
}

void QuasiDefiniteLdlt::buildPermutedUpper(const SparseMat& K) {
  const Index n = n_;
  const Index* Kp = K.outerIndexPtr();
  const Index* Ki = K.innerIndexPtr();

  Cp_.assign(n + 1, 0);
  for (Index j = 0; j < n; ++j) {
    for (Index p = Kp[j]; p < Kp[j + 1]; ++p) {
      if (Ki[p] >= j) ++Cp_[std::max(iperm_[Ki[p]], iperm_[j]) + 1];
    }
  }
  for (Index j = 0; j < n; ++j) Cp_[j + 1] += Cp_[j];

  Ci_.resize(Cp_[n]);
  Cx_.assign(Cp_[n], Scalar(0));
  KtoC_.assign(Kp[n], -1);
  std::vector<Index> next(Cp_.begin(), Cp_.end() - 1);
  for (Index j = 0; j < n; ++j) {
    for (Index p = Kp[j]; p < Kp[j + 1]; ++p) {
      if (Ki[p] < j) continue;  // upper triangle: ignored
      const Index a = iperm_[Ki[p]], b = iperm_[j];
      const Index q = next[std::max(a, b)]++;
      Ci_[q] = std::min(a, b);
      KtoC_[p] = q;
    }
  }
}

void QuasiDefiniteLdlt::setStaticRegularization(const Vec& reg) {
  assert(reg.size() == n_);
  for (Index k = 0; k < n_; ++k) reg_[k] = reg[perm_[k]];
}

void QuasiDefiniteLdlt::analyzeSupernodes(const std::vector<Index>& etree) {
  const Index n = n_;
  // Fundamental supernodes: column j + 1 continues the supernode of j if it is j's parent and its
  // pattern is j's without row j + 1. The pattern of every column is then the tail of the row list
  // of its supernode, so the dense blocks map one to one onto Lx_.
  sn_ptr_.assign(1, 0);
  for (Index j = 0; j < n; ++j) {
    const bool continues = j + 1 < n && etree[j] == j + 1 && Lp_[j + 1] - Lp_[j] == Lp_[j + 2] - Lp_[j + 1] + 1;
    if (!continues) sn_ptr_.push_back(j + 1);
  }
  const Index nsn = static_cast<Index>(sn_ptr_.size()) - 1;
  sn_of_.resize(n);
  sn_rows_ptr_.assign(nsn + 1, 0);
  sn_x_ptr_.assign(nsn + 1, 0);
  max_rows_ = max_width_ = 0;
  for (Index sn = 0; sn < nsn; ++sn) {
    const Index f = sn_ptr_[sn], w = sn_ptr_[sn + 1] - f;
    const Index nr = 1 + (Lp_[f + 1] - Lp_[f]);  // the first column's own row plus its pattern
    for (Index c = f; c < f + w; ++c) sn_of_[c] = sn;
    sn_rows_ptr_[sn + 1] = sn_rows_ptr_[sn] + nr;
    sn_x_ptr_[sn + 1] = sn_x_ptr_[sn] + nr * w;
    max_rows_ = std::max(max_rows_, nr);
    max_width_ = std::max(max_width_, w);
  }
  sn_rows_.resize(sn_rows_ptr_[nsn]);
  for (Index sn = 0; sn < nsn; ++sn) {
    const Index f = sn_ptr_[sn];
    Index* rows = &sn_rows_[sn_rows_ptr_[sn]];
    rows[0] = f;
    std::copy(Li_.begin() + Lp_[f], Li_.begin() + Lp_[f + 1], rows + 1);
  }
  sn_x_.assign(sn_x_ptr_[nsn], Scalar(0));

  // Lower entries of C by column (C holds the upper triangle by column = the lower one by row).
  lo_p_.assign(n + 1, 0);
  diag_pos_.assign(n, -1);
  for (Index j = 0; j < n; ++j) {
    for (Index p = Cp_[j]; p < Cp_[j + 1]; ++p) {
      if (Ci_[p] == j) {
        diag_pos_[j] = p;
      } else {
        ++lo_p_[Ci_[p] + 1];
      }
    }
  }
  for (Index j = 0; j < n; ++j) lo_p_[j + 1] += lo_p_[j];
  lo_r_.resize(lo_p_[n]);
  lo_pos_.resize(lo_p_[n]);
  std::vector<Index> fill(lo_p_.begin(), lo_p_.end() - 1);
  for (Index j = 0; j < n; ++j) {
    for (Index p = Cp_[j]; p < Cp_[j + 1]; ++p) {
      if (Ci_[p] == j) continue;
      const Index q = fill[Ci_[p]]++;
      lo_r_[q] = j;
      lo_pos_[q] = p;
    }
  }

  map_.assign(n, 0);
  head_.assign(nsn, -1);
  next_.assign(nsn, -1);
  dpos_.assign(nsn, 0);
  upd_.assign(static_cast<size_t>(max_rows_) * max_width_, Scalar(0));
  wbuf_.assign(static_cast<size_t>(max_rows_) * max_width_, Scalar(0));
}

bool QuasiDefiniteLdlt::setPivot(Index k, Scalar d) {
  if (params.dynamic_enable && d * sign_[k] < params.dynamic_eps) {
    d = sign_[k] * params.dynamic_delta;
    ++num_dyn_reg_;
  }
  if (!std::isfinite(d) || d == Scalar(0)) return false;
  if (d > 0) ++num_positive_;
  D_[k] = d;
  Dinv_[k] = Scalar(1) / d;
  return true;
}

bool QuasiDefiniteLdlt::factorize(const SparseMat& K) {
  if (!analyzed_ || K.nonZeros() != static_cast<Eigen::Index>(KtoC_.size())) return false;
  const Scalar* Kx = K.valuePtr();
  const Index nnzK = static_cast<Index>(KtoC_.size());
  for (Index p = 0; p < nnzK; ++p) {
    if (KtoC_[p] >= 0) Cx_[KtoC_[p]] = Kx[p];
  }
  num_dyn_reg_ = num_positive_ = 0;
  return supernodal_ ? factorizeSupernodal() : factorizeSimplicial();
}

bool QuasiDefiniteLdlt::factorizeSimplicial() {
  for (Index k = 0; k < n_; ++k) {
    // Scatter column k of C: the strict upper part into y, the diagonal into the pivot.
    Scalar d = sign_[k] * reg_[k];
    for (Index p = Cp_[k]; p < Cp_[k + 1]; ++p) {
      if (Ci_[p] == k) {
        d += Cx_[p];
      } else {
        y_[Ci_[p]] = Cx_[p];
      }
    }
    // Row k of L: solve L(0:k-1, 0:k-1) y = c over the precomputed pattern.
    for (Index q = Rp_[k]; q < Rp_[k + 1]; ++q) {
      const Index j = Rj_[q], pos = Rpos_[q];
      const Scalar yj = y_[j];
      for (Index p = Lp_[j]; p < pos; ++p) y_[Li_[p]] -= Lx_[p] * yj;
      const Scalar l = yj * Dinv_[j];
      Lx_[pos] = l;
      d -= yj * l;
      y_[j] = Scalar(0);
    }
    if (!setPivot(k, d)) {
      std::fill(y_.begin(), y_.end(), Scalar(0));
      return false;
    }
  }
  return true;
}

bool QuasiDefiniteLdlt::factorizeSupernodal() {
  // Left-looking: supernode s gathers the updates of every earlier supernode d whose rows reach
  // into s's columns (linked lists head_/next_, dpos_[d] = first row of d not yet applied), then
  // factorizes its dense block. Within the block a panel factorization followed by a matrix-product
  // update of the trailing columns.
  using MatMap = Eigen::Map<Mat, 0, Eigen::OuterStride<>>;
  constexpr Index kPanel = 32;
  const Index nsn = static_cast<Index>(sn_ptr_.size()) - 1;
  std::fill(sn_x_.begin(), sn_x_.end(), Scalar(0));
  std::fill(head_.begin(), head_.end(), -1);

  for (Index sn = 0; sn < nsn; ++sn) {
    const Index f = sn_ptr_[sn], l = sn_ptr_[sn + 1], w = l - f;
    const Index* rows = &sn_rows_[sn_rows_ptr_[sn]];
    const Index nr = sn_rows_ptr_[sn + 1] - sn_rows_ptr_[sn];
    Scalar* X = &sn_x_[sn_x_ptr_[sn]];
    for (Index i = 0; i < nr; ++i) map_[rows[i]] = i;

    // Assemble the columns of C, the pivots starting from the static regularization.
    for (Index c = f; c < l; ++c) {
      Scalar* col = X + (c - f) * nr;
      col[c - f] = sign_[c] * reg_[c] + (diag_pos_[c] >= 0 ? Cx_[diag_pos_[c]] : Scalar(0));
      for (Index q = lo_p_[c]; q < lo_p_[c + 1]; ++q) col[map_[lo_r_[q]]] = Cx_[lo_pos_[q]];
    }

    // Updates from descendants: X(rows_d, cols) -= L_d D_d L_d(cols)'.
    Index d = head_[sn];
    head_[sn] = -1;
    while (d != -1) {
      const Index dnext = next_[d];
      const Index fd = sn_ptr_[d], wd = sn_ptr_[d + 1] - fd;
      const Index* rows_d = &sn_rows_[sn_rows_ptr_[d]];
      const Index nrd = sn_rows_ptr_[d + 1] - sn_rows_ptr_[d];
      const Scalar* Xd = &sn_x_[sn_x_ptr_[d]];
      const Index p1 = dpos_[d];
      Index p2 = p1;
      while (p2 < nrd && rows_d[p2] < l) ++p2;
      const Index m = nrd - p1, k = p2 - p1;
      if (wd >= kMinProductRank) {
        // W = L_d(p1:, :) D_d, U = W L_d(p1:p2, :)'
        MatMap W(wbuf_.data(), m, wd, Eigen::OuterStride<>(m));
        const MatMap Ld(const_cast<Scalar*>(Xd), nrd, wd, Eigen::OuterStride<>(nrd));
        for (Index t = 0; t < wd; ++t) W.col(t) = Ld.col(t).segment(p1, m) * D_[fd + t];
        MatMap U(upd_.data(), m, k, Eigen::OuterStride<>(m));
        U.noalias() = W * Ld.block(p1, 0, k, wd).transpose();
        for (Index jj = 0; jj < k; ++jj) {
          Scalar* col = X + (rows_d[p1 + jj] - f) * nr;
          for (Index ii = jj; ii < m; ++ii) col[map_[rows_d[p1 + ii]]] -= U(ii, jj);
        }
      } else {
        for (Index jj = 0; jj < k; ++jj) {
          Scalar* col = X + (rows_d[p1 + jj] - f) * nr;
          for (Index t = 0; t < wd; ++t) {
            const Scalar* Ldt = Xd + t * nrd;
            const Scalar coef = Ldt[p1 + jj] * D_[fd + t];
            if (coef == Scalar(0)) continue;
            for (Index ii = jj; ii < m; ++ii) col[map_[rows_d[p1 + ii]]] -= Ldt[p1 + ii] * coef;
          }
        }
      }
      dpos_[d] = p2;
      if (p2 < nrd) {
        const Index t = sn_of_[rows_d[p2]];
        next_[d] = head_[t];
        head_[t] = d;
      }
      d = dnext;
    }

    // Dense LDL' of the block: panels of kPanel columns, left-looking within a panel, then the
    // trailing columns of the block are updated with one matrix product per panel.
    MatMap B(X, nr, w, Eigen::OuterStride<>(nr));
    for (Index j0 = 0; j0 < w; j0 += kPanel) {
      const Index pw = std::min(kPanel, w - j0);
      for (Index j = j0; j < j0 + pw; ++j) {
        // column j: subtract the contributions of the panel columns j0..j-1
        Scalar dj = B(j, j);
        for (Index t = j0; t < j; ++t) {
          const Scalar v = B(j, t) * D_[f + t];
          dj -= B(j, t) * v;
          B.col(j).tail(nr - j - 1) -= B.col(t).tail(nr - j - 1) * v;
        }
        if (!setPivot(f + j, dj)) return false;
        B.col(j).tail(nr - j - 1) *= Dinv_[f + j];
      }
      const Index rest = w - j0 - pw;
      if (rest > 0) {
        // B(j0+pw:, j0+pw:w) -= L_panel D L_panel(j0+pw:w)'
        MatMap W(wbuf_.data(), nr - j0 - pw, pw, Eigen::OuterStride<>(nr - j0 - pw));
        for (Index t = 0; t < pw; ++t) W.col(t) = B.col(j0 + t).tail(nr - j0 - pw) * D_[f + j0 + t];
        B.block(j0 + pw, j0 + pw, nr - j0 - pw, rest).noalias() -=
            W * B.block(j0 + pw, j0, rest, pw).transpose();
      }
    }

    // Copy the columns into L and register s for the supernode of its first row below the block.
    for (Index c = f; c < l; ++c) {
      const Index jc = c - f;
      std::copy(X + jc * nr + jc + 1, X + (jc + 1) * nr, Lx_.begin() + Lp_[c]);
    }
    if (nr > w) {
      dpos_[sn] = w;
      const Index t = sn_of_[rows[w]];
      next_[sn] = head_[t];
      head_[t] = sn;
    }
  }
  return true;
}

void QuasiDefiniteLdlt::solveInPlace(Vec& x) const {
  for (Index j = 0; j < n_; ++j) {
    const Scalar xj = x[j];
    if (xj == Scalar(0)) continue;
    for (Index p = Lp_[j]; p < Lp_[j + 1]; ++p) x[Li_[p]] -= Lx_[p] * xj;
  }
  for (Index j = 0; j < n_; ++j) x[j] *= Dinv_[j];
  for (Index j = n_ - 1; j >= 0; --j) {
    Scalar s = x[j];
    for (Index p = Lp_[j]; p < Lp_[j + 1]; ++p) s -= Lx_[p] * x[Li_[p]];
    x[j] = s;
  }
}

void QuasiDefiniteLdlt::solvePairInPlace(Vec& y) const {
  for (Index j = 0; j < n_; ++j) {
    const Scalar a = y[2 * j], b = y[2 * j + 1];
    if (a == Scalar(0) && b == Scalar(0)) continue;
    for (Index p = Lp_[j]; p < Lp_[j + 1]; ++p) {
      const Index i = 2 * Li_[p];
      y[i] -= Lx_[p] * a;
      y[i + 1] -= Lx_[p] * b;
    }
  }
  for (Index j = 0; j < n_; ++j) {
    y[2 * j] *= Dinv_[j];
    y[2 * j + 1] *= Dinv_[j];
  }
  for (Index j = n_ - 1; j >= 0; --j) {
    Scalar a = y[2 * j], b = y[2 * j + 1];
    for (Index p = Lp_[j]; p < Lp_[j + 1]; ++p) {
      const Index i = 2 * Li_[p];
      a -= Lx_[p] * y[i];
      b -= Lx_[p] * y[i + 1];
    }
    y[2 * j] = a;
    y[2 * j + 1] = b;
  }
}

Scalar QuasiDefiniteLdlt::residual(const Vec& x, Vec& r) const {
  r = bp_;
  for (Index j = 0; j < n_; ++j) {
    const Scalar xj = x[j];
    Scalar s = 0;
    for (Index p = Cp_[j]; p < Cp_[j + 1]; ++p) {
      const Index i = Ci_[p];
      r[i] -= Cx_[p] * xj;
      if (i != j) s += Cx_[p] * x[i];
    }
    r[j] -= s;
  }
  return n_ > 0 ? r.lpNorm<Eigen::Infinity>() : Scalar(0);
}

Scalar QuasiDefiniteLdlt::refine(Vec& xp, Vec& r, int& steps) {
  // Iterative refinement on the exact matrix against bp_: accept a correction only if it reduces
  // the residual, and stop when it no longer shrinks by refine_stop_ratio per step. Without
  // refinement no residual is computed (returns -1).
  if (params.refine_max_iter <= 0) return Scalar(-1);
  Scalar norm = residual(xp, r);
  const Scalar tol =
      params.refine_abstol + params.refine_reltol * (n_ > 0 ? bp_.lpNorm<Eigen::Infinity>() : 0);
  for (int it = 0; it < params.refine_max_iter && norm > tol; ++it) {
    xt_ = r;
    solveInPlace(xt_);
    xt_ += xp;
    const Scalar trial = residual(xt_, rt_);
    ++steps;
    if (!(trial < norm)) break;
    xp.swap(xt_);
    r.swap(rt_);
    const bool fast = norm >= params.refine_stop_ratio * trial;
    norm = trial;
    if (!fast) break;
  }
  return norm;
}

Scalar QuasiDefiniteLdlt::solve(const Vec& b, Vec& x) {
  assert(b.size() == n_);
  if (x.size() != n_) x.resize(n_);
  for (Index k = 0; k < n_; ++k) bp_[k] = b[perm_[k]];
  xp_ = bp_;
  solveInPlace(xp_);
  last_refine_steps_ = 0;
  const Scalar norm = refine(xp_, r_, last_refine_steps_);
  for (Index k = 0; k < n_; ++k) x[perm_[k]] = xp_[k];
  return norm;
}

Scalar QuasiDefiniteLdlt::solvePair(const Vec& b1, Vec& x1, const Vec& b2, Vec& x2, bool refine2) {
  assert(b1.size() == n_ && b2.size() == n_);
  if (x1.size() != n_) x1.resize(n_);
  if (x2.size() != n_) x2.resize(n_);
  for (Index k = 0; k < n_; ++k) {
    y2_[2 * k] = b1[perm_[k]];
    y2_[2 * k + 1] = b2[perm_[k]];
  }
  solvePairInPlace(y2_);
  last_refine_steps_ = 0;
  if (refine2) {
    for (Index k = 0; k < n_; ++k) {
      bp_[k] = b2[perm_[k]];
      xp2_[k] = y2_[2 * k + 1];
    }
    refine(xp2_, r2_, last_refine_steps_);
    for (Index k = 0; k < n_; ++k) x2[perm_[k]] = xp2_[k];
  } else {
    for (Index k = 0; k < n_; ++k) x2[perm_[k]] = y2_[2 * k + 1];
  }
  for (Index k = 0; k < n_; ++k) {
    bp_[k] = b1[perm_[k]];
    xp_[k] = y2_[2 * k];
  }
  const Scalar norm = refine(xp_, r_, last_refine_steps_);
  for (Index k = 0; k < n_; ++k) x1[perm_[k]] = xp_[k];
  return norm;
}

}  // namespace conicxx::detail
