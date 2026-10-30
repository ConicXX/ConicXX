#include "conicxx/ipm/coulomb_ipm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace conicxx::detail {

namespace {

Scalar weightedInfNorm(const Eigen::Ref<const Vec>& v, const Vec& weights) {
  if (v.size() == 0) return 0.0;
  if (weights.size() == 0) return v.cwiseAbs().maxCoeff();
  return (v.array() * weights.head(v.size()).array()).abs().maxCoeff();
}

Scalar tailNorm(const Scalar* x, Index d) {
  Scalar t = 0;
  for (Index j = 1; j < d; ++j) t += x[j] * x[j];
  return std::sqrt(t);
}

}  // namespace

CoulombIpm::CoulombIpm(const ConeSpec& spec, Index n, Index m) : map_(spec), n_(n), m_(m) {
  for (Vec* v : {&Px_, &Atz_, &rd_, &dx_aff_, &dx_, &best_x_, &tx_, &ux_, &px_, &cx_, &ax_, &gx_, &gdx_}) v->setZero(n);
  for (Vec* v : {&us_, &uz_, &ps_, &pz_, &cs_, &cz_, &as_, &az_, &ws_, &wz_, &gs_, &gz_, &gds_, &gdz_}) v->setZero(m);
  for (Vec* v : {&Ax_, &rp_, &mob_, &tmp_m_, &tmp2_m_, &tz_, &ts_, &row0_, &prod_, &dz_aff_, &ds_aff_, &dz_, &ds_, &lambda_,
                 &lambda_prod_, &Winv_ds_aff_, &W_dz_aff_, &corrector_, &ds_comb_, &lambda_inv_ds_,
                 &ds_const_, &Hz_, &s_trial_, &z_trial_, &best_s_, &best_z_}) {
    v->setZero(m);
  }
  rhs_.setZero(n + m);
  sol_.setZero(n + m);
}

void CoulombIpm::linearResiduals(const CoulombProblem& pb, const Vec& x, const Vec& s, const Vec& z) {
  Pmul(pb, x, Px_);
  Atz_.noalias() = pb.A.transpose() * z;
  rd_ = Px_ + pb.q + Atz_;
  Ax_.noalias() = pb.A * x;
  rp_ = Ax_ + s - pb.b;
  if (pb.has_compliance) rp_.head(pb.zero_dim) -= pb.compliance.cwiseProduct(z.head(pb.zero_dim));
}

CoulombIpm::Metrics CoulombIpm::metrics(const CoulombProblem& pb, const Settings& settings, const Vec& x,
                                        const Vec& s, const Vec& z) {
  linearResiduals(pb, x, s, z);
  tmp_m_ = rp_;
  map_.subtractExactShift(s, tmp_m_, 1.0);

  Metrics mt;
  const Scalar normx = weightedInfNorm(x, pb.d);
  const Scalar norms = weightedInfNorm(s, pb.einv);
  const Scalar normz = pb.cinv * weightedInfNorm(z, pb.e);
  mt.res_dual = pb.cinv * weightedInfNorm(rd_, pb.dinv) / std::max(Scalar(1), pb.normq + normx + normz);
  mt.res_primal = weightedInfNorm(tmp_m_, pb.einv) / std::max(Scalar(1), pb.normb + normx + norms);

  // Complementarity per cone (orthant rows: cones of dimension 1) in velocity units, user units:
  //   v_i = ||s_i o z_i|| / (||z_i|| + m_i ||s_i||),  m_i = 1 / max_rows (A P^-1 A')_rr,
  // the effective mass of the contact (diagonal of P). For an active contact (s_i -> 0) v_i is its
  // velocity error, for a separating one (z_i -> 0) the velocity z_i / m_i its residual impulse
  // induces. Accepted if v_i <= tol_gap_abs + tol_gap_rel ||s_i|| for every cone (absolute
  // velocity tolerance plus one relative to the contact's own speed). Global scales (max norms over
  // all bodies) would let every contact keep an error of tol times the fastest velocity: on
  // interior points every active contact separates (u_i0 > 0), so these errors add up to energy
  // injected by thousands of resting contacts. Unscaled: s_i = s_hat_i / e_i,
  // z_i = cinv e_i z_hat_i, s_i o z_i = cinv (s_hat o z_hat)_i, and with A = E^-1 A_hat D^-1,
  // P = D^-1 P_hat D^-1 / cinv: (A P^-1 A')_rr = (A_hat P_hat^-1 A_hat')_rr / (cinv e_r^2).
  mob_.setZero();
  for (Index cidx = 0; cidx < pb.A.outerSize(); ++cidx) {
    Scalar pjj = 0;
    for (SparseMat::InnerIterator it(pb.P, cidx); it; ++it) {
      if (it.row() == cidx) pjj = it.value();
    }
    if (!(pjj > 0)) continue;
    for (SparseMat::InnerIterator it(pb.A, cidx); it; ++it) mob_[it.row()] += it.value() * it.value() / pjj;
  }
  const ConeSet& cones = pb.cones;
  cones.product(s, z, prod_);
  Scalar comp_abs = 0;
  mt.comp = 0;
  mt.comp_bound = std::numeric_limits<Scalar>::infinity();
  const auto addCone = [&](Index off, Index d, Scalar pn, Scalar sn, Scalar zn) {
    const Scalar e = pb.e.size() > 0 ? pb.e[off] : Scalar(1);
    const Scalar g = mob_.segment(off, d).maxCoeff() / (pb.cinv * e * e);  // 1 / m_i, user units
    const Scalar s_u = sn / e, z_u = pb.cinv * e * zn;
    const Scalar den = g > 0 ? z_u + s_u / g : z_u;       // ||z_i|| + m_i ||s_i||
    const Scalar tol = settings.tol_gap_abs + settings.tol_gap_rel * s_u;
    comp_abs = std::max(comp_abs, pn);
    if (den > 0) mt.comp = std::max(mt.comp, pb.cinv * pn / (den * tol));
    mt.comp_bound = std::min(mt.comp_bound, tol * den / pb.cinv);  // ||s_hat o z_hat|| allowed
  };
  for (Index bi = 0; bi < cones.numBlocks(); ++bi) {
    const Index off = cones.blockOffset(bi), d = cones.blockDim(bi);
    if (cones.blockType(bi) == ConeType::Nonnegative) {
      for (Index j = off; j < off + d; ++j) addCone(j, 1, std::abs(prod_[j]), std::abs(s[j]), std::abs(z[j]));
    } else if (cones.blockType(bi) == ConeType::SecondOrder) {
      addCone(off, d, prod_.segment(off, d).norm(), s.segment(off, d).norm(), z.segment(off, d).norm());
    }
  }
  if (!std::isfinite(mt.comp_bound)) mt.comp_bound = 0;
  mt.comp_abs = pb.cinv * comp_abs;
  mt.merit = std::max({mt.res_primal / settings.tol_feas, mt.res_dual / settings.tol_feas, mt.comp});
  return mt;
}

bool CoulombIpm::solveNewton(const CoulombProblem& pb, const Vec& ds_const, Vec& dx, Vec& dz, Vec& ds) {
  // [P A'; A -(I - E L) Hs] (dx, dz) = (-rd, -rp + (I - E L) ds_const), ds = -(Hs dz + ds_const).
  tmp_m_ = ds_const;
  map_.applyBinv(tmp_m_, 1.0);
  rhs_.head(n_) = -rd_;
  rhs_.tail(m_) = tmp_m_ - rp_;
  pb.kkt.solve(rhs_, sol_);
  dx = sol_.head(n_);
  dz = sol_.tail(m_);
  pb.cones.mulHs(dz, Hz_);
  ds = -(Hz_ + ds_const);
  return dx.allFinite() && dz.allFinite() && ds.allFinite();
}

bool CoulombIpm::linearize(const CoulombProblem& pb, const Vec& x, const Vec& s, const Vec& z, Scalar eps,
                           bool have_residuals) {
  if (!have_residuals) linearResiduals(pb, x, s, z);
  map_.evaluate(s, eps);
  map_.subtractShift(rp_, 1.0);
  // Newton matrix: row 0 of every Coulomb block becomes Hs(0,:) - (Hs L')'.
  pb.cones.updateScaling(s, z);
  pb.cones.mulHs(map_.jac(), row0_);
  row0_ = -row0_;
  return pb.kkt.updateScalingAndFactorize(pb.cones, map_.blocks(), row0_);
}

void CoulombIpm::centeringConst(const CoulombProblem& pb, const Vec& z, Scalar tau) {
  ConeSet& cones = pb.cones;
  cones.applyW(z, lambda_);
  cones.product(lambda_, lambda_, lambda_prod_);
  ds_comb_ = lambda_prod_ - tau * cones.identityElement();
  cones.inverseProduct(lambda_, ds_comb_, lambda_inv_ds_);
  cones.applyW(lambda_inv_ds_, ds_const_);
}

bool CoulombIpm::tangent(const CoulombProblem& pb, const Vec& z, Scalar c) {
  // dF/dmu = (0, -E c dl/deps, -e) for s o z = mu e: the complementarity part enters as ds_const
  // = -V (v \ e) (derivative of V (v \ (v o v - mu e))), the smoothing part as the primal residual.
  ConeSet& cones = pb.cones;
  cones.applyW(z, lambda_);
  cones.inverseProduct(lambda_, cones.identityElement(), lambda_inv_ds_);
  cones.applyW(lambda_inv_ds_, tmp2_m_);
  tmp2_m_ = -tmp2_m_;
  tmp_m_ = tmp2_m_;
  map_.applyBinv(tmp_m_, 1.0);
  rhs_.head(n_).setZero();
  rhs_.tail(m_) = tmp_m_;
  map_.subtractEpsDerivative(rhs_.tail(m_), -c);  // -dr_p/dmu = +E c dl/deps
  pb.kkt.solve(rhs_, sol_);
  tx_ = sol_.head(n_);
  tz_ = sol_.tail(m_);
  cones.mulHs(tz_, ts_);
  ts_ = -(ts_ + tmp2_m_);
  return tx_.allFinite() && ts_.allFinite() && tz_.allFinite();
}

void CoulombIpm::printPathMonitor(const CoulombProblem& pb, const Vec& s, const Vec& z, Scalar mu, Scalar c) {
  tangent(pb, z, c);
  Scalar ord_s = 0, ord_z = 0;
  Index worst_s = -1, worst_z = -1;
  for (Index i = 0; i < map_.numCones(); ++i) {
    const Index off = map_.offset(i), d = map_.dim(i);
    const Scalar os = mu * ts_.segment(off, d).norm() / s.segment(off, d).norm();
    const Scalar oz = mu * tz_.segment(off, d).norm() / z.segment(off, d).norm();
    if (os > ord_s) ord_s = os, worst_s = i;
    if (oz > ord_z) ord_z = oz, worst_z = i;
  }
  int n_half = 0;  // cones with an order in [0.3, 0.7] in s or z: degenerate (non-strict) pairs
  for (Index i = 0; i < map_.numCones(); ++i) {
    const Index off = map_.offset(i), d = map_.dim(i);
    const Scalar os = mu * ts_.segment(off, d).norm() / s.segment(off, d).norm();
    const Scalar oz = mu * tz_.segment(off, d).norm() / z.segment(off, d).norm();
    if ((os > 0.3 && os < 0.7) || (oz > 0.3 && oz < 0.7)) ++n_half;
  }
  const Index flips = std::abs(pb.kkt.ldlt().numPositivePivots() - n_);
  const auto spectral = [](const Vec& v, Index off, Index d, Scalar& l1, Scalar& l2) {
    const Scalar t = v.segment(off + 1, d - 1).norm();
    l1 = v[off] - t;
    l2 = v[off] + t;
  };
  if (worst_s >= 0) {
    const Index off = map_.offset(worst_s), d = map_.dim(worst_s);
    Scalar s1, s2, z1, z2;
    spectral(s, off, d, s1, s2);
    spectral(z, off, d, z1, z2);
    const Scalar cosang = s.segment(off + 1, d - 1).dot(z.segment(off + 1, d - 1)) /
                          (s.segment(off + 1, d - 1).norm() * z.segment(off + 1, d - 1).norm());
    std::printf("      cone %d: s spectral (%.2e, %.2e), z spectral (%.2e, %.2e), cos(s_bar, z_bar) %.3f\n",
                static_cast<int>(worst_s), s1, s2, z1, z2, cosang);
  }
  std::printf("      path: max order s %.2e (cone %d), z %.2e (cone %d), %d cones of order ~1/2, "
              "mu ||dx/dmu|| %.2e, pivots of unexpected sign %d\n",
              ord_s, static_cast<int>(worst_s), ord_z, static_cast<int>(worst_z), n_half,
              mu * tx_.lpNorm<Eigen::Infinity>(), static_cast<int>(flips));
}

Scalar CoulombIpm::maxStep(const CoulombProblem& pb, const Vec& s, const Vec& z, const Vec& ds,
                           const Vec& dz) const {
  return pb.cones.maxStep(s, ds, pb.cones.maxStep(z, dz, 1.0));
}

Scalar CoulombIpm::safeguardedStep(const CoulombProblem& pb, const Settings& settings, const Vec& s,
                                   const Vec& z, Scalar alpha_max) {
  // As SolverImpl::safeguardedStepLength(): interior and centrality >= centrality_theta * mu.
  const Scalar nu = static_cast<Scalar>(std::max(pb.cones.degree(), Index(1)));
  Scalar alpha = alpha_max;
  constexpr int kMaxBacktracks = 64;
  for (int attempt = 0; attempt < kMaxBacktracks; ++attempt, alpha *= settings.linesearch_backtrack) {
    s_trial_ = s + alpha * ds_;
    z_trial_ = z + alpha * dz_;
    if (!(pb.cones.margins(s_trial_).first > 0.0) || !(pb.cones.margins(z_trial_).first > 0.0)) continue;
    const Scalar mu_trial = s_trial_.dot(z_trial_) / nu;
    if (pb.cones.minCentrality(s_trial_, z_trial_) >= settings.centrality_theta * mu_trial) return alpha;
  }
  return alpha;
}

void CoulombIpm::pathWeights(const CoulombProblem& pb, const Vec& x, const Vec& s, const Vec& z, Scalar tau) {
  // Local relative metric: x relative to ||x||, every cone (orthant row) relative to its own norm,
  // mu relative to mu. A unit step changes the point by about 100 %, independent of the scales.
  constexpr Scalar kTiny = std::numeric_limits<Scalar>::min();
  wx_ = 1 / std::max(x.lpNorm<Eigen::Infinity>(), kTiny);
  wtau_ = 1 / std::max(tau, kTiny);
  ws_.setZero();
  const Scalar z0max = pb.zero_dim > 0 ? z.head(pb.zero_dim).cwiseAbs().maxCoeff() : Scalar(0);
  wz_.setConstant(1 / std::max(z0max, kTiny));
  const ConeSet& cones = pb.cones;
  for (Index bi = 0; bi < cones.numBlocks(); ++bi) {
    const Index off = cones.blockOffset(bi), d = cones.blockDim(bi);
    if (cones.blockType(bi) == ConeType::Nonnegative) {
      for (Index j = off; j < off + d; ++j) {
        ws_[j] = 1 / std::max(std::abs(s[j]), kTiny);
        wz_[j] = 1 / std::max(std::abs(z[j]), kTiny);
      }
    } else if (cones.blockType(bi) == ConeType::SecondOrder) {
      ws_.segment(off, d).setConstant(1 / std::max(s.segment(off, d).norm(), kTiny));
      wz_.segment(off, d).setConstant(1 / std::max(z.segment(off, d).norm(), kTiny));
    }
  }
}

Scalar CoulombIpm::pathDot(const Vec& ax, const Vec& as, const Vec& az, Scalar at, const Vec& bx, const Vec& bs,
                           const Vec& bz, Scalar bt) const {
  return wx_ * wx_ * ax.dot(bx) + (ws_.array().square() * as.array() * bs.array()).sum() +
         (wz_.array().square() * az.array() * bz.array()).sum() + wtau_ * wtau_ * at * bt;
}

CoulombIpm::PathResult CoulombIpm::continuePath(const CoulombProblem& pb, const Settings& settings, Vec& x,
                                                Vec& s, Vec& z, Scalar c, Scalar nu, Scalar mu_exit, int& steps) {
  // Pseudo-arclength continuation of F(y, mu) = (r_d, r_p with eps = c mu, s o z - mu e) = 0
  // (Allgower, Georg): unit tangent t = (dy/dmu, 1) / ||.|| in the local metric, predictor
  // (y, mu) + h t, corrector: Newton on F with the bordering t'W((y, mu) - predicted) = 0, i.e. the
  // two solves J a = -F(y, mu), J b = -dF/dmu with one factorization, dmu = (-g - t'W a) /
  // (t'W b + t_mu), dy = a + dmu b. Steps are damped to stay interior.
  constexpr int kMaxSteps = 60, kMaxCorrector = 6;
  constexpr Scalar kCorrTol = 1e-3, kHMin = 1e-6;
  const int verbose = settings.verbose;
  Scalar tau = s.dot(z) / nu;
  // Orientation: the last Mehrotra direction, which decreased mu.
  Scalar dtau_prev = -tau;
  pathWeights(pb, x, s, z, tau);
  ux_ = gdx_;
  us_ = gds_;
  uz_ = gdz_;
  Scalar ut = dtau_prev;
  Scalar h = 0.1;
  bool first = true;
  for (int k = 0; k < kMaxSteps; ++k) {
    // Tangent at the current point.
    pathWeights(pb, x, s, z, tau);
    if (!linearize(pb, x, s, z, c * tau, false) || !tangent(pb, z, c)) return PathResult::Failed;
    Scalar nt = std::sqrt(pathDot(tx_, ts_, tz_, 1, tx_, ts_, tz_, 1));
    Scalar sign = pathDot(tx_, ts_, tz_, 1, ux_, us_, uz_, ut) >= 0 ? Scalar(1) : Scalar(-1);
    ux_ = sign / nt * tx_;
    us_ = sign / nt * ts_;
    uz_ = sign / nt * tz_;
    ut = sign / nt;
    if (verbose >= 2) std::printf("   cont %2d: mu %.3e, t_mu (relative) %+.2e, h %.2e\n", k, tau, ut * tau * wtau_, h);

    bool accepted = false;
    int corr_iters = 0;
    while (!accepted && h >= kHMin) {
      // Predictor, kept interior.
      const Scalar amax = pb.cones.maxStep(s, us_, pb.cones.maxStep(z, uz_, std::numeric_limits<Scalar>::infinity()));
      Scalar hp = std::min(h, Scalar(0.9) * amax);
      if (tau + hp * ut <= 0) hp = Scalar(0.5) * tau / std::abs(ut);
      px_ = x + hp * ux_;
      ps_ = s + hp * us_;
      pz_ = z + hp * uz_;
      const Scalar ptau = tau + hp * ut;
      cx_ = px_;
      cs_ = ps_;
      cz_ = pz_;
      Scalar ctau = ptau;
      bool converged = false;
      for (corr_iters = 1; corr_iters <= kMaxCorrector; ++corr_iters) {
        if (!linearize(pb, cx_, cs_, cz_, c * ctau, false)) break;
        centeringConst(pb, cz_, ctau);
        if (!solveNewton(pb, ds_const_, ax_, az_, as_) || !tangent(pb, cz_, c)) break;
        // g = t'W((y, mu) - predicted); the bordering asks t'W(dy, dmu) = -g.
        px_ = cx_ - px_;
        ps_ = cs_ - ps_;
        pz_ = cz_ - pz_;
        const Scalar g = pathDot(ux_, us_, uz_, ut, px_, ps_, pz_, ctau - ptau);
        px_ = cx_ - px_;
        ps_ = cs_ - ps_;
        pz_ = cz_ - pz_;
        const Scalar den = pathDot(ux_, us_, uz_, 0, tx_, ts_, tz_, 0) + wtau_ * wtau_ * ut;
        const Scalar dtau = (-g - pathDot(ux_, us_, uz_, 0, ax_, as_, az_, 0)) / den;
        ax_ += dtau * tx_;
        as_ += dtau * ts_;
        az_ += dtau * tz_;
        Scalar alpha = std::min(Scalar(1), Scalar(0.95) * pb.cones.maxStep(cs_, as_, pb.cones.maxStep(cz_, az_, 2.0)));
        if (ctau + alpha * dtau <= 0) alpha = Scalar(0.5) * ctau / std::abs(dtau);
        if (!std::isfinite(alpha) || alpha <= 0) break;
        cx_ += alpha * ax_;
        cs_ += alpha * as_;
        cz_ += alpha * az_;
        ctau += alpha * dtau;
        const Scalar step = std::sqrt(pathDot(ax_, as_, az_, dtau, ax_, as_, az_, dtau));
        if (verbose >= 3) {
          std::printf("      corr %d: h %.2e, alpha %.2e, |dy,dmu| %.2e, dmu/mu %+.2e, mu %.3e, cent/mu %.2e, kkt %.1e\n",
                      corr_iters, hp, alpha, step, dtau / ctau, ctau, pb.cones.minCentrality(cs_, cz_) / ctau,
                      pb.kkt.lastRefinementResidual());
        }
        if (alpha == 1 && step < kCorrTol) {
          converged = true;
          break;
        }
      }
      if (converged && pb.cones.minCentrality(cs_, cz_) >= Scalar(0.1) * ctau) {
        accepted = true;
        x = cx_;
        s = cs_;
        z = cz_;
        dtau_prev = ctau - tau;
        tau = ctau;
      } else {
        h *= 0.5;
      }
    }
    if (!accepted) return PathResult::Failed;
    ++steps;
    if (corr_iters <= 2) h = std::min(Scalar(1.5) * h, Scalar(1));
    if (!first && tau < mu_exit && ut < 0) return PathResult::HandBack;
    first = false;
  }
  return PathResult::Failed;
}

CoulombIpm::Result CoulombIpm::solve(const CoulombProblem& pb, const Settings& settings, Vec& x, Vec& s,
                                     Vec& z, std::chrono::steady_clock::time_point start) {
  Result res;
  ConeSet& cones = pb.cones;
  const Scalar nu = static_cast<Scalar>(std::max(cones.degree(), Index(1)));
  const int verbose = settings.verbose;

  // Smoothing scale c = smoothing / z_ref, z_ref the median normal reaction at the start, fixed
  // for the solve so that the smoothed problems form one family in mu.
  std::vector<Scalar> zn;
  zn.reserve(static_cast<size_t>(map_.numCones()));
  Scalar zmax = 0;
  for (Index i = 0; i < map_.numCones(); ++i) {
    zn.push_back(z[map_.offset(i)]);
    zmax = std::max(zmax, z[map_.offset(i)]);
  }
  Scalar z_ref = 1;
  if (!zn.empty()) {
    std::nth_element(zn.begin(), zn.begin() + zn.size() / 2, zn.end());
    z_ref = std::max(zn[zn.size() / 2], Scalar(1e-8) * zmax);
  }
  const Scalar c = z_ref > 0 ? settings.coulomb.smoothing / z_ref : Scalar(0);
  res.smoothing = c;

  bool have_best = false;
  Metrics best;
  Scalar best_mu = 0;
  int tiny_steps = 0, continuations = 0;
  constexpr Scalar kContinuationAlpha = 0.1, kCenteredForPath = 0.1;
  have_centered_ = false;
  constexpr int kMaxContinuations = 5;
  Scalar alpha = 0, sigma = 0;
  res.status = Status::MaxIterations;
  if (verbose >= 2) {
    std::printf("iter    pres      dres      comp      mu        cent/mu   step    sigma   kkt\n");
  }

  for (int iter = 0; iter < settings.max_iter; ++iter) {
    res.iterations = iter + 1;
    const Metrics mt = metrics(pb, settings, x, s, z);
    const Scalar mu = s.dot(z) / nu;
    if (!have_best || mt.merit < best.merit) {
      best_x_ = x;
      best_s_ = s;
      best_z_ = z;
      best = mt;
      best_mu = mu;
      have_best = true;
    }
    if (verbose >= 2) {
      std::printf("%4d  %.2e  %.2e  %.2e  %.2e  %.2e  %.4f  %.4f  %.1e\n", iter, mt.res_primal, mt.res_dual,
                  mt.comp, mu, cones.minCentrality(s, z) / mu, alpha, sigma, pb.kkt.lastRefinementResidual());
    }
    if (mt.res_primal < settings.tol_feas && mt.res_dual < settings.tol_feas && mt.comp <= 1) {
      res.status = Status::Solved;
      break;
    }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > settings.time_limit) {
      res.status = Status::MaxTime;
      break;
    }

    // Smoothed De Saxce term, frozen for the iteration; rp_ and rd_ are those of metrics().
    if (!linearize(pb, x, s, z, c * mu, /*have_residuals=*/true)) {
      res.status = Status::NumericalError;
      break;
    }

    if (verbose >= 3) printPathMonitor(pb, s, z, mu, c);

    // Predictor (affine scaling direction): ds_const = s.
    if (!solveNewton(pb, s, dx_aff_, dz_aff_, ds_aff_)) {
      res.status = Status::NumericalError;
      break;
    }
    const Scalar alpha_aff = maxStep(pb, s, z, ds_aff_, dz_aff_);
    const Scalar mu_aff = (s + alpha_aff * ds_aff_).dot(z + alpha_aff * dz_aff_) / nu;
    const Scalar ratio = mu > 0 ? std::max(mu_aff / mu, Scalar(0)) : Scalar(1);
    sigma = std::min(ratio * ratio * ratio, Scalar(1));

    // Corrector: lambda o (W dz + W^-1 ds) = target e - lambda o lambda - (W^-1 ds_aff) o (W dz_aff).
    // The centering target is bounded below by a fraction of the complementarity tolerance: on the
    // central path ||s_i o z_i|| = mu (scaled units), and the measure of metrics() is met for
    // mu <= tol * comp_bound. Near the boundary the NT neighborhood allows ||s o z|| ~ sqrt(mu), so
    // once the bound is reached the steps are Newton steps onto the central point at that mu,
    // without the second-order term of the affine direction (which aims at mu = 0), instead of a
    // further decrease of mu (docs/design.md).
    const Scalar mu_floor = settings.coulomb.mu_floor * mt.comp_bound;
    const Scalar target = std::max(sigma * mu, mu_floor);
    cones.applyW(z, lambda_);
    cones.product(lambda_, lambda_, lambda_prod_);
    ds_comb_ = lambda_prod_ - target * cones.identityElement();
    if (sigma * mu >= mu_floor) {
      cones.applyWInv(ds_aff_, Winv_ds_aff_);
      cones.applyW(dz_aff_, W_dz_aff_);
      cones.product(Winv_ds_aff_, W_dz_aff_, corrector_);
      if (iter == 0) corrector_ *= alpha_aff;  // the start may be badly centred (as SolverImpl)
      ds_comb_ += corrector_;
    }
    cones.inverseProduct(lambda_, ds_comb_, lambda_inv_ds_);
    cones.applyW(lambda_inv_ds_, ds_const_);
    if (!solveNewton(pb, ds_const_, dx_, dz_, ds_)) {
      res.status = Status::NumericalError;
      break;
    }

    const Scalar alpha_max = maxStep(pb, s, z, ds_, dz_) * settings.max_step_fraction;
    if (!std::isfinite(alpha_max) || alpha_max <= 0.0) {
      res.status = Status::NumericalError;
      break;
    }
    alpha = safeguardedStep(pb, settings, s, z, alpha_max);
    if (cones.minCentrality(s, z) >= kCenteredForPath * mu) {
      gx_ = x;
      gs_ = s;
      gz_ = z;
      gdx_ = dx_;
      gds_ = ds_;
      gdz_ = dz_;
      have_centered_ = true;
    }
    x += alpha * dx_;
    s += alpha * ds_;
    z += alpha * dz_;
    // Two blocked steps: mu cannot be decreased along the path here (a fold of the central path
    // in mu at a stick-slip transition, docs/design.md); follow the path by pseudo-arclength
    // continuation until mu is below a tenth of its value, then continue with Mehrotra steps.
    if (alpha < kContinuationAlpha) {
      if (++tiny_steps >= 2) {
        tiny_steps = 0;
        if (++continuations > kMaxContinuations) {
          res.status = Status::InsufficientProgress;
          break;
        }
        const Scalar mu_now = s.dot(z) / nu;
        if (have_centered_) {
          x = gx_;
          s = gs_;
          z = gz_;
          have_centered_ = false;
        }
        if (continuePath(pb, settings, x, s, z, c, nu, Scalar(0.1) * mu_now, res.continuation_steps) ==
            PathResult::Failed) {
          res.status = Status::InsufficientProgress;
          break;
        }
      }
    } else {
      tiny_steps = 0;
    }
  }

  if (res.status == Status::Solved) {
    res.metrics = metrics(pb, settings, x, s, z);
    res.mu = s.dot(z) / nu;
  } else if (have_best) {
    x = best_x_;
    s = best_s_;
    z = best_z_;
    res.metrics = best;
    res.mu = best_mu;
    const bool comp_ok = best.comp * settings.tol_gap_abs <= settings.reduced_tol_gap_abs;
    if ((res.status == Status::MaxIterations || res.status == Status::MaxTime ||
         res.status == Status::InsufficientProgress) &&
        best.res_primal < settings.reduced_tol_feas && best.res_dual < settings.reduced_tol_feas && comp_ok) {
      res.status = Status::AlmostSolved;
    }
  }
  return res;
}

std::vector<ContactState> classifyContacts(const CoulombMap& map, const Vec& s, const Vec& z, Scalar threshold) {
  const Index nc = map.numCones();
  std::vector<ContactState> states(static_cast<size_t>(nc), ContactState::Undecided);
  const Scalar t = threshold;
  for (Index i = 0; i < nc; ++i) {
    const Index off = map.offset(i), d = map.dim(i);
    const Scalar sn = tailNorm(s.data() + off, d), zn = tailNorm(z.data() + off, d);
    const Scalar s1 = s[off] - sn, s2 = s[off] + sn, z1 = z[off] - zn, z2 = z[off] + zn;
    ContactState& st = states[static_cast<size_t>(i)];
    if (z2 <= t && s1 > t) {
      st = ContactState::TakeOff;
    } else if (z1 > t && s2 <= t) {
      st = ContactState::Stick;
    } else if (z1 <= t && t < z2 && s1 <= t && t < s2) {
      st = ContactState::Slip;
    }
  }
  return states;
}

}  // namespace conicxx::detail
