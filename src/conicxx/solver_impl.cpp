#include "conicxx/solver_impl.h"

#include <algorithm>
#include <cmath>

namespace conicxx::detail {

bool SolverImpl::setup(const SparseMat& P, const Vec& q, const SparseMat& A, const Vec& b,
                       const ConeSpec& cone_spec, const Settings& settings) {
  setup_done_ = false;
  if (!cone_spec.isValid()) return false;
  if (P.rows() != P.cols()) return false;

  n_ = static_cast<Index>(P.rows());
  m_ = cone_spec.totalDim();
  if (A.rows() != m_ || A.cols() != n_ || b.size() != m_ || q.size() != n_) return false;

  cone_spec_ = cone_spec;
  settings_ = settings;
  cones_ = std::make_unique<ConeSet>(cone_spec_);

  P_ = P;
  q_ = q;
  A_ = A;
  b_ = b;

  equil_ = Equilibration();
  if (settings_.equilibrate) {
    equil_.compute(P_, q_, A_, b_, *cones_, settings_);
    equil_.scaleProblem(P_, q_, A_, b_);
  }

  if (!kkt_.setup(P_, A_, *cones_, settings_)) return false;

  x_ = Vec::Zero(n_);
  s_ = Vec::Zero(m_);
  z_ = Vec::Zero(m_);
  Px_ = Vec::Zero(n_);
  rx_ = Vec::Zero(n_);
  rz_ = Vec::Zero(m_);
  x1_ = Vec::Zero(n_);
  z1_ = Vec::Zero(m_);
  Px1_ = Vec::Zero(n_);
  dx_aff_ = Vec::Zero(n_);
  dz_aff_ = Vec::Zero(m_);
  ds_aff_ = Vec::Zero(m_);
  dx_ = Vec::Zero(n_);
  dz_ = Vec::Zero(m_);
  ds_ = Vec::Zero(m_);

  have_warm_start_ = false;
  solution_ = Solution{};
  setup_done_ = true;
  return true;
}

bool SolverImpl::updateData(const SparseMat* P, const Vec* q, const SparseMat* A, const Vec* b) {
  if (!setup_done_) return false;

  SparseMat P_scaled, A_scaled;
  Vec q_scaled, b_scaled;

  if (P) {
    if (P->rows() != n_ || P->cols() != n_) return false;
    P_scaled = *P;
    if (settings_.equilibrate) equil_.scaleP(P_scaled);
  }
  if (A) {
    if (A->rows() != m_ || A->cols() != n_) return false;
    A_scaled = *A;
    if (settings_.equilibrate) equil_.scaleA(A_scaled);
  }
  if (q) {
    if (q->size() != n_) return false;
    q_scaled = *q;
    if (settings_.equilibrate) equil_.scaleQ(q_scaled);
  }
  if (b) {
    if (b->size() != m_) return false;
    b_scaled = *b;
    if (settings_.equilibrate) equil_.scaleB(b_scaled);
  }

  if (!kkt_.updateData(P ? &P_scaled : nullptr, A ? &A_scaled : nullptr)) return false;

  if (P) P_ = P_scaled;
  if (A) A_ = A_scaled;
  if (q) q_ = q_scaled;
  if (b) b_ = b_scaled;
  return true;
}

void SolverImpl::setWarmStart(const Vec& x, const Vec& s, const Vec& z) {
  if (!setup_done_) return;
  warm_x_ = x;
  warm_s_ = s;
  warm_z_ = z;
  if (settings_.equilibrate) equil_.scaleSolution(warm_x_, warm_s_, warm_z_);
  have_warm_start_ = true;
}

Vec SolverImpl::Pmul(const Vec& v) const { return P_.selfadjointView<Eigen::Upper>() * v; }

void SolverImpl::shiftToInteriorCold(Eigen::Ref<Vec> v) const {
  const auto [min_margin, pos_margin] = cones_->margins(v);
  const Scalar deg = static_cast<Scalar>(std::max(cones_->degree(), Index(1)));
  const Scalar target = std::max(Scalar(1.0), Scalar(0.1) * pos_margin / deg);
  if (min_margin <= 0.0) {
    cones_->scaledUnitShift(v, -min_margin);
    cones_->scaledUnitShift(v, target);
  } else if (min_margin < target) {
    cones_->scaledUnitShift(v, target - min_margin);
  } else {
    cones_->scaledUnitShift(v, 0.0);
  }
}

void SolverImpl::ensureStrictlyInteriorWarm(Eigen::Ref<Vec> v) const {
  const auto [min_margin, pos_margin] = cones_->margins(v);
  (void)pos_margin;
  if (min_margin <= 1e-8) {
    cones_->scaledUnitShift(v, -min_margin + 1e-6);
  }
}

bool SolverImpl::computeInitialPoint() {
  cones_->updateScaling(cones_->identityElement(), cones_->identityElement());
  if (!kkt_.updateScalingAndFactorize(*cones_)) return false;

  Vec rhs(n_ + m_), sol;
  if (P_.nonZeros() == 0) {
    rhs.head(n_).setZero();
    rhs.tail(m_) = b_;
    kkt_.solve(rhs, sol);
    x_ = sol.head(n_);
    s_ = -sol.tail(m_);

    rhs.head(n_) = -q_;
    rhs.tail(m_).setZero();
    kkt_.solve(rhs, sol);
    z_ = sol.tail(m_);
  } else {
    rhs.head(n_) = -q_;
    rhs.tail(m_) = b_;
    kkt_.solve(rhs, sol);
    x_ = sol.head(n_);
    z_ = sol.tail(m_);
    s_ = -z_;
  }

  cones_->zeroPrimalZeroConeBlocks(s_);
  shiftToInteriorCold(s_);
  shiftToInteriorCold(z_);

  tau_ = 1.0;
  kappa_ = 1.0;
  return x_.allFinite() && s_.allFinite() && z_.allFinite();
}

bool SolverImpl::refactorizeForCurrentScaling() {
  cones_->updateScaling(s_, z_);
  return kkt_.updateScalingAndFactorize(*cones_);
}

bool SolverImpl::computeConstantSolve() {
  Vec rhs(n_ + m_);
  rhs.head(n_) = -q_;
  rhs.tail(m_) = b_;
  Vec sol;
  kkt_.solve(rhs, sol);
  x1_ = sol.head(n_);
  z1_ = sol.tail(m_);
  Px1_ = Pmul(x1_);
  return x1_.allFinite() && z1_.allFinite();
}

void SolverImpl::computeResiduals() {
  Px_ = Pmul(x_);
  rx_ = -(A_.transpose() * z_) - Px_ - tau_ * q_;
  rz_ = A_ * x_ + s_ - tau_ * b_;
  dot_qx_ = q_.dot(x_);
  dot_bz_ = b_.dot(z_);
  dot_sz_ = s_.dot(z_);
  dot_xPx_ = x_.dot(Px_);
  rtau_ = dot_qx_ + dot_bz_ + kappa_ + dot_xPx_ / tau_;
}

Scalar SolverImpl::computeMu() const {
  return (dot_sz_ + tau_ * kappa_) / static_cast<Scalar>(cones_->degree() + 1);
}

bool SolverImpl::computeAffineStep() {
  // Two solves feed the tau/kappa elimination: the "variable" solve (xv,zv,
  // fresh RHS below, recomputed for every affine/combined call) and the
  // "constant" solve (x1_,z1_, RHS=(-q,b), cached once per iteration by
  // computeConstantSolve()). tau_num uses the VARIABLE solve, tau_den uses
  // the CONSTANT one -- this maps onto Clarabel's kktsystem.rs exactly,
  // where (confusingly) `self.x1/z1` denotes the per-call variable solve
  // and `self.x2/z2` denotes the constant one.
  Vec rhs(n_ + m_);
  rhs.head(n_) = rx_;
  rhs.tail(m_) = s_ - rz_;
  Vec sol;
  kkt_.solve(rhs, sol);
  Vec xv = sol.head(n_), zv = sol.tail(m_);
  if (!xv.allFinite() || !zv.allFinite()) return false;

  const Scalar dtau_rhs = rtau_;
  const Scalar dkappa_rhs = tau_ * kappa_;
  const Vec xi = x_ / tau_;

  const Scalar tau_num =
      dtau_rhs - dkappa_rhs / tau_ + q_.dot(xv) + b_.dot(zv) + 2.0 * xi.dot(Pmul(xv));
  const Vec xi_minus_x1 = xi - x1_;
  const Scalar tau_den = kappa_ / tau_ - q_.dot(x1_) - b_.dot(z1_) +
                          xi_minus_x1.dot(Pmul(xi_minus_x1)) - x1_.dot(Px1_);
  if (tau_den == 0.0 || !std::isfinite(tau_num) || !std::isfinite(tau_den)) return false;

  dtau_aff_ = tau_num / tau_den;
  dx_aff_ = xv + dtau_aff_ * x1_;
  dz_aff_ = zv + dtau_aff_ * z1_;

  Vec Hz(m_);
  cones_->mulHs(dz_aff_, Hz);
  ds_aff_ = -(Hz + s_);

  dkappa_aff_ = -(dkappa_rhs + kappa_ * dtau_aff_) / tau_;

  return std::isfinite(dtau_aff_) && std::isfinite(dkappa_aff_) && dx_aff_.allFinite() &&
         dz_aff_.allFinite() && ds_aff_.allFinite();
}

bool SolverImpl::computeCombinedStep(Scalar sigma, Scalar mu) {
  const Vec dz_table = (1.0 - sigma) * rz_;

  Vec lambda(m_);
  cones_->applyW(z_, lambda);
  Vec lambda_prod(m_);
  cones_->product(lambda, lambda, lambda_prod);

  Vec Winv_ds_aff(m_);
  cones_->applyWInv(ds_aff_, Winv_ds_aff);
  Vec W_dz_aff(m_);
  cones_->applyW(dz_aff_, W_dz_aff);
  Vec corrector(m_);
  cones_->product(Winv_ds_aff, W_dz_aff, corrector);

  const Vec ds_combined = lambda_prod + corrector - sigma * mu * cones_->identityElement();

  Vec lambda_inv_ds(m_);
  cones_->inverseProduct(lambda, ds_combined, lambda_inv_ds);
  Vec ds_const(m_);
  cones_->applyW(lambda_inv_ds, ds_const);

  Vec rhs(n_ + m_);
  rhs.head(n_) = (1.0 - sigma) * rx_;
  rhs.tail(m_) = ds_const - dz_table;
  Vec sol;
  kkt_.solve(rhs, sol);
  Vec xv = sol.head(n_), zv = sol.tail(m_);
  if (!xv.allFinite() || !zv.allFinite()) return false;

  const Scalar dtau_rhs = (1.0 - sigma) * rtau_;
  const Scalar dkappa_rhs = tau_ * kappa_ + dtau_aff_ * dkappa_aff_ - sigma * mu;

  const Vec xi = x_ / tau_;
  const Scalar tau_num =
      dtau_rhs - dkappa_rhs / tau_ + q_.dot(xv) + b_.dot(zv) + 2.0 * xi.dot(Pmul(xv));
  const Vec xi_minus_x1 = xi - x1_;
  const Scalar tau_den = kappa_ / tau_ - q_.dot(x1_) - b_.dot(z1_) +
                          xi_minus_x1.dot(Pmul(xi_minus_x1)) - x1_.dot(Px1_);
  if (tau_den == 0.0 || !std::isfinite(tau_num) || !std::isfinite(tau_den)) return false;

  dtau_ = tau_num / tau_den;
  dx_ = xv + dtau_ * x1_;
  dz_ = zv + dtau_ * z1_;

  Vec Hz(m_);
  cones_->mulHs(dz_, Hz);
  ds_ = -(Hz + ds_const);

  dkappa_ = -(dkappa_rhs + kappa_ * dtau_) / tau_;

  return std::isfinite(dtau_) && std::isfinite(dkappa_) && dx_.allFinite() && dz_.allFinite() &&
         ds_.allFinite();
}

Scalar SolverImpl::computeStepLength(const Vec& ds, const Vec& dz, Scalar dtau,
                                     Scalar dkappa) const {
  Scalar alpha = 1.0;
  if (dtau < 0) alpha = std::min(alpha, -tau_ / dtau);
  if (dkappa < 0) alpha = std::min(alpha, -kappa_ / dkappa);
  alpha = cones_->maxStep(z_, dz, alpha);
  alpha = cones_->maxStep(s_, ds, alpha);
  return alpha;
}

void SolverImpl::addStep(Scalar alpha) {
  x_ += alpha * dx_;
  s_ += alpha * ds_;
  z_ += alpha * dz_;
  tau_ += alpha * dtau_;
  kappa_ += alpha * dkappa_;
}

void SolverImpl::maybeRescale() {
  const Scalar scale = std::max(tau_, kappa_);
  if (scale > 1e8 || scale < 1e-8) {
    const Scalar inv = 1.0 / scale;
    x_ *= inv;
    s_ *= inv;
    z_ *= inv;
    tau_ *= inv;
    kappa_ *= inv;
  }
}

bool SolverImpl::checkConvergence(Scalar norm_rx, Scalar norm_rz, Scalar primal_obj,
                                  Scalar dual_obj) const {
  const Scalar gap_abs = std::abs(primal_obj - dual_obj);
  const Scalar gap_tol = settings_.tol_gap_abs +
                          settings_.tol_gap_rel * std::max(std::abs(primal_obj), std::abs(dual_obj));
  return norm_rx < settings_.tol_feas && norm_rz < settings_.tol_feas && gap_abs < gap_tol &&
         tau_ > 1e-6;
}

bool SolverImpl::checkInfeasibility() {
  const Scalar ratio = tau_ / std::max(kappa_, Scalar(1e-30));
  if (ratio > settings_.tol_infeas) return false;

  if (dot_bz_ < -settings_.tol_infeas * std::max(Scalar(1.0), z_.norm())) {
    solution_.status = Status::PrimalInfeasible;
    return true;
  }
  if (dot_qx_ < -settings_.tol_infeas * std::max(Scalar(1.0), x_.norm())) {
    solution_.status = Status::DualInfeasible;
    return true;
  }
  return false;
}

void SolverImpl::finalizeSolution(bool /*converged*/) {
  const bool infeasible =
      (solution_.status == Status::PrimalInfeasible || solution_.status == Status::DualInfeasible);
  const Scalar scale =
      infeasible ? Scalar(1.0) / std::max(kappa_, Scalar(1e-30)) : Scalar(1.0) / std::max(tau_, Scalar(1e-30));

  Vec x_out = x_ * scale, s_out = s_ * scale, z_out = z_ * scale;
  if (settings_.equilibrate) equil_.unscaleSolution(x_out, s_out, z_out);

  solution_.x = x_out;
  solution_.s = s_out;
  solution_.z = z_out;

  Scalar primal_obj = (0.5 * dot_xPx_ / tau_ + dot_qx_) / tau_;
  if (settings_.equilibrate) primal_obj = equil_.unscaleObjective(primal_obj);
  solution_.objective = primal_obj;

  Scalar gap = dot_sz_ / (tau_ * tau_);
  if (settings_.equilibrate) gap = equil_.unscaleObjective(gap);
  solution_.info.duality_gap = gap;
  solution_.info.primal_residual = rz_.norm() / std::max(tau_, Scalar(1e-30));
  solution_.info.dual_residual = rx_.norm() / std::max(tau_, Scalar(1e-30));
  solution_.info.mu = computeMu();
}

const Solution& SolverImpl::solve() {
  solution_ = Solution{};
  if (!setup_done_) {
    solution_.status = Status::NumericalError;
    return solution_;
  }

  const bool use_warm = settings_.warm_start && have_warm_start_;
  if (use_warm) {
    x_ = warm_x_;
    s_ = warm_s_;
    z_ = warm_z_;
    tau_ = 1.0;
    kappa_ = 1.0;
    cones_->zeroPrimalZeroConeBlocks(s_);
    ensureStrictlyInteriorWarm(s_);
    ensureStrictlyInteriorWarm(z_);
  } else if (!computeInitialPoint()) {
    solution_.status = Status::NumericalError;
    finalizeSolution(false);
    return solution_;
  }

  Status final_status = Status::MaxIterations;
  int iterations = 0;

  for (int iter = 0; iter < settings_.max_iter; ++iter) {
    iterations = iter + 1;

    if (!refactorizeForCurrentScaling() || !computeConstantSolve()) {
      final_status = Status::NumericalError;
      break;
    }
    computeResiduals();

    const Scalar mu = computeMu();
    const Scalar norm_rx = rx_.norm() / tau_;
    const Scalar norm_rz = rz_.norm() / tau_;
    const Scalar primal_obj = (0.5 * dot_xPx_ / tau_ + dot_qx_) / tau_;
    const Scalar dual_obj = (-0.5 * dot_xPx_ / tau_ - dot_bz_) / tau_;

    if (checkConvergence(norm_rx, norm_rz, primal_obj, dual_obj)) {
      final_status = Status::Solved;
      break;
    }
    if (checkInfeasibility()) {
      final_status = solution_.status;
      break;
    }

    if (!computeAffineStep()) {
      final_status = Status::NumericalError;
      break;
    }

    const Scalar alpha_aff = computeStepLength(ds_aff_, dz_aff_, dtau_aff_, dkappa_aff_);
    const Vec s_aff = s_ + alpha_aff * ds_aff_;
    const Vec z_aff = z_ + alpha_aff * dz_aff_;
    const Scalar tau_aff = tau_ + alpha_aff * dtau_aff_;
    const Scalar kappa_aff = kappa_ + alpha_aff * dkappa_aff_;
    const Scalar mu_aff =
        (s_aff.dot(z_aff) + tau_aff * kappa_aff) / static_cast<Scalar>(cones_->degree() + 1);

    Scalar sigma;
    if (mu > 0) {
      const Scalar ratio = std::max(mu_aff / mu, Scalar(0.0));
      sigma = std::min(std::max(ratio * ratio * ratio, Scalar(0.0)), Scalar(1.0));
    } else {
      sigma = 1.0;
    }

    if (!computeCombinedStep(sigma, mu)) {
      final_status = Status::NumericalError;
      break;
    }

    const Scalar alpha =
        computeStepLength(ds_, dz_, dtau_, dkappa_) * settings_.max_step_fraction;
    if (!std::isfinite(alpha) || alpha <= 0.0) {
      final_status = Status::NumericalError;
      break;
    }

    addStep(alpha);
    maybeRescale();
  }

  solution_.status = final_status;
  solution_.info.iterations = iterations;
  finalizeSolution(final_status == Status::Solved);

  if (final_status == Status::Solved && settings_.warm_start) {
    warm_x_ = x_;
    warm_s_ = s_;
    warm_z_ = z_;
    have_warm_start_ = true;
  }

  return solution_;
}

}  // namespace conicxx::detail
