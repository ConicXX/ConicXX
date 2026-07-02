#pragma once

#include <memory>

#include "conicxx/cone_spec.h"
#include "conicxx/cones/cone_set.h"
#include "conicxx/kkt/equilibration.h"
#include "conicxx/kkt/kkt_system.h"
#include "conicxx/settings.h"
#include "conicxx/solution.h"
#include "conicxx/types.h"

namespace conicxx::detail {

/// Implements the homogeneous self-dual embedding interior-point method
/// described in the design notes: residuals/search-direction formulas
/// follow Domahidi/Chu/Boyd's ECOS paper (Nesterov-Todd scaling, Mehrotra
/// predictor-corrector, dynamic regularization + iterative refinement),
/// extended with the quadratic-objective (x'Px/tau) terms exactly as
/// implemented in Clarabel.rs's residuals.rs/kktsystem.rs (verified against
/// that source directly).
class SolverImpl {
 public:
  bool setup(const SparseMat& P, const Vec& q, const SparseMat& A, const Vec& b,
             const ConeSpec& cone_spec, const Settings& settings);
  bool updateData(const SparseMat* P, const Vec* q, const SparseMat* A, const Vec* b);
  void setWarmStart(const Vec& x, const Vec& s, const Vec& z);

  void setSettings(const Settings& settings) { settings_ = settings; }
  const Settings& settings() const { return settings_; }

  const Solution& solve();
  const Solution& solution() const { return solution_; }

 private:
  // --- initialization ---
  bool computeInitialPoint();
  void shiftToInteriorCold(Eigen::Ref<Vec> v) const;
  void ensureStrictlyInteriorWarm(Eigen::Ref<Vec> v) const;

  // --- per-iteration pipeline ---
  bool refactorizeForCurrentScaling();
  bool computeConstantSolve();  // x1_, z1_, Px1_
  void computeResiduals();
  Scalar computeMu() const;
  bool computeAffineStep();
  bool computeCombinedStep(Scalar sigma, Scalar mu);
  Scalar computeStepLength(const Vec& ds, const Vec& dz, Scalar dtau, Scalar dkappa) const;
  void addStep(Scalar alpha);
  void maybeRescale();

  Vec Pmul(const Vec& v) const;  // P (symmetric, upper-stored) * v

  bool checkConvergence(Scalar norm_rx, Scalar norm_rz, Scalar primal_obj, Scalar dual_obj) const;
  bool checkInfeasibility();

  void finalizeSolution(bool converged);

  // --- problem data (equilibrated in place if enabled) ---
  SparseMat P_, A_;
  Vec q_, b_;
  Index n_ = 0, m_ = 0;
  ConeSpec cone_spec_;
  std::unique_ptr<ConeSet> cones_;
  Settings settings_;
  Equilibration equil_;
  KktSystem kkt_;
  bool setup_done_ = false;

  // --- iterate ---
  Vec x_, s_, z_;
  Scalar tau_ = 1.0, kappa_ = 1.0;

  // --- residual cache ---
  Vec Px_, rx_, rz_;
  Scalar rtau_ = 0, dot_qx_ = 0, dot_bz_ = 0, dot_sz_ = 0, dot_xPx_ = 0;

  // --- per-iteration constant KKT solve cache ---
  Vec x1_, z1_, Px1_;

  // --- step direction storage ---
  Vec dx_aff_, dz_aff_, ds_aff_;
  Scalar dtau_aff_ = 0, dkappa_aff_ = 0;
  Vec dx_, dz_, ds_;
  Scalar dtau_ = 0, dkappa_ = 0;

  // --- warm start ---
  bool have_warm_start_ = false;
  Vec warm_x_, warm_s_, warm_z_;

  Solution solution_;
};

}  // namespace conicxx::detail
