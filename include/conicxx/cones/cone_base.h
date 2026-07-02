#pragma once

#include <Eigen/Core>

#include "conicxx/types.h"

namespace conicxx {

enum class ConeType { Zero, Nonnegative, SecondOrder };

/// Interface implemented by each cone block (Zero, Nonnegative, SecondOrder).
///
/// All cones here are symmetric cones (or, for the Zero cone, the degenerate
/// dual-free case), so a single Nesterov-Todd-scaling-based interface
/// suffices -- there is no need for the more general non-symmetric-cone
/// barrier machinery that full Clarabel uses for exponential/power cones.
///
/// Convention: the Zero cone is modeled as a genuine cone whose identity
/// element, Jordan product, inverse product and scaling block are all
/// identically zero. This makes it participate safely in every generic
/// formula used by the IPM loop (affine/combined ds, Delta s, step length,
/// interior-shift) without any special-casing in the solver: the s-block for
/// a Zero cone is invariant at 0 once initialized there, and z is left
/// completely unconstrained (free dual variable), exactly matching the fact
/// that the dual cone of {0} is R^k.
class ConeBase {
 public:
  explicit ConeBase(Index dim) : dim_(dim) {}
  virtual ~ConeBase() = default;

  Index dim() const { return dim_; }
  virtual ConeType type() const = 0;

  /// Degree (rank) of the symmetric cone as used in mu = (s'z+tau*kappa)/(sum(degree)+1):
  /// 0 for Zero, dim for Nonnegative, 1 per block for SecondOrder (regardless
  /// of the block's dimension -- a standard fact about the Lorentz cone,
  /// verified directly from s'z = lambda'lambda = mu * e'e = mu at the
  /// analytic center e = (1,0,...,0), independent of the ambient dimension).
  virtual Index degree() const = 0;

  virtual const Vec& identityElement() const = 0;

  /// Jordan product u o v.
  virtual void product(const Eigen::Ref<const Vec>& u, const Eigen::Ref<const Vec>& v,
                        Eigen::Ref<Vec> out) const = 0;

  /// Inverse Jordan product: solves u o out = w for out.
  virtual void inverseProduct(const Eigen::Ref<const Vec>& u, const Eigen::Ref<const Vec>& w,
                               Eigen::Ref<Vec> out) const = 0;

  /// Compute and cache the Nesterov-Todd scaling point/matrix from the
  /// current (s, z) pair (both assumed strictly in the cone interior).
  virtual void updateScaling(const Eigen::Ref<const Vec>& s, const Eigen::Ref<const Vec>& z) = 0;

  virtual void applyW(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const = 0;
  virtual void applyWInv(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const = 0;

  /// Dense Hs = W^T W block contributed to the (2,2) KKT block, dim() x dim().
  virtual const Mat& scalingBlock() const = 0;

  /// Distance-to-boundary style measure of x within this cone; used to build
  /// a strictly-interior starting point. +infinity for the Zero cone (always
  /// "interior", never binding in an aggregate min-margin computation).
  virtual Scalar margin(const Eigen::Ref<const Vec>& x) const = 0;

  /// x += alpha * e (the cone's identity element). A no-op for the Zero cone.
  virtual void scaledUnitShift(Eigen::Ref<Vec> x, Scalar alpha) const = 0;

  /// Largest step length in [0, alpha_max] such that x + alpha*dx remains in
  /// the (closed) cone. Returns alpha_max unchanged if unconstrained.
  virtual Scalar maxStep(const Eigen::Ref<const Vec>& x, const Eigen::Ref<const Vec>& dx,
                          Scalar alpha_max) const = 0;

 protected:
  Index dim_;
};

}  // namespace conicxx
