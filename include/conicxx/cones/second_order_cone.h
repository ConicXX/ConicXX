#pragma once

#include "conicxx/cones/cone_base.h"

namespace conicxx {

/// The second-order (Lorentz) cone Q^dim = { (x0,x1) : x0 >= ||x1||_2 }, dim >= 2.
///
/// The Jordan product/inverse-product follow the standard Euclidean Jordan algebra for the
/// Lorentz cone. The Nesterov-Todd scaling is stored in the closed-form (eta, w) arrowhead
/// representation (Domahidi/Chu/Boyd, ECOS paper eq. 7), not as a dense dim x dim matrix:
///
///   a = sqrt(s'Js), b = sqrt(z'Jz), eta = sqrt(a/b),  J = diag(1,-1,...,-1)
///   sbar = s/a, zbar = z/b, gamma = sqrt((1+sbar'zbar)/2), w = (sbar + J zbar)/(2 gamma)
///   W = eta * [ w0  w1' ; w1  I + w1 w1'/(1+w0) ]   (=> w'Jw = 1 is what makes this symmetric
///                                                       and W^2 z == s hold)
///   W^-1 = J W J / eta^2,  H = W^2 = eta^2 (2 w w' - J)
///
/// applyW/applyWInv are O(dim) rank-1-update formulas derived directly from the block form above
/// -- no matrix is ever built or factored to apply W or its inverse. mulHs()/writeHsLowerTriangle()
/// (needed for the KKT (2,2) block, see cone_base.h) still materialize H's O(dim^2) entries, but
/// only when actually requested, and dim is always small for the intended friction-cone
/// application, so this is cheap.
///
/// A note on formula provenance: an earlier attempt at this same arrowhead formula (using a
/// different, non-matching normalization of w) reproduced W^2 z == s only to ~1% accuracy and was
/// replaced with a "hyperbolic Householder" formulation (Andersen/Dahl/Vandenberghe) instead. The
/// two are algebraically the same W; the formula above (with w defined exactly as in the
/// derivation, not further renormalized) has been checked numerically against that
/// previously-verified Householder form to ~1e-15 agreement, and W^2 z == s / W^-1 == JWJ/eta^2 /
/// W^2 == eta^2(2ww'-J) all confirmed directly, so it does not repeat that earlier mistake.
///
/// No clamps, floors, or identity fallback here (contrast with earlier versions of this class):
/// updateScaling() requires s, z strictly interior, the same as the formulas above require to
/// stay well-defined (a, b > 0). Guaranteeing that is the caller's job -- the IPM step-length
/// safeguard (interior check + centrality check + backtracking, see SolverImpl and
/// Settings::centrality_theta/min_terminate_step_length) now owns that guarantee, instead of this
/// class silently absorbing a bad point by breaking the NT identity.
class SecondOrderCone final : public ConeBase {
 public:
  explicit SecondOrderCone(Index dim);

  ConeType type() const override { return ConeType::SecondOrder; }
  Index degree() const override { return 1; }

  const Vec& identityElement() const override { return e_; }

  void product(const Eigen::Ref<const Vec>& u, const Eigen::Ref<const Vec>& v,
               Eigen::Ref<Vec> out) const override;
  void inverseProduct(const Eigen::Ref<const Vec>& u, const Eigen::Ref<const Vec>& w,
                       Eigen::Ref<Vec> out) const override;

  void updateScaling(const Eigen::Ref<const Vec>& s, const Eigen::Ref<const Vec>& z) override;

  void applyW(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const override;
  void applyWInv(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const override;

  void mulHs(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const override;
  Index numHsEntries() const override { return dim_ * (dim_ + 1) / 2; }
  void writeHsLowerTriangle(Eigen::Ref<Vec> out) const override;

  Scalar margin(const Eigen::Ref<const Vec>& x) const override;

  Scalar minSquaredEigenvalue(const Eigen::Ref<const Vec>& lambda) const override;

  void scaledUnitShift(Eigen::Ref<Vec> x, Scalar alpha) const override { x[0] += alpha; }

  Scalar maxStep(const Eigen::Ref<const Vec>& x, const Eigen::Ref<const Vec>& dx,
                 Scalar alpha_max) const override;

 private:
  Vec e_;  // (1, 0, ..., 0)

  // NT scaling, closed-form arrowhead representation -- see class comment.
  Scalar eta_ = 1.0;
  Vec w_;  // w[0] = w0, w.tail(dim-1) = w1; w'Jw = 1

  mutable Vec scratch_;  // dim_-sized scratch for mulHs()'s two applyW() calls
};

}  // namespace conicxx
