#pragma once

#include <Eigen/LU>

#include "conicxx/cones/cone_base.h"

namespace conicxx {

/// The second-order (Lorentz) cone Q^dim = { (x0,x1) : x0 >= ||x1||_2 }, dim >= 2.
///
/// The Jordan product/inverse-product follow the standard Euclidean Jordan
/// algebra for the Lorentz cone. The Nesterov-Todd scaling matrix W is built
/// via the "hyperbolic Householder" construction W = beta*(2*v*v' - J),
/// J = diag(1,-1,...,-1), following Andersen/Dahl/Vandenberghe (CVXOPT's
/// `misc.compute_scaling`), which is verified (both analytically and by a
/// direct numerical check against the defining property W^2 z == s) to be
/// exact -- an earlier attempt based on a more literal reading of the
/// arrowhead formula in Domahidi/Chu/Boyd's ECOS paper (eq. 7) was subtly
/// off by a missing renormalization step and reproduced W^2 z == s only to
/// ~1% accuracy, which the unit tests below caught.
///
/// W is cached as a dense dim x dim matrix (rather than using the O(dim)
/// rank-1-update fast-apply formulas that large-scale SOCP solvers rely on):
/// the target application (multibody contact friction cones) uses tiny
/// blocks (typically dim 3), so dense construction/solve is simpler and no
/// less robust, at negligible cost.
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

  const Mat& scalingBlock() const override { return Hs_; }

  Scalar margin(const Eigen::Ref<const Vec>& x) const override;

  void scaledUnitShift(Eigen::Ref<Vec> x, Scalar alpha) const override { x[0] += alpha; }

  Scalar maxStep(const Eigen::Ref<const Vec>& x, const Eigen::Ref<const Vec>& dx,
                 Scalar alpha_max) const override;

 private:
  Vec e_;   // (1, 0, ..., 0)
  Mat W_;   // dense NT scaling matrix, dim x dim (symmetric, invertible)
  Mat Hs_;  // W^T W = W * W (W symmetric), dim x dim
  Eigen::PartialPivLU<Mat> W_lu_;
};

}  // namespace conicxx
