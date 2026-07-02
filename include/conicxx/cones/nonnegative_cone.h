#pragma once

#include "conicxx/cones/cone_base.h"

namespace conicxx {

/// The nonnegative orthant R^dim_+.
class NonnegativeCone final : public ConeBase {
 public:
  explicit NonnegativeCone(Index dim);

  ConeType type() const override { return ConeType::Nonnegative; }
  Index degree() const override { return dim_; }

  const Vec& identityElement() const override { return e_; }

  void product(const Eigen::Ref<const Vec>& u, const Eigen::Ref<const Vec>& v,
               Eigen::Ref<Vec> out) const override;
  void inverseProduct(const Eigen::Ref<const Vec>& u, const Eigen::Ref<const Vec>& w,
                       Eigen::Ref<Vec> out) const override;

  void updateScaling(const Eigen::Ref<const Vec>& s, const Eigen::Ref<const Vec>& z) override;

  void applyW(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const override;
  void applyWInv(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const override;

  const Mat& scalingBlock() const override { return Hs_; }

  Scalar margin(const Eigen::Ref<const Vec>& x) const override { return x.minCoeff(); }

  void scaledUnitShift(Eigen::Ref<Vec> x, Scalar alpha) const override;

  Scalar maxStep(const Eigen::Ref<const Vec>& x, const Eigen::Ref<const Vec>& dx,
                 Scalar alpha_max) const override;

 private:
  Vec e_;   // ones
  Vec w_;   // NT scaling: w_i = sqrt(s_i/z_i)
  Mat Hs_;  // diag(w_i^2) = diag(s_i/z_i)
};

}  // namespace conicxx
