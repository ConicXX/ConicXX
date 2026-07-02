#pragma once

#include <limits>

#include "conicxx/cones/cone_base.h"

namespace conicxx {

/// The zero cone {0}^dim, used to model equality constraints Ax_block = b_block.
/// The associated dual variable is free (dual cone of {0} is R^dim).
class ZeroCone final : public ConeBase {
 public:
  explicit ZeroCone(Index dim);

  ConeType type() const override { return ConeType::Zero; }
  Index degree() const override { return 0; }

  const Vec& identityElement() const override { return e_; }

  void product(const Eigen::Ref<const Vec>& u, const Eigen::Ref<const Vec>& v,
               Eigen::Ref<Vec> out) const override;
  void inverseProduct(const Eigen::Ref<const Vec>& u, const Eigen::Ref<const Vec>& w,
                       Eigen::Ref<Vec> out) const override;

  void updateScaling(const Eigen::Ref<const Vec>& s, const Eigen::Ref<const Vec>& z) override;

  void applyW(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const override;
  void applyWInv(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const override;

  const Mat& scalingBlock() const override { return Hs_; }

  Scalar margin(const Eigen::Ref<const Vec>& x) const override {
    (void)x;
    return std::numeric_limits<Scalar>::infinity();
  }

  void scaledUnitShift(Eigen::Ref<Vec> x, Scalar alpha) const override {
    (void)x;
    (void)alpha;
  }

  Scalar maxStep(const Eigen::Ref<const Vec>& x, const Eigen::Ref<const Vec>& dx,
                 Scalar alpha_max) const override {
    (void)x;
    (void)dx;
    return alpha_max;
  }

 private:
  Vec e_;   // identically zero
  Mat Hs_;  // identically zero (dim x dim)
};

}  // namespace conicxx
