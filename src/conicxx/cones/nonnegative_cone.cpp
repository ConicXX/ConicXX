#include "conicxx/cones/nonnegative_cone.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace conicxx {

NonnegativeCone::NonnegativeCone(Index dim)
    : ConeBase(dim), e_(Vec::Ones(dim)), w_(Vec::Ones(dim)), Hs_(Mat::Identity(dim, dim)) {}

void NonnegativeCone::product(const Eigen::Ref<const Vec>& u, const Eigen::Ref<const Vec>& v,
                               Eigen::Ref<Vec> out) const {
  out.array() = u.array() * v.array();
}

void NonnegativeCone::inverseProduct(const Eigen::Ref<const Vec>& u,
                                      const Eigen::Ref<const Vec>& w, Eigen::Ref<Vec> out) const {
  out.array() = w.array() / u.array();
}

void NonnegativeCone::updateScaling(const Eigen::Ref<const Vec>& s,
                                     const Eigen::Ref<const Vec>& z) {
  w_.array() = (s.array() / z.array()).sqrt();
  Hs_.diagonal().array() = w_.array().square();
}

void NonnegativeCone::applyW(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const {
  out.array() = w_.array() * x.array();
}

void NonnegativeCone::applyWInv(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const {
  out.array() = x.array() / w_.array();
}

void NonnegativeCone::scaledUnitShift(Eigen::Ref<Vec> x, Scalar alpha) const {
  x.array() += alpha;
}

Scalar NonnegativeCone::maxStep(const Eigen::Ref<const Vec>& x, const Eigen::Ref<const Vec>& dx,
                                 Scalar alpha_max) const {
  Scalar alpha = alpha_max;
  for (Index i = 0; i < dim_; ++i) {
    if (dx[i] < 0) {
      alpha = std::min(alpha, -x[i] / dx[i]);
    }
  }
  return alpha;
}

}  // namespace conicxx
