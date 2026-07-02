#include "conicxx/cones/zero_cone.h"

namespace conicxx {

ZeroCone::ZeroCone(Index dim) : ConeBase(dim), e_(Vec::Zero(dim)), Hs_(Mat::Zero(dim, dim)) {}

void ZeroCone::product(const Eigen::Ref<const Vec>&, const Eigen::Ref<const Vec>&,
                        Eigen::Ref<Vec> out) const {
  out.setZero();
}

void ZeroCone::inverseProduct(const Eigen::Ref<const Vec>&, const Eigen::Ref<const Vec>&,
                               Eigen::Ref<Vec> out) const {
  out.setZero();
}

void ZeroCone::updateScaling(const Eigen::Ref<const Vec>&, const Eigen::Ref<const Vec>&) {}

void ZeroCone::applyW(const Eigen::Ref<const Vec>&, Eigen::Ref<Vec> out) const { out.setZero(); }

void ZeroCone::applyWInv(const Eigen::Ref<const Vec>&, Eigen::Ref<Vec> out) const {
  out.setZero();
}

}  // namespace conicxx
