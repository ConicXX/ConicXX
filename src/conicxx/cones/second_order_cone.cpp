#include "conicxx/cones/second_order_cone.h"

#include "conicxx/cones/soc_kernels.h"

namespace conicxx {

SecondOrderCone::SecondOrderCone(Index dim) : ConeBase(dim), e_(Vec::Zero(dim)), w_(dim), scratch_(dim) {
  e_[0] = 1.0;
  w_[0] = 1.0;
  w_.tail(dim - 1).setZero();
}

void SecondOrderCone::product(const Eigen::Ref<const Vec>& u, const Eigen::Ref<const Vec>& v,
                              Eigen::Ref<Vec> out) const {
  detail::soc::product<-1>(u.data(), v.data(), out.data(), dim_);
}

void SecondOrderCone::inverseProduct(const Eigen::Ref<const Vec>& u, const Eigen::Ref<const Vec>& w,
                                     Eigen::Ref<Vec> out) const {
  detail::soc::inverseProduct<-1>(u.data(), w.data(), out.data(), dim_);
}

void SecondOrderCone::updateScaling(const Eigen::Ref<const Vec>& s, const Eigen::Ref<const Vec>& z) {
  eta_ = detail::soc::updateScaling<-1>(s.data(), z.data(), w_.data(), dim_);
}

void SecondOrderCone::applyW(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const {
  scratch_ = x;
  detail::soc::applyW<-1>(eta_, w_.data(), scratch_.data(), out.data(), dim_);
}

void SecondOrderCone::applyWInv(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const {
  scratch_ = x;
  detail::soc::applyWInv<-1>(eta_, w_.data(), scratch_.data(), out.data(), dim_);
}

void SecondOrderCone::mulHs(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const {
  scratch_ = x;
  detail::soc::mulHs<-1>(eta_, w_.data(), scratch_.data(), out.data(), dim_);
}

void SecondOrderCone::writeHsLowerTriangle(Eigen::Ref<Vec> out) const {
  detail::soc::writeHsLowerTriangle<-1>(eta_, w_.data(), out.data(), dim_);
}

Scalar SecondOrderCone::margin(const Eigen::Ref<const Vec>& x) const {
  return detail::soc::margin<-1>(x.data(), dim_);
}

Scalar SecondOrderCone::centrality(const Eigen::Ref<const Vec>& s, const Eigen::Ref<const Vec>& z) const {
  return detail::soc::centrality<-1>(s.data(), z.data(), dim_);
}

Scalar SecondOrderCone::maxStep(const Eigen::Ref<const Vec>& x, const Eigen::Ref<const Vec>& dx,
                                Scalar alpha_max) const {
  return detail::soc::maxStep<-1>(x.data(), dx.data(), alpha_max, dim_);
}

}  // namespace conicxx
