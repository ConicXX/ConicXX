#include "conicxx/cone_set.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "conicxx/soc_kernels.h"

namespace conicxx {

namespace soc = detail::soc;

ConeSet::ConeSet(const ConeSpec& spec) : zero_dim_(spec.zero_dim), nonneg_dim_(spec.nonneg_dim) {
  const auto addBlock = [this](ConeType type, Index off, Index dim) {
    block_type_.push_back(type);
    block_off_.push_back(off);
    block_dim_.push_back(dim);
  };
  const size_t nblocks = (zero_dim_ > 0) + (nonneg_dim_ > 0) + spec.soc_dims.size();
  block_type_.reserve(nblocks);
  block_off_.reserve(nblocks);
  block_dim_.reserve(nblocks);
  soc_off_.reserve(spec.soc_dims.size());
  soc_dim_.reserve(spec.soc_dims.size());

  Index offset = 0;
  if (zero_dim_ > 0) addBlock(ConeType::Zero, 0, zero_dim_);
  offset += zero_dim_;
  if (nonneg_dim_ > 0) addBlock(ConeType::Nonnegative, offset, nonneg_dim_);
  offset += nonneg_dim_;
  degree_ = nonneg_dim_;
  for (Index d : spec.soc_dims) {
    addBlock(ConeType::SecondOrder, offset, d);
    soc_off_.push_back(offset);
    soc_dim_.push_back(d);
    all_soc3_ = all_soc3_ && d == 3;
    offset += d;
    ++degree_;
  }
  total_dim_ = offset;

  identity_ = Vec::Zero(total_dim_);
  identity_.segment(zero_dim_, nonneg_dim_).setOnes();
  for (Index off : soc_off_) identity_[off] = 1.0;
  nonneg_w_ = Vec::Ones(nonneg_dim_);
  soc_eta_ = Vec::Ones(static_cast<Index>(soc_off_.size()));
  soc_w_ = identity_;
}

template <typename F>
void ConeSet::forEachSoc(F&& f) const {
  const Index nsoc = static_cast<Index>(soc_off_.size());
  if (all_soc3_) {
    for (Index k = 0; k < nsoc; ++k) f(std::integral_constant<int, 3>{}, k, soc_off_[k], Index(3));
  } else {
    for (Index k = 0; k < nsoc; ++k) f(std::integral_constant<int, -1>{}, k, soc_off_[k], soc_dim_[k]);
  }
}

Index ConeSet::numHsEntries(Index i) const {
  const Index d = blockDim(i);
  return blockType(i) == ConeType::SecondOrder ? d * (d + 1) / 2 : d;
}

void ConeSet::writeHsLowerTriangle(Index i, Scalar* out) const {
  const Index off = blockOffset(i), d = blockDim(i);
  switch (blockType(i)) {
    case ConeType::Zero:
      std::fill(out, out + d, Scalar(0));
      break;
    case ConeType::Nonnegative:
      for (Index j = 0; j < d; ++j) out[j] = nonneg_w_[j] * nonneg_w_[j];
      break;
    case ConeType::SecondOrder: {
      const Index k = i - (zero_dim_ > 0) - (nonneg_dim_ > 0);  // blocks: zero, orthant, SOCs
      soc::writeHsLowerTriangle<-1>(soc_eta_[k], soc_w_.data() + off, out, d);
      break;
    }
  }
}

void ConeSet::product(const Vec& u, const Vec& v, Eigen::Ref<Vec> out) const {
  out.head(zero_dim_).setZero();
  out.segment(zero_dim_, nonneg_dim_).array() =
      u.segment(zero_dim_, nonneg_dim_).array() * v.segment(zero_dim_, nonneg_dim_).array();
  forEachSoc([&](auto D, Index, Index off, Index d) {
    soc::product<decltype(D)::value>(u.data() + off, v.data() + off, out.data() + off, d);
  });
}

void ConeSet::inverseProduct(const Vec& u, const Vec& w, Eigen::Ref<Vec> out) const {
  out.head(zero_dim_).setZero();
  out.segment(zero_dim_, nonneg_dim_).array() =
      w.segment(zero_dim_, nonneg_dim_).array() / u.segment(zero_dim_, nonneg_dim_).array();
  forEachSoc([&](auto D, Index, Index off, Index d) {
    soc::inverseProduct<decltype(D)::value>(u.data() + off, w.data() + off, out.data() + off, d);
  });
}

void ConeSet::updateScaling(const Vec& s, const Vec& z) {
  nonneg_w_.array() =
      (s.segment(zero_dim_, nonneg_dim_).array() / z.segment(zero_dim_, nonneg_dim_).array()).sqrt();
  Scalar* eta = soc_eta_.data();
  Scalar* w = soc_w_.data();
  forEachSoc([&](auto D, Index k, Index off, Index d) {
    eta[k] = soc::updateScaling<decltype(D)::value>(s.data() + off, z.data() + off, w + off, d);
  });
}

void ConeSet::applyW(const Vec& x, Eigen::Ref<Vec> out) const {
  out.head(zero_dim_).setZero();
  out.segment(zero_dim_, nonneg_dim_).array() = nonneg_w_.array() * x.segment(zero_dim_, nonneg_dim_).array();
  forEachSoc([&](auto D, Index k, Index off, Index d) {
    soc::applyW<decltype(D)::value>(soc_eta_[k], soc_w_.data() + off, x.data() + off, out.data() + off, d);
  });
}

void ConeSet::applyWInv(const Vec& x, Eigen::Ref<Vec> out) const {
  out.head(zero_dim_).setZero();
  out.segment(zero_dim_, nonneg_dim_).array() = x.segment(zero_dim_, nonneg_dim_).array() / nonneg_w_.array();
  forEachSoc([&](auto D, Index k, Index off, Index d) {
    soc::applyWInv<decltype(D)::value>(soc_eta_[k], soc_w_.data() + off, x.data() + off, out.data() + off, d);
  });
}

void ConeSet::mulHs(const Vec& x, Eigen::Ref<Vec> out) const {
  out.head(zero_dim_).setZero();
  out.segment(zero_dim_, nonneg_dim_).array() =
      nonneg_w_.array().square() * x.segment(zero_dim_, nonneg_dim_).array();
  forEachSoc([&](auto D, Index k, Index off, Index d) {
    soc::mulHs<decltype(D)::value>(soc_eta_[k], soc_w_.data() + off, x.data() + off, out.data() + off, d);
  });
}

std::pair<Scalar, Scalar> ConeSet::margins(const Eigen::Ref<const Vec>& x) const {
  Scalar min_margin = std::numeric_limits<Scalar>::infinity();
  Scalar pos_margin_sum = 0.0;
  if (nonneg_dim_ > 0) {
    const Scalar m = x.segment(zero_dim_, nonneg_dim_).minCoeff();
    min_margin = m;
    pos_margin_sum = std::max(m, Scalar(0));
  }
  forEachSoc([&](auto D, Index, Index off, Index d) {
    const Scalar m = soc::margin<decltype(D)::value>(x.data() + off, d);
    min_margin = std::min(min_margin, m);
    pos_margin_sum += std::max(m, Scalar(0));
  });
  if (!std::isfinite(min_margin)) min_margin = 0.0;  // only Zero-cone blocks present
  return {min_margin, pos_margin_sum};
}

Scalar ConeSet::minCentrality(const Vec& s, const Vec& z) const {
  Scalar c = std::numeric_limits<Scalar>::infinity();
  if (nonneg_dim_ > 0) {
    c = (s.segment(zero_dim_, nonneg_dim_).array() * z.segment(zero_dim_, nonneg_dim_).array()).minCoeff();
  }
  forEachSoc([&](auto D, Index, Index off, Index d) {
    c = std::min(c, soc::centrality<decltype(D)::value>(s.data() + off, z.data() + off, d));
  });
  return c;
}

void ConeSet::scaledUnitShift(Eigen::Ref<Vec> x, Scalar alpha) const {
  x.segment(zero_dim_, nonneg_dim_).array() += alpha;
  for (Index off : soc_off_) x[off] += alpha;
}

void ConeSet::projectOntoCones(Eigen::Ref<Vec> x) const {
  x.segment(zero_dim_, nonneg_dim_) = x.segment(zero_dim_, nonneg_dim_).cwiseMax(Scalar(0));
  forEachSoc([&](auto D, Index, Index off, Index d) { soc::project<decltype(D)::value>(x.data() + off, d); });
}

void ConeSet::zeroPrimalZeroConeBlocks(Eigen::Ref<Vec> s) const { s.head(zero_dim_).setZero(); }

Scalar ConeSet::maxStep(const Vec& x, const Vec& dx, Scalar alpha_max) const {
  Scalar alpha = alpha_max;
  for (Index i = zero_dim_; i < zero_dim_ + nonneg_dim_; ++i) {
    if (dx[i] < 0) alpha = std::min(alpha, -x[i] / dx[i]);
  }
  forEachSoc([&](auto D, Index, Index off, Index d) {
    alpha = soc::maxStep<decltype(D)::value>(x.data() + off, dx.data() + off, alpha, d);
  });
  return alpha;
}

}  // namespace conicxx
