#include "conicxx/ipm/equilibration.h"

#include <algorithm>
#include <cmath>

namespace conicxx::detail {

namespace {

// Clarabel's limit_scaling for a row/column norm: zero rows and columns are not scaled, norms
// are clipped to [min_scale, max_scale].
Scalar limitNorm(Scalar v, Scalar min_scale, Scalar max_scale) {
  if (v < min_scale) return 1.0;
  return std::min(v, max_scale);
}

}  // namespace

void Equilibration::compute(const SparseMat& P_upper, const Vec& q, const SparseMat& A,
                            const Vec& /*b*/, const Vec& compliance, const ConeSet& cones,
                            const Settings& settings) {
  const Index n = static_cast<Index>(P_upper.rows());
  const Index m = static_cast<Index>(A.rows());
  d_ = Vec::Ones(n);
  e_ = Vec::Ones(m);
  c_ = 1.0;

  if (!settings.equilibrate || n == 0) return;

  // Ruiz iterations as in Clarabel (equilibration.rs): each iteration's factors are clipped so
  // that the cumulative d, e and c stay within [min_scale, max_scale], the data are scaled with
  // exactly these factors, and the cost is rescaled inside the loop by
  // 1 / max(mean column inf-norm of P, ||q||_inf).
  SparseMat Pw = P_upper;
  SparseMat Aw = A;
  Vec qw = q;
  // C_hat follows the scaling but, like Hs, belongs to the (2,2) block of the KKT matrix and does
  // not enter the norms (as in Clarabel, which equilibrates [P A'; A 0]).
  Vec cw = compliance;

  const Scalar minS = settings.equilibrate_min_scale;
  const Scalar maxS = settings.equilibrate_max_scale;
  Vec norm_x(n), norm_z(m), delta_x(n), delta_z(m), p_col(n);

  for (int iter = 0; iter < settings.equilibrate_max_iter; ++iter) {
    norm_x.setZero();
    norm_z.setZero();
    for (Index c = 0; c < Pw.outerSize(); ++c) {
      for (SparseMat::InnerIterator it(Pw, c); it; ++it) {
        const Index r = static_cast<Index>(it.row());
        const Scalar av = std::abs(it.value());
        norm_x[r] = std::max(norm_x[r], av);
        norm_x[c] = std::max(norm_x[c], av);
      }
    }
    for (Index c = 0; c < Aw.outerSize(); ++c) {
      for (SparseMat::InnerIterator it(Aw, c); it; ++it) {
        const Index r = static_cast<Index>(it.row());
        const Scalar av = std::abs(it.value());
        norm_x[c] = std::max(norm_x[c], av);
        norm_z[r] = std::max(norm_z[r], av);
      }
    }

    // One factor per SOC block preserves the cone.
    for (Index bi = 0; bi < cones.numBlocks(); ++bi) {
      if (cones.blockType(bi) != ConeType::SecondOrder) continue;
      const Index off = cones.blockOffset(bi);
      const Index d = cones.blockDim(bi);
      norm_z.segment(off, d).setConstant(norm_z.segment(off, d).maxCoeff());
    }

    for (Index i = 0; i < n; ++i) {
      const Scalar f = 1.0 / std::sqrt(limitNorm(norm_x[i], minS, maxS));
      delta_x[i] = std::clamp(f, minS / d_[i], maxS / d_[i]);
    }
    for (Index i = 0; i < m; ++i) {
      const Scalar f = 1.0 / std::sqrt(limitNorm(norm_z[i], minS, maxS));
      delta_z[i] = std::clamp(f, minS / e_[i], maxS / e_[i]);
    }

    for (Index c = 0; c < Pw.outerSize(); ++c) {
      for (SparseMat::InnerIterator it(Pw, c); it; ++it) it.valueRef() *= delta_x[it.row()] * delta_x[c];
    }
    for (Index c = 0; c < Aw.outerSize(); ++c) {
      for (SparseMat::InnerIterator it(Aw, c); it; ++it) it.valueRef() *= delta_z[it.row()] * delta_x[c];
    }
    qw.array() *= delta_x.array();
    cw.array() *= delta_z.head(cw.size()).array().square();
    d_.array() *= delta_x.array();
    e_.array() *= delta_z.array();

    // Cost scaling: mean column inf-norm of the symmetric P, or ||q||_inf if larger.
    p_col.setZero();
    for (Index c = 0; c < Pw.outerSize(); ++c) {
      for (SparseMat::InnerIterator it(Pw, c); it; ++it) {
        const Scalar av = std::abs(it.value());
        p_col[it.row()] = std::max(p_col[it.row()], av);
        p_col[c] = std::max(p_col[c], av);
      }
    }
    const Scalar cost_norm = std::max(p_col.mean(), n > 0 ? qw.lpNorm<Eigen::Infinity>() : Scalar(0));
    if (cost_norm > 0) {
      const Scalar ctmp = std::clamp(1.0 / limitNorm(cost_norm, minS, maxS), minS / c_, maxS / c_);
      Pw *= ctmp;
      qw *= ctmp;
      cw /= ctmp;
      c_ *= ctmp;
    }
  }
}

void Equilibration::scaleP(SparseMat& P_upper) const {
  for (Index c = 0; c < P_upper.outerSize(); ++c) {
    for (SparseMat::InnerIterator it(P_upper, c); it; ++it) {
      it.valueRef() *= c_ * d_[it.row()] * d_[c];
    }
  }
}

void Equilibration::scaleA(SparseMat& A) const {
  for (Index c = 0; c < A.outerSize(); ++c) {
    for (SparseMat::InnerIterator it(A, c); it; ++it) {
      it.valueRef() *= e_[it.row()] * d_[c];
    }
  }
}

void Equilibration::scaleQ(Vec& q) const { q.array() *= c_ * d_.array(); }

void Equilibration::scaleB(Vec& b) const { b.array() *= e_.array(); }

void Equilibration::scaleCompliance(Vec& compliance) const {
  compliance.array() *= e_.head(compliance.size()).array().square() / c_;
}

void Equilibration::scaleProblem(SparseMat& P_upper, Vec& q, SparseMat& A, Vec& b) const {
  scaleP(P_upper);
  scaleA(A);
  scaleQ(q);
  scaleB(b);
}

void Equilibration::unscaleSolution(Vec& x, Vec& s, Vec& z) const {
  x.array() *= d_.array();
  z.array() = z.array() * e_.array() / c_;
  s.array() /= e_.array();
}

void Equilibration::scaleSolution(Vec& x, Vec& s, Vec& z) const {
  x.array() /= d_.array();
  z.array() = z.array() * c_ / e_.array();
  s.array() *= e_.array();
}

}  // namespace conicxx::detail
