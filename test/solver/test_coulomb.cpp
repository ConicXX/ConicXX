#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <Eigen/LU>

#include <cmath>
#include <random>

#include "conicxx/cones/cone_set.h"
#include "conicxx/cones/coulomb_map.h"
#include "conicxx/kkt/kkt_system.h"
#include "conicxx/kkt/quasidefinite_ldlt.h"
#include "conicxx/solver.h"
#include "test_helpers.h"

using namespace conicxx;
using conicxx::detail::CoulombMap;

namespace {

Scalar tangentialNorm(const Vec& v, Index off, Index d) { return v.segment(off + 1, d - 1).norm(); }

/// Checks the exact Coulomb conditions directly on the unscaled solution: stationarity, the primal
/// equation with the unsmoothed De Saxce term, both cones and complementarity per cone.
void expectCoulombSolution(const Mat& P, const Vec& q, const Mat& A, const Vec& b, const ConeSpec& spec,
                           const Solution& sol, Scalar tol) {
  ASSERT_TRUE(sol.ok()) << "status=" << toString(sol.status);
  Vec rp = A * sol.x + sol.s - b;
  CoulombMap map(spec);
  map.subtractExactShift(sol.s, rp, 1.0);
  EXPECT_LT((P * sol.x + q + A.transpose() * sol.z).lpNorm<Eigen::Infinity>(), tol) << "stationarity";
  EXPECT_LT(rp.lpNorm<Eigen::Infinity>(), tol) << "primal equation";
  ConeSet cones(spec);
  EXPECT_GT(cones.margins(sol.s).first, -tol) << "s not in K";
  EXPECT_GT(cones.margins(sol.z).first, -tol) << "z not in K";
  Vec prod(sol.s.size());
  cones.product(sol.s, sol.z, prod);
  // The solver's complementarity measure is relative (velocity and impulse errors, see
  // CoulombIpm::Metrics::comp), so the product is checked relative to the magnitudes.
  const Scalar scale = (1 + sol.s.lpNorm<Eigen::Infinity>()) * (1 + sol.z.lpNorm<Eigen::Infinity>());
  EXPECT_LT(prod.lpNorm<Eigen::Infinity>(), tol * scale) << "complementarity";
}

/// One point mass with velocity (v_n, v_t1, v_t2) on a plane, friction coefficient mu, time step h:
/// M v - M v_free = H' r, u = H v (no offset), scaled to the unit cone with S = diag(1, mu, mu).
struct BlockOnPlane {
  Scalar m = 2.0, mu = 0.5;
  Vec v_free = Vec::Zero(3);

  Mat P() const { return m * Mat::Identity(3, 3); }
  Vec q() const { return -m * v_free; }
  Mat A() const { return -Eigen::Vector3d(1.0, mu, mu).asDiagonal().toDenseMatrix(); }
  Vec b() const { return Vec::Zero(3); }
};

ConeSpec coulombSpec(Index ncones) {
  ConeSpec spec;
  spec.soc_dims.assign(static_cast<size_t>(ncones), 3);
  spec.soc_coulomb.assign(static_cast<size_t>(ncones), 1);
  return spec;
}

Settings tightSettings() {
  Settings s;
  s.tol_feas = s.tol_gap_abs = s.tol_gap_rel = 1e-9;
  return s;
}

}  // namespace

TEST(CoulombMap, ValueAndJacobianMatchFiniteDifferences) {
  ConeSpec spec;
  spec.nonneg_dim = 1;
  spec.soc_dims = {3, 3, 2};
  spec.soc_coulomb = {1, 0, 1};
  CoulombMap map(spec);
  ASSERT_EQ(map.numCones(), 2);
  std::mt19937 gen(3);
  std::normal_distribution<Scalar> nd;
  for (Scalar eps : {0.0, 1e-3, 0.3}) {
    for (int trial = 0; trial < 20; ++trial) {
      Vec s(spec.totalDim());
      for (Index i = 0; i < s.size(); ++i) s[i] = nd(gen);
      if (trial == 0) s.segment(map.offset(0) + 1, 2).setZero();  // s_bar = 0
      map.evaluate(s, eps);
      const Vec ell = map.ell();
      const Vec jac = map.jac();
      for (Index i = 0; i < map.numCones(); ++i) {
        const Index off = map.offset(i), d = map.dim(i);
        const Scalar t = tangentialNorm(s, off, d);
        EXPECT_NEAR(ell[i], std::sqrt(t * t + eps * eps), 1e-15);
        EXPECT_LE(ell[i] - t, eps + 1e-15);  // smoothing error bound
        EXPECT_EQ(jac[off], 0.0);
        if (eps == 0.0 && t == 0.0) continue;  // kink of the exact term
        for (Index j = 1; j < d; ++j) {
          const Scalar h = 1e-6;
          Vec sp = s, sm = s;
          sp[off + j] += h;
          sm[off + j] -= h;
          map.evaluate(sp, eps);
          const Scalar lp = map.ell()[i];
          map.evaluate(sm, eps);
          const Scalar lm = map.ell()[i];
          EXPECT_NEAR(jac[off + j], (lp - lm) / (2 * h), 1e-7);
        }
        map.evaluate(s, eps);
      }
    }
  }
}

TEST(CoulombMap, BAndItsInverse) {
  ConeSpec spec;
  spec.zero_dim = 2;
  spec.soc_dims = {3, 3};
  spec.soc_coulomb = {1, 1};
  CoulombMap map(spec);
  std::mt19937 gen(5);
  std::normal_distribution<Scalar> nd;
  Vec s(spec.totalDim()), v(spec.totalDim());
  for (Index i = 0; i < s.size(); ++i) {
    s[i] = nd(gen);
    v[i] = nd(gen);
  }
  map.evaluate(s, 0.1);
  for (Scalar theta : {0.0, 0.5, 1.0}) {
    Vec w = v;
    map.applyB(w, theta);
    map.applyBinv(w, theta);
    EXPECT_LT((w - v).lpNorm<Eigen::Infinity>(), 1e-15);
    // (E L)^2 = 0: applying B twice adds 2 theta E L v.
    Vec once = v, twice = v;
    map.applyB(once, theta);
    map.applyB(twice, theta);
    map.applyB(twice, theta);
    EXPECT_LT((twice - (2 * once - v)).lpNorm<Eigen::Infinity>(), 1e-14);
  }
}

TEST(QuasiDefiniteLdlt, UnsymmetricFactorizationMatchesDenseLu) {
  // Quasidefinite pattern [P A'; A -H] with unsymmetric values in the (2,2) block and A.
  std::mt19937 gen(11);
  std::uniform_real_distribution<Scalar> ud(-1, 1);
  const Index n = 6, m = 9, dim = n + m;
  Mat K = Mat::Zero(dim, dim);
  for (Index i = 0; i < n; ++i) K(i, i) = 3 + ud(gen);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      if (ud(gen) > 0.2) continue;
      K(n + i, j) = ud(gen);
      K(j, n + i) = K(n + i, j) + 0.3 * ud(gen);
    }
  }
  for (Index blk = 0; blk < 3; ++blk) {
    Mat H = Mat::Random(3, 3);
    H = H * H.transpose() + Mat::Identity(3, 3);
    H.row(0) += 0.5 * Mat::Random(1, 3);  // unsymmetric first row
    K.block(n + 3 * blk, n + 3 * blk, 3, 3) = -H;
  }
  // Lower triangle with the union pattern, upper values transposed onto it.
  std::vector<Triplet> trip;
  for (Index c = 0; c < dim; ++c) {
    for (Index r = c; r < dim; ++r) {
      if (K(r, c) != 0 || K(c, r) != 0 || r == c) trip.emplace_back(r, c, K(r, c));
    }
  }
  SparseMat L = testutil::makeSparse(dim, dim, trip);
  std::vector<Scalar> upper(static_cast<size_t>(L.nonZeros()));
  for (Index c = 0; c < dim; ++c) {
    for (Index p = L.outerIndexPtr()[c]; p < L.outerIndexPtr()[c + 1]; ++p) upper[p] = K(c, L.innerIndexPtr()[p]);
  }
  std::vector<signed char> signs(dim, -1);
  for (Index i = 0; i < n; ++i) signs[i] = 1;
  for (bool supernodal : {false, true}) {
    detail::QuasiDefiniteLdlt ldlt;
    ldlt.params.supernodal = supernodal;
    ASSERT_TRUE(ldlt.analyzePattern(L, signs));
    ASSERT_TRUE(ldlt.factorizeUnsymmetric(L, upper.data()));
    EXPECT_EQ(ldlt.numDynamicRegularized(), 0);
    const Vec rhs = Vec::Random(dim);
    Vec x;
    ldlt.solve(rhs, x);
    const Vec x_ref = K.fullPivLu().solve(rhs);
    EXPECT_LT((x - x_ref).norm() / x_ref.norm(), 1e-12);
    // The symmetric factorization afterwards is unaffected.
    ASSERT_TRUE(ldlt.factorize(L));
    ldlt.solve(rhs, x);
    Mat Ks = Mat(L).selfadjointView<Eigen::Lower>();
    EXPECT_LT((x - Ks.fullPivLu().solve(rhs)).norm(), 1e-10);
  }
}

TEST(QuasiDefiniteLdlt, UnsymmetricSupernodalMatchesSimplicial) {
  // Dense enough for supernodes with wide dense blocks (several panels).
  std::mt19937 gen(13);
  std::uniform_real_distribution<Scalar> ud(-1, 1);
  const Index n = 60, m = 90, dim = n + m;
  Mat K = Mat::Zero(dim, dim);
  Mat Pd = Mat::Random(n, n);
  K.topLeftCorner(n, n) = Pd * Pd.transpose() / n + Mat::Identity(n, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      if (ud(gen) > 0.0) continue;
      K(n + i, j) = ud(gen);
      K(j, n + i) = K(n + i, j);
    }
  }
  for (Index blk = 0; blk < m / 3; ++blk) {
    Mat H = Mat::Random(3, 3);
    H = H * H.transpose() + Mat::Identity(3, 3);
    H.row(0) += 0.5 * Mat::Random(1, 3);
    K.block(n + 3 * blk, n + 3 * blk, 3, 3) = -H;
  }
  std::vector<Triplet> trip;
  for (Index c = 0; c < dim; ++c) {
    for (Index r = c; r < dim; ++r) {
      if (K(r, c) != 0 || K(c, r) != 0 || r == c) trip.emplace_back(r, c, K(r, c));
    }
  }
  SparseMat L = testutil::makeSparse(dim, dim, trip);
  std::vector<Scalar> upper(static_cast<size_t>(L.nonZeros()));
  for (Index c = 0; c < dim; ++c) {
    for (Index p = L.outerIndexPtr()[c]; p < L.outerIndexPtr()[c + 1]; ++p) upper[p] = K(c, L.innerIndexPtr()[p]);
  }
  std::vector<signed char> signs(dim, -1);
  for (Index i = 0; i < n; ++i) signs[i] = 1;
  const Vec rhs = Vec::Random(dim);
  const Vec x_ref = K.fullPivLu().solve(rhs);
  Vec x[2];
  for (bool supernodal : {false, true}) {
    detail::QuasiDefiniteLdlt ldlt;
    ldlt.params.supernodal = supernodal;
    ldlt.params.refine_max_iter = 0;  // compare the factors, not the refined solutions
    ASSERT_TRUE(ldlt.analyzePattern(L, signs));
    ASSERT_EQ(ldlt.isSupernodal(), supernodal);
    ASSERT_TRUE(ldlt.factorizeUnsymmetric(L, upper.data()));
    EXPECT_EQ(ldlt.numDynamicRegularized(), 0);
    ldlt.solve(rhs, x[supernodal]);
    EXPECT_LT((x[supernodal] - x_ref).norm() / x_ref.norm(), 1e-10) << "supernodal=" << supernodal;
  }
}

TEST(CoulombKkt, ReducedNewtonMatrixMatchesUnreducedSystem) {
  // [P A'; A -(I - theta E L) Hs] (dx, dz) = rhs against the unreduced system in (dx, dz, ds):
  // P dx + A'dz = r1, A dx + (I - theta E L) ds = r2, ds + Hs dz = r3.
  std::mt19937 gen(7);
  std::uniform_real_distribution<Scalar> ud(-1, 1);
  ConeSpec spec;
  spec.zero_dim = 1;
  spec.nonneg_dim = 1;
  spec.soc_dims = {3, 3, 3};
  spec.soc_coulomb = {1, 0, 1};
  const Index n = 5, m = spec.totalDim();
  Mat Pd = Mat::Random(n, n);
  Pd = Pd * Pd.transpose() + Mat::Identity(n, n);
  Mat Ad = Mat::Random(m, n);
  SparseMat P = SparseMat(Pd.sparseView()).triangularView<Eigen::Upper>();
  SparseMat A = Ad.sparseView();
  P.makeCompressed();
  A.makeCompressed();

  ConeSet cones(spec);
  Vec s = Vec::Zero(m), z = Vec::Zero(m);
  for (Index i = 1; i < m; ++i) {
    s[i] = 0.3 * ud(gen);
    z[i] = 0.3 * ud(gen);
  }
  cones.scaledUnitShift(s, 1.0);
  cones.scaledUnitShift(z, 1.0);
  cones.updateScaling(s, z);
  CoulombMap map(spec);
  map.evaluate(s, 0.01);

  Mat Hs = Mat::Zero(m, m);
  for (Index j = 0; j < m; ++j) {
    Vec ej = Vec::Unit(m, j), col(m);
    cones.mulHs(ej, col);
    Hs.col(j) = col;
  }
  for (Scalar theta : {0.0, 0.5, 1.0}) {
    Vec row0(m);
    cones.mulHs(map.jac(), row0);
    row0 *= -theta;
    detail::KktSystem kkt;
    Settings settings;
    ASSERT_TRUE(kkt.setup(P, A, cones, settings));
    ASSERT_TRUE(kkt.updateScalingAndFactorize(cones, map.blocks(), row0));

    Mat Binv = Mat::Identity(m, m);
    for (Index i = 0; i < map.numCones(); ++i) {
      for (Index j = 1; j < 3; ++j) Binv(map.offset(i), map.offset(i) + j) = -theta * map.jac()[map.offset(i) + j];
    }
    const Vec r1 = Vec::Random(n), r2 = Vec::Random(m), r3 = Vec::Random(m);
    Mat U = Mat::Zero(n + 2 * m, n + 2 * m);
    U.block(0, 0, n, n) = Pd;
    U.block(0, n, n, m) = Ad.transpose();
    U.block(n, 0, m, n) = Ad;
    U.block(n, n + m, m, m) = Binv;
    U.block(n + m, n, m, m) = Hs;
    U.block(n + m, n + m, m, m) = Mat::Identity(m, m);
    // The zero cone has s = 0: its ds row reads ds = 0 there.
    U.block(n + m, n, 1, m).setZero();
    Vec rhs_u(n + 2 * m);
    rhs_u << r1, r2, r3;
    rhs_u[n + m] = 0;
    const Vec ref = U.fullPivLu().solve(rhs_u);

    Vec r3z = r3;
    r3z[0] = 0;
    Vec rhs(n + m), sol;
    rhs << r1, r2 - Binv * r3z;
    kkt.solve(rhs, sol);
    EXPECT_LT((sol - ref.head(n + m)).norm() / ref.head(n + m).norm(), 1e-12) << "theta=" << theta;
  }
}

TEST(SolveCoulomb, SingleContactStickSlipTakeOff) {
  BlockOnPlane bp;
  const ConeSpec spec = coulombSpec(1);
  struct Case {
    Vec v_free;
    ContactState state;
  };
  // v_free = (normal, t1, t2); m g h = m * 1 here, mu r_n = 1.
  const std::vector<Case> cases = {{(Vec(3) << -1.0, 0.3, -0.2).finished(), ContactState::Stick},
                                   {(Vec(3) << -1.0, 1.5, 2.0).finished(), ContactState::Slip},
                                   {(Vec(3) << 0.7, 1.5, 2.0).finished(), ContactState::TakeOff}};
  for (const Case& c : cases) {
    bp.v_free = c.v_free;
    Vec v_exp = c.v_free, r_exp = Vec::Zero(3);
    if (c.v_free[0] < 0) {
      v_exp[0] = 0;
      r_exp[0] = -bp.m * c.v_free[0];
      const Scalar vt = c.v_free.tail(2).norm();
      if (bp.m * vt <= bp.mu * r_exp[0]) {
        v_exp.tail(2).setZero();
        r_exp.tail(2) = -bp.m * c.v_free.tail(2);
      } else {
        r_exp.tail(2) = -bp.mu * r_exp[0] * c.v_free.tail(2) / vt;
        v_exp.tail(2) = c.v_free.tail(2) + r_exp.tail(2) / bp.m;
      }
    }
    Solver solver;
    ASSERT_TRUE(solver.setup(bp.P(), bp.q(), bp.A(), bp.b(), spec, tightSettings()));
    const Solution& sol = solver.solve();
    expectCoulombSolution(bp.P(), bp.q(), bp.A(), bp.b(), spec, sol, 1e-8);
    testutil::expectVecNear(sol.x, v_exp, 1e-8);
    const Vec r = Eigen::Vector3d(1.0, bp.mu, bp.mu).asDiagonal() * sol.z;  // z = S^-1 r
    testutil::expectVecNear(r, r_exp, 1e-8);
    ASSERT_EQ(sol.contact_states.size(), 1u);
    EXPECT_EQ(sol.contact_states[0], c.state);
  }
}

TEST(SolveCoulomb, ConvexRelaxationSeparatesInSlipExactDoesNot) {
  BlockOnPlane bp;
  bp.v_free = (Vec(3) << -1.0, 1.5, 2.0).finished();
  ConeSpec convex;
  convex.soc_dims = {3};
  Solver relaxed;
  ASSERT_TRUE(relaxed.setup(bp.P(), bp.q(), bp.A(), bp.b(), convex, tightSettings()));
  const Solution& sol_relaxed = relaxed.solve();
  ASSERT_TRUE(sol_relaxed.ok());
  // The relaxation's sliding contact separates with u_N = mu ||u_T||.
  EXPECT_GT(sol_relaxed.x[0], 1e-3);
  EXPECT_NEAR(sol_relaxed.x[0], bp.mu * sol_relaxed.x.tail(2).norm(), 1e-7);

  Solver exact;
  ASSERT_TRUE(exact.setup(bp.P(), bp.q(), bp.A(), bp.b(), coulombSpec(1), tightSettings()));
  const Solution& sol = exact.solve();
  ASSERT_TRUE(sol.ok());
  EXPECT_NEAR(sol.x[0], 0.0, 1e-8);  // relative tolerance 1e-9 of velocities of order 1
}

TEST(SolveCoulomb, HyperstaticBoxSlidesWithUniqueVelocity) {
  // Rigid box (mass m, inertia diag(J)) with its center 0.5 above a plane, four corner contacts
  // (+-1, +-1, -0.5): H is 12 x 6 of rank 6 (hyperstatic). Pushed sideways beyond the friction
  // limit, it slides without rotating: v_n = 0 at all corners, omega = 0, and the tangential
  // velocity drops by mu g h along the sliding direction.
  const Scalar m = 1.0, mu = 0.3, g = 9.81, h = 0.01;
  const Eigen::Vector3d J(0.4, 0.4, 0.6);
  Mat P = Mat::Zero(6, 6);
  P.diagonal() << m, m, m, J;
  Vec v_free(6);
  v_free << 1.0, 0.5, -g * h, 0, 0, 0;
  const Vec q = -P * v_free;
  Mat A = Mat::Zero(12, 6);
  const Eigen::Vector3d n(0, 0, 1), t1(1, 0, 0), t2(0, 1, 0);
  int k = 0;
  for (Scalar px : {-1.0, 1.0}) {
    for (Scalar py : {-1.0, 1.0}) {
      const Eigen::Vector3d p(px, py, -0.5);
      const Eigen::Vector3d dirs[3] = {n, mu * t1, mu * t2};
      for (int r = 0; r < 3; ++r) {
        A.block(3 * k + r, 0, 1, 3) = -dirs[r].transpose();
        A.block(3 * k + r, 3, 1, 3) = -p.cross(dirs[r]).transpose();
      }
      ++k;
    }
  }
  const Vec b = Vec::Zero(12);
  const ConeSpec spec = coulombSpec(4);
  Solver solver;
  ASSERT_TRUE(solver.setup(P, q, A, b, spec, tightSettings()));
  const Solution& sol = solver.solve();
  expectCoulombSolution(P, q, A, b, spec, sol, 1e-8);
  Vec v_exp = Vec::Zero(6);
  v_exp.head(2) = v_free.head(2) * (1 - mu * g * h / v_free.head(2).norm());
  testutil::expectVecNear(sol.x, v_exp, 1e-7);
  for (ContactState st : sol.contact_states) EXPECT_EQ(st, ContactState::Slip);
}

TEST(SolveCoulomb, CompliantRowAndFrictionlessContact) {
  // Point mass on a plane (Coulomb cone) tied by a compliant spring row along t1 to the origin
  // (zero cone, compliance c) and touching a frictionless wall at x = 0 (orthant). Checked against
  // the Coulomb conditions only.
  const Scalar m = 1.0, mu = 0.4, c = 0.05;
  Mat P = m * Mat::Identity(3, 3);
  Vec v_free(3);
  v_free << -0.2, 0.8, -1.0;  // (x, y, z): into the wall, away from the spring, onto the floor
  const Vec q = -P * v_free;
  // Rows: spring (zero) on y; wall on x (orthant); contact on the floor along z with tangents x, y.
  Mat A(5, 3);
  A << 0, 1, 0,      // spring: v_y - c z_E + s = b, s = 0
      -1, 0, 0,      // wall: s = v_x >= 0
      0, 0, -1,      // floor normal: v_z
      -mu, 0, 0,     // tangent x
      0, -mu, 0;     // tangent y
  Vec b = Vec::Zero(5);
  b[0] = -0.1;
  ConeSpec spec;
  spec.zero_dim = 1;
  spec.nonneg_dim = 1;
  spec.soc_dims = {3};
  spec.soc_coulomb = {1};
  Vec compliance(1);
  compliance << c;
  Solver solver;
  ASSERT_TRUE(solver.setup(testutil::makeSparse(3, 3, {{0, 0, m}, {1, 1, m}, {2, 2, m}}), q,
                           SparseMat(A.sparseView()), b, spec, tightSettings(), &compliance));
  const Solution& sol = solver.solve();
  ASSERT_TRUE(sol.ok()) << toString(sol.status);
  Vec rp = A * sol.x + sol.s - b;
  rp[0] -= c * sol.z[0];
  CoulombMap map(spec);
  map.subtractExactShift(sol.s, rp, 1.0);
  EXPECT_LT(rp.lpNorm<Eigen::Infinity>(), 1e-8);
  EXPECT_LT((P * sol.x + q + A.transpose() * sol.z).lpNorm<Eigen::Infinity>(), 1e-8);
  EXPECT_NEAR(sol.s[0], 0.0, 1e-12);
  EXPECT_NEAR(sol.x[2], 0.0, 1e-8);  // floor contact closed, no separation in slip
}

TEST(SolveCoulomb, WarmStartAfterDataUpdate) {
  BlockOnPlane bp;
  bp.v_free = (Vec(3) << -1.0, 1.5, 2.0).finished();
  Solver solver;
  ASSERT_TRUE(solver.setup(bp.P(), bp.q(), bp.A(), bp.b(), coulombSpec(1), tightSettings()));
  ASSERT_TRUE(solver.solve().ok());
  bp.v_free = (Vec(3) << -1.0, 1.4, 2.05).finished();
  const Vec q = bp.q();
  ASSERT_TRUE(solver.updateData(nullptr, &q, nullptr, nullptr));
  const Solution& sol = solver.solve();
  expectCoulombSolution(bp.P(), q, bp.A(), bp.b(), coulombSpec(1), sol, 1e-8);
  EXPECT_TRUE(sol.info.warm_start_used);
}

TEST(SolveCoulomb, StickingWithAndWithoutSmoothing) {
  // Sticking contact: s_bar -> 0 at the solution, where the unsmoothed term has its kink.
  BlockOnPlane bp;
  bp.v_free = (Vec(3) << -1.0, 0.3, -0.2).finished();
  for (Scalar smoothing : {0.0, 0.1, 1.0}) {
    Settings settings = tightSettings();
    settings.coulomb.smoothing = smoothing;
    Solver solver;
    ASSERT_TRUE(solver.setup(bp.P(), bp.q(), bp.A(), bp.b(), coulombSpec(1), settings));
    const Solution& sol = solver.solve();
    expectCoulombSolution(bp.P(), bp.q(), bp.A(), bp.b(), coulombSpec(1), sol, 1e-8);
    EXPECT_LT(sol.x.norm(), 1e-8) << "smoothing=" << smoothing;
  }
}
