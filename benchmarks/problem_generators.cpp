#include "problem_generators.h"

#include <conicxx/cones/cone_set.h>

#include <cmath>
#include <vector>

namespace conicxx::bench {

namespace {

Vec randomUniform(Index n, std::mt19937& rng, Scalar lo, Scalar hi) {
  std::uniform_real_distribution<Scalar> dist(lo, hi);
  Vec v(n);
  for (Index i = 0; i < n; ++i) v[i] = dist(rng);
  return v;
}

Mat randomNormalDense(Index rows, Index cols, std::mt19937& rng, Scalar stddev) {
  std::normal_distribution<Scalar> dist(0.0, stddev);
  Mat M(rows, cols);
  for (Index j = 0; j < cols; ++j) {
    for (Index i = 0; i < rows; ++i) M(i, j) = dist(rng);
  }
  return M;
}

/// Fills a strictly-interior point for one cone block, writing into
/// out.segment(offset, dim). Used to build both s0 and z0 below.
void fillInteriorBlock(ConeType type, Index offset, Index dim, std::mt19937& rng,
                       Eigen::Ref<Vec> out) {
  switch (type) {
    case ConeType::Zero:
      out.segment(offset, dim).setZero();
      break;
    case ConeType::Nonnegative:
      out.segment(offset, dim) = randomUniform(dim, rng, 0.5, 1.5);
      break;
    case ConeType::SecondOrder: {
      const Scalar r = 2.0;
      out[offset] = r;
      if (dim > 1) {
        Vec tail = randomUniform(dim - 1, rng, -1.0, 1.0);
        const Scalar n = tail.norm();
        if (n > 1e-12) tail *= (0.3 * r) / n;
        out.segment(offset + 1, dim - 1) = tail;
      }
      break;
    }
  }
}

/// Constructs (q, b) from an explicit strictly-feasible primal-dual pair
/// (x0, s0, z0) satisfying A*x0+s0=b and P*x0+A'*z0+q=0 exactly, so the
/// resulting problem is guaranteed solvable with a finite optimum --
/// avoids ever benchmarking an accidentally infeasible/unbounded random
/// instance, which would make "iterations to converge" meaningless.
void makeFeasiblePrimalDualData(const SparseMat& P, const SparseMat& A, const ConeSpec& spec,
                                std::mt19937& rng, Vec& q_out, Vec& b_out) {
  const ConeSet cones(spec);
  const Index n = static_cast<Index>(P.rows());
  const Index m = cones.totalDim();

  const Vec x0 = randomUniform(n, rng, -1.0, 1.0);
  Vec s0 = Vec::Zero(m), z0 = Vec::Zero(m);
  for (Index i = 0; i < cones.numBlocks(); ++i) {
    const ConeBase& blk = cones.block(i);
    fillInteriorBlock(blk.type(), cones.blockOffset(i), blk.dim(), rng, s0);
    fillInteriorBlock(blk.type(), cones.blockOffset(i), blk.dim(), rng, z0);
  }

  b_out = A * x0 + s0;
  const Vec Px0 = P.selfadjointView<Eigen::Upper>() * x0;
  q_out = -Px0 - A.transpose() * z0;
}

std::vector<Triplet> denseToTriplets(const Mat& M, Index row_offset = 0, Index col_offset = 0) {
  std::vector<Triplet> triplets;
  triplets.reserve(static_cast<size_t>(M.rows() * M.cols()));
  for (Index j = 0; j < M.cols(); ++j) {
    for (Index i = 0; i < M.rows(); ++i) {
      triplets.emplace_back(row_offset + i, col_offset + j, M(i, j));
    }
  }
  return triplets;
}

}  // namespace

std::mt19937 makeRng(const std::string& family, Index size, int trial) {
  const unsigned family_hash = static_cast<unsigned>(std::hash<std::string>{}(family));
  std::seed_seq seed{family_hash, static_cast<unsigned>(size), static_cast<unsigned>(trial)};
  return std::mt19937(seed);
}

BenchProblem makeRandomQp(Index n, std::mt19937& rng) {
  BenchProblem prob;
  prob.family = "RandomQP";
  prob.size_label = "n=" + std::to_string(n);

  const Index rank = std::max<Index>(1, n / 2);
  const Mat M = randomNormalDense(n, rank, rng, 1.0 / std::sqrt(static_cast<Scalar>(rank)));
  Mat Pdense = M * M.transpose() + 1e-2 * Mat::Identity(n, n);
  std::vector<Triplet> p_triplets;
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i <= j; ++i) p_triplets.emplace_back(i, j, Pdense(i, j));
  }
  prob.P = SparseMat(n, n);
  prob.P.setFromTriplets(p_triplets.begin(), p_triplets.end());

  // Box constraints x in [-1, 1]: A = [-I; I], b_box handled via the
  // feasible-pair construction below (lb/ub are implicit in s0's range).
  Mat Abox(2 * n, n);
  Abox.topRows(n) = -Mat::Identity(n, n);
  Abox.bottomRows(n) = Mat::Identity(n, n);
  prob.A = SparseMat(2 * n, n);
  auto a_triplets = denseToTriplets(Abox);
  prob.A.setFromTriplets(a_triplets.begin(), a_triplets.end());

  prob.cone_spec.nonneg_dim = 2 * n;

  makeFeasiblePrimalDualData(prob.P, prob.A, prob.cone_spec, rng, prob.q, prob.b);
  return prob;
}

BenchProblem makeRandomLp(Index n, std::mt19937& rng) {
  BenchProblem prob = makeRandomQp(n, rng);
  prob.family = "RandomLP";
  prob.P = SparseMat(n, n);  // zero out P -> pure LP
  makeFeasiblePrimalDualData(prob.P, prob.A, prob.cone_spec, rng, prob.q, prob.b);
  return prob;
}

BenchProblem makeRandomSocp(Index num_blocks, Index block_dim, std::mt19937& rng) {
  BenchProblem prob;
  prob.family = "RandomSOCP";
  prob.size_label = "blocks=" + std::to_string(num_blocks) + ",dim=" + std::to_string(block_dim);

  const Index n = num_blocks * block_dim;
  const Index m = num_blocks * block_dim;
  prob.P = SparseMat(n, n);

  const Mat Adense = randomNormalDense(m, n, rng, 1.0 / std::sqrt(static_cast<Scalar>(n)));
  prob.A = SparseMat(m, n);
  auto a_triplets = denseToTriplets(Adense);
  prob.A.setFromTriplets(a_triplets.begin(), a_triplets.end());

  prob.cone_spec.soc_dims.assign(static_cast<size_t>(num_blocks), block_dim);

  makeFeasiblePrimalDualData(prob.P, prob.A, prob.cone_spec, rng, prob.q, prob.b);
  return prob;
}

BenchProblem makePortfolio(Index num_assets, Index num_factors, std::mt19937& rng) {
  BenchProblem prob;
  prob.family = "Portfolio";
  prob.size_label = "assets=" + std::to_string(num_assets) + ",factors=" + std::to_string(num_factors);

  const Index n = num_assets, k = num_factors;
  const Scalar gamma = 5.0;

  const Vec mu = randomUniform(n, rng, 0.0, 0.15);
  const Mat F = randomNormalDense(n, k, rng, 1.0 / std::sqrt(static_cast<Scalar>(k)));
  const Vec Dsqrt = randomUniform(n, rng, 0.05, 0.2);

  // Variables: x(n), t, s, u, v  -- total n+4. Index helpers:
  const Index nx = n + 4;
  const Index idx_t = n, idx_s = n + 1, idx_u = n + 2, idx_v = n + 3;

  prob.P = SparseMat(nx, nx);  // pure SOCP, no quadratic term needed

  Vec q = Vec::Zero(nx);
  q.head(n) = -mu;
  q[idx_t] = gamma;
  q[idx_s] = gamma;
  prob.q = q;

  std::vector<Triplet> a_triplets;
  std::vector<Scalar> b_entries;
  Index row = 0;

  // Zero cone: 1'x = 1
  for (Index i = 0; i < n; ++i) a_triplets.emplace_back(row, i, 1.0);
  b_entries.push_back(1.0);
  ++row;
  prob.cone_spec.zero_dim = 1;

  // Nonneg: x >= 0  =>  -x + s = 0
  for (Index i = 0; i < n; ++i) {
    a_triplets.emplace_back(row, i, -1.0);
    b_entries.push_back(0.0);
    ++row;
  }
  prob.cone_spec.nonneg_dim = n;

  // SOC (dim n+1): (u, Dsqrt.*x) in Q  =>  -u+s0=0, -Dsqrt_i*x_i+s_i=0
  a_triplets.emplace_back(row, idx_u, -1.0);
  b_entries.push_back(0.0);
  ++row;
  for (Index i = 0; i < n; ++i) {
    a_triplets.emplace_back(row, i, -Dsqrt[i]);
    b_entries.push_back(0.0);
    ++row;
  }
  prob.cone_spec.soc_dims.push_back(n + 1);

  // SOC (dim k+1): (v, F'x) in Q
  a_triplets.emplace_back(row, idx_v, -1.0);
  b_entries.push_back(0.0);
  ++row;
  for (Index j = 0; j < k; ++j) {
    for (Index i = 0; i < n; ++i) {
      if (F(i, j) != 0.0) a_triplets.emplace_back(row, i, -F(i, j));
    }
    b_entries.push_back(0.0);
    ++row;
  }
  prob.cone_spec.soc_dims.push_back(k + 1);

  // SOC (dim 3): (1+t, 1-t, 2u) in Q  =>  s0-t=1, s1+t=1, s2-2u=0
  a_triplets.emplace_back(row, idx_t, -1.0);
  b_entries.push_back(1.0);
  ++row;
  a_triplets.emplace_back(row, idx_t, 1.0);
  b_entries.push_back(1.0);
  ++row;
  a_triplets.emplace_back(row, idx_u, -2.0);
  b_entries.push_back(0.0);
  ++row;
  prob.cone_spec.soc_dims.push_back(3);

  // SOC (dim 3): (1+s, 1-s, 2v) in Q
  a_triplets.emplace_back(row, idx_s, -1.0);
  b_entries.push_back(1.0);
  ++row;
  a_triplets.emplace_back(row, idx_s, 1.0);
  b_entries.push_back(1.0);
  ++row;
  a_triplets.emplace_back(row, idx_v, -2.0);
  b_entries.push_back(0.0);
  ++row;
  prob.cone_spec.soc_dims.push_back(3);

  prob.A = SparseMat(row, nx);
  prob.A.setFromTriplets(a_triplets.begin(), a_triplets.end());
  prob.b = Eigen::Map<Vec>(b_entries.data(), static_cast<Index>(b_entries.size()));

  return prob;
}

BenchProblem makeGroupLasso(Index num_groups, Index group_dim, Index num_measurements,
                            std::mt19937& rng) {
  BenchProblem prob;
  prob.family = "GroupLasso";
  prob.size_label = "groups=" + std::to_string(num_groups) + ",dim=" + std::to_string(group_dim);

  const Index n = num_groups * group_dim;
  const Index nx = n + num_groups;
  const Scalar lambda = 0.1;

  // A "true" sparse-in-groups signal, plus noisy linear measurements.
  Vec x_true = Vec::Zero(n);
  std::uniform_int_distribution<int> active_dist(0, static_cast<int>(num_groups) - 1);
  for (int k = 0; k < std::max<Index>(1, num_groups / 4); ++k) {
    const Index g = active_dist(rng);
    x_true.segment(g * group_dim, group_dim) = randomUniform(group_dim, rng, -2.0, 2.0);
  }
  const Mat Als = randomNormalDense(num_measurements, n, rng, 1.0 / std::sqrt(static_cast<Scalar>(n)));
  const Vec noise = randomUniform(num_measurements, rng, -0.05, 0.05);
  const Vec bls = Als * x_true + noise;

  const Mat AtA = Als.transpose() * Als;
  std::vector<Triplet> p_triplets;
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i <= j; ++i) p_triplets.emplace_back(i, j, 2.0 * AtA(i, j));
  }
  prob.P = SparseMat(nx, nx);
  prob.P.setFromTriplets(p_triplets.begin(), p_triplets.end());

  Vec q = Vec::Zero(nx);
  q.head(n) = -2.0 * (Als.transpose() * bls);
  q.tail(num_groups).setConstant(lambda);
  prob.q = q;

  std::vector<Triplet> a_triplets;
  Index row = 0;
  for (Index g = 0; g < num_groups; ++g) {
    const Index t_idx = n + g;
    a_triplets.emplace_back(row, t_idx, -1.0);
    ++row;
    for (Index j = 0; j < group_dim; ++j) {
      a_triplets.emplace_back(row, g * group_dim + j, -1.0);
      ++row;
    }
    prob.cone_spec.soc_dims.push_back(group_dim + 1);
  }
  prob.A = SparseMat(row, nx);
  prob.A.setFromTriplets(a_triplets.begin(), a_triplets.end());
  prob.b = Vec::Zero(row);

  return prob;
}

BenchProblem makeFrictionChain(Index num_contacts, Scalar mu, std::mt19937& rng) {
  BenchProblem prob;
  prob.family = "FrictionChain";
  prob.size_label = "contacts=" + std::to_string(num_contacts);

  const Index nx = 3 * num_contacts;
  prob.P = SparseMat(nx, nx);
  std::vector<Triplet> p_triplets;
  for (Index i = 0; i < nx; ++i) p_triplets.emplace_back(i, i, 1.0);
  prob.P.setFromTriplets(p_triplets.begin(), p_triplets.end());

  Vec f_des(nx);
  Scalar total_normal = 0.0;
  for (Index c = 0; c < num_contacts; ++c) {
    const Scalar fn = randomUniform(1, rng, 5.0, 15.0)[0];
    const Vec ft = randomUniform(2, rng, -6.0, 6.0);  // may exceed the friction cone
    f_des[3 * c] = fn;
    f_des.segment(3 * c + 1, 2) = ft;
    total_normal += fn;
  }
  prob.q = -f_des;

  std::vector<Triplet> a_triplets;
  Index row = 0;

  // One coupling equality: sum of normal components = target total load,
  // with generous headroom so the instance stays feasible regardless of how
  // much the random tangential components exceed the friction cone.
  for (Index c = 0; c < num_contacts; ++c) a_triplets.emplace_back(row, 3 * c, 1.0);
  ++row;
  prob.cone_spec.zero_dim = 1;

  for (Index c = 0; c < num_contacts; ++c) {
    a_triplets.emplace_back(row, 3 * c, -mu);
    ++row;
    a_triplets.emplace_back(row, 3 * c + 1, -1.0);
    ++row;
    a_triplets.emplace_back(row, 3 * c + 2, -1.0);
    ++row;
    prob.cone_spec.soc_dims.push_back(3);
  }

  prob.A = SparseMat(row, nx);
  prob.A.setFromTriplets(a_triplets.begin(), a_triplets.end());

  Vec b = Vec::Zero(row);
  b[0] = 1.2 * total_normal;
  prob.b = b;

  return prob;
}

BenchProblem makeFrictionChainXL(Index num_contacts, Scalar mu, std::mt19937& rng) {
  BenchProblem prob;
  prob.family = "FrictionChainXL";
  prob.size_label = "contacts=" + std::to_string(num_contacts);

  // Ratio observed on a real per-timestep multibody friction-contact
  // problem: zero_dim=4944, soc_blocks=66 -> ~75 equality rows per contact.
  constexpr Index kEqualityPerContact = 75;
  const Index n = kEqualityPerContact * num_contacts;

  prob.P = SparseMat(n, n);
  std::vector<Triplet> p_triplets;
  for (Index i = 0; i < n; ++i) p_triplets.emplace_back(i, i, 1.0);
  prob.P.setFromTriplets(p_triplets.begin(), p_triplets.end());

  std::vector<Triplet> a_triplets;
  Index row = 0;

  // Banded (tridiagonal-like) equality block over ALL n variables --
  // stands in for a long kinematic/dynamics constraint chain over many
  // bodies. Coefficients are structural, not physically meaningful: the
  // instance's feasibility is guaranteed below via an explicit
  // strictly-feasible primal-dual pair, independent of these values.
  std::uniform_real_distribution<Scalar> band_dist(0.5, 1.5);
  for (Index i = 0; i < n; ++i) {
    a_triplets.emplace_back(row, i, band_dist(rng));
    a_triplets.emplace_back(row, (i + 1) % n, 0.5 * band_dist(rng));
    a_triplets.emplace_back(row, (i + 2) % n, 0.25 * band_dist(rng));
    ++row;
  }
  prob.cone_spec.zero_dim = n;

  // num_contacts independent 3D Coulomb friction cones on the first
  // 3*num_contacts variables, identical in form to makeFrictionChain.
  for (Index c = 0; c < num_contacts; ++c) {
    a_triplets.emplace_back(row, 3 * c, -mu);
    ++row;
    a_triplets.emplace_back(row, 3 * c + 1, -1.0);
    ++row;
    a_triplets.emplace_back(row, 3 * c + 2, -1.0);
    ++row;
    prob.cone_spec.soc_dims.push_back(3);
  }

  prob.A = SparseMat(row, n);
  prob.A.setFromTriplets(a_triplets.begin(), a_triplets.end());

  makeFeasiblePrimalDualData(prob.P, prob.A, prob.cone_spec, rng, prob.q, prob.b);
  return prob;
}

}  // namespace conicxx::bench
