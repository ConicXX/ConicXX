#pragma once

#include <random>
#include <string>

#include <conicxx/cone_spec.h>
#include <conicxx/types.h>

namespace conicxx::bench {

/// A self-contained problem instance plus a human-readable size label used
/// for reporting (e.g. "n=100" or "assets=60,factors=10").
struct BenchProblem {
  std::string family;      // e.g. "RandomQP"
  std::string size_label;  // e.g. "n=100"
  SparseMat P, A;
  Vec q, b;
  ConeSpec cone_spec;
};

/// Deterministic RNG so benchmark instances (and therefore reported
/// iteration counts) are exactly reproducible across runs -- this is what
/// makes the suite useful as a regression check: if a future change alters
/// the iteration count on one of these fixed instances, that's a real
/// algorithmic difference, not sampling noise.
std::mt19937 makeRng(const std::string& family, Index size, int trial);

/// Random box-constrained QP: min 0.5x'Px+q'x s.t. lb<=x<=ub.
/// P, q, and the box are randomized but (q, b) are always constructed from
/// an explicit strictly-feasible primal-dual pair, so the instance is
/// guaranteed solvable (see makeFeasiblePrimalDualData in the .cpp).
BenchProblem makeRandomQp(Index n, std::mt19937& rng);

/// Random LP (same box-constrained structure as makeRandomQp, but P = 0).
BenchProblem makeRandomLp(Index n, std::mt19937& rng);

/// Random pure SOCP (P = 0): several independent second-order-cone blocks.
BenchProblem makeRandomSocp(Index num_blocks, Index block_dim, std::mt19937& rng);

/// Long-only mean-variance portfolio optimization, formulated exactly as
/// in Domahidi/Chu/Boyd, "ECOS: An SOCP Solver for Embedded Systems"
/// (2013 ECC), eq. (17): a factor-model covariance is turned into two
/// second-order-cone constraints plus two rotated-cone-as-SOC constraints
/// (rather than passed to the solver as a quadratic P), so this benchmark
/// specifically stresses the second-order cone code path (including one
/// cone block of dimension num_assets+1, the largest single cone block in
/// the whole suite).
BenchProblem makePortfolio(Index num_assets, Index num_factors, std::mt19937& rng);

/// Group lasso: min ||Ax-b||^2 + lambda*sum_i ||x_i||_2, following the
/// group-lasso benchmark family described in Chari/Kandala/... "QOCO"
/// (arXiv:2503.12658), sec. C.3. Stresses many independent SOC blocks of
/// moderate size.
BenchProblem makeGroupLasso(Index num_groups, Index group_dim, Index num_measurements,
                            std::mt19937& rng);

/// Domain-specific benchmark for the target application: num_contacts
/// independent 3D Coulomb friction cones (QP regularization towards a
/// desired contact force, one shared equality row coupling the normal
/// components to a target total support load), mirroring the sparsity
/// structure (many small SOC blocks + light coupling) expected from a
/// nonsmooth multibody contact solve.
BenchProblem makeFrictionChain(Index num_contacts, Scalar mu, std::mt19937& rng);

/// Arbitrary-size scaling benchmark, sized after a real per-timestep
/// multibody friction-contact problem reported against this solver
/// (zero=4944, soc_blocks=66, an equality-to-friction-cone ratio of
/// ~75:1). `num_contacts` is the single size knob: num_contacts 3D Coulomb
/// friction cones (as in makeFrictionChain) are embedded in a banded chain
/// of `75*num_contacts` equality constraints -- standing in for a long
/// kinematic/dynamics constraint chain over many bodies -- instead of the
/// single coupling row makeFrictionChain uses. This is what stresses KKT
/// factorization at the scale where symbolic-analysis cost matters, and can
/// be driven arbitrarily large by increasing num_contacts.
BenchProblem makeFrictionChainXL(Index num_contacts, Scalar mu, std::mt19937& rng);

}  // namespace conicxx::bench
