// Phase 5 accept criterion (CONICXX_AGENT_TASKS.md T5.3): a FrictionChain-style sequence of
// per-timestep problems (q and b perturbed by a small percentage each step, as a per-timestep
// multibody simulation loop would produce) -- warm-started solves must need no more iterations
// than cold ones on average for small (<=1%) perturbations, and never more than cold+2 on any
// single step.

#include "conicxx/solver.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>

#include "test_helpers.h"

using namespace conicxx;

namespace {

// One 3D Coulomb friction cone per contact (fn, ft1, ft2), mu*fn >= ||ft||, plus one coupling
// equality row (sum of normal forces == target_load) -- the same structure as
// benchmarks/problem_generators.cpp's makeFrictionChain(), kept self-contained here rather than
// linking the benchmarks target from a unit test.
struct FrictionChainProblem {
  SparseMat P, A;
  ConeSpec spec;
  Index num_contacts = 0;
};

FrictionChainProblem makeFrictionChainStructure(Index num_contacts, Scalar mu) {
  FrictionChainProblem prob;
  prob.num_contacts = num_contacts;
  const Index nx = 3 * num_contacts;

  std::vector<Triplet> p_triplets;
  for (Index i = 0; i < nx; ++i) p_triplets.emplace_back(i, i, 1.0);
  prob.P = testutil::makeSparse(nx, nx, p_triplets);

  std::vector<Triplet> a_triplets;
  Index row = 0;
  for (Index c = 0; c < num_contacts; ++c) a_triplets.emplace_back(row, 3 * c, 1.0);
  ++row;
  prob.spec.zero_dim = 1;

  for (Index c = 0; c < num_contacts; ++c) {
    a_triplets.emplace_back(row, 3 * c, -mu);
    ++row;
    a_triplets.emplace_back(row, 3 * c + 1, -1.0);
    ++row;
    a_triplets.emplace_back(row, 3 * c + 2, -1.0);
    ++row;
    prob.spec.soc_dims.push_back(3);
  }
  prob.A = testutil::makeSparse(row, nx, a_triplets);
  return prob;
}

// (q, b) for a given desired-force vector f_des and total target load -- q = -f_des, b = (target
// load, 0, 0, ..., 0) matching the coupling row followed by the (always-zero-RHS) SOC rows.
void makeQb(const FrictionChainProblem& prob, const Vec& f_des, Scalar target_load, Vec& q,
           Vec& b) {
  q = -f_des;
  b = Vec::Zero(prob.A.rows());
  b[0] = target_load;
}

// Runs `num_steps` timesteps of f_des/target_load perturbed by up to +/-perturbation_frac each
// step, returning the iteration count at each step for a fresh (cold) Solver and a warm-started
// (reused, updateData()-only) Solver respectively.
struct StepIterations {
  std::vector<int> cold, warm;
};

StepIterations runSequence(Index num_contacts, Scalar mu, Scalar perturbation_frac,
                           int num_steps, unsigned seed) {
  StepIterations result;
  FrictionChainProblem prob = makeFrictionChainStructure(num_contacts, mu);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<Scalar> pert(-perturbation_frac, perturbation_frac);

  Vec f_des = Vec::Zero(3 * num_contacts);
  for (Index c = 0; c < num_contacts; ++c) f_des[3 * c] = 10.0;  // start with pure normal load
  Scalar target_load = 10.0 * static_cast<Scalar>(num_contacts);

  Solver warm_solver;
  bool warm_setup = false;

  for (int step = 0; step < num_steps; ++step) {
    // Perturb every component by up to perturbation_frac (relative), same perturbation applied
    // to both the cold and warm problem instances this step so they solve the identical problem.
    for (Index i = 0; i < f_des.size(); ++i) f_des[i] *= (1.0 + pert(rng));
    target_load *= (1.0 + pert(rng));

    Vec q, b;
    makeQb(prob, f_des, target_load, q, b);

    Solver cold_solver;
    if (!cold_solver.setup(prob.P, q, prob.A, b, prob.spec)) {
      ADD_FAILURE() << "cold step " << step << ": setup() failed";
      return result;
    }
    const Solution& cold_sol = cold_solver.solve();
    if (!cold_sol.ok()) {
      ADD_FAILURE() << "cold step " << step << " status=" << toString(cold_sol.status);
      return result;
    }
    result.cold.push_back(cold_sol.info.iterations);

    if (!warm_setup) {
      if (!warm_solver.setup(prob.P, q, prob.A, b, prob.spec)) {
        ADD_FAILURE() << "warm step " << step << ": setup() failed";
        return result;
      }
      warm_setup = true;
    } else {
      if (!warm_solver.updateData(nullptr, &q, nullptr, &b)) {
        ADD_FAILURE() << "warm step " << step << ": updateData() failed";
        return result;
      }
    }
    const Solution& warm_sol = warm_solver.solve();
    if (!warm_sol.ok()) {
      ADD_FAILURE() << "warm step " << step << " status=" << toString(warm_sol.status);
      return result;
    }
    result.warm.push_back(warm_sol.info.iterations);
  }
  return result;
}

}  // namespace

TEST(WarmStart, SmallPerturbationsNeverExceedColdByMoreThanTwo) {
  const StepIterations r = runSequence(/*num_contacts=*/8, /*mu=*/0.7,
                                       /*perturbation_frac=*/0.01, /*num_steps=*/50, /*seed=*/7);
  ASSERT_EQ(r.cold.size(), r.warm.size());
  for (size_t i = 0; i < r.cold.size(); ++i) {
    EXPECT_LE(r.warm[i], r.cold[i] + 2) << "step " << i << ": warm=" << r.warm[i]
                                        << " cold=" << r.cold[i];
  }
}

TEST(WarmStart, SmallPerturbationsWarmNotWorseThanColdOnAverage) {
  // <=1% perturbation per step, matching the accept criterion exactly.
  const StepIterations r = runSequence(/*num_contacts=*/8, /*mu=*/0.7,
                                       /*perturbation_frac=*/0.01, /*num_steps=*/50, /*seed=*/7);
  const double mean_cold =
      std::accumulate(r.cold.begin(), r.cold.end(), 0.0) / static_cast<double>(r.cold.size());
  const double mean_warm =
      std::accumulate(r.warm.begin(), r.warm.end(), 0.0) / static_cast<double>(r.warm.size());
  EXPECT_LE(mean_warm, mean_cold) << "mean_warm=" << mean_warm << " mean_cold=" << mean_cold;
}

TEST(WarmStart, LargerPerturbationsStillConvergeAndNeverBlowUp) {
  // Up to 5% per step (the upper end of T5.3's stated range) -- not required to beat cold on
  // average (only the <=1% case is), but must still converge every step and stay within the
  // cold+2 bound (warm start must never actively hurt, even when it doesn't help).
  const StepIterations r = runSequence(/*num_contacts=*/8, /*mu=*/0.7,
                                       /*perturbation_frac=*/0.05, /*num_steps=*/50, /*seed=*/11);
  ASSERT_EQ(r.cold.size(), r.warm.size());
  for (size_t i = 0; i < r.cold.size(); ++i) {
    EXPECT_LE(r.warm[i], r.cold[i] + 2) << "step " << i << ": warm=" << r.warm[i]
                                        << " cold=" << r.cold[i];
  }
}
