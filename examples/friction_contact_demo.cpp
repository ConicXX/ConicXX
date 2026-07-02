// Illustrates the intended usage pattern for ConicXX inside a per-timestep
// nonsmooth multibody simulation loop: one contact with Coulomb friction.
//
// We regularize a desired contact impulse f_des towards the feasible
// friction cone { (fn, ft1, ft2) : mu*fn >= ||(ft1,ft2)|| }, i.e.
//
//   min  0.5||f - f_des||^2
//   s.t. mu*fn = y0, ft1 = y1, ft2 = y2,   y in Q^3
//
// f_des changes every "timestep" (mimicking a changing contact force target
// from the dynamics solver) while the sparsity pattern of P/A stays fixed,
// so after the initial setup() every timestep only needs a cheap
// updateData() + solve().

#include <conicxx/solver.h>

#include <cstdio>

int main() {
  using namespace conicxx;

  const Scalar mu = 0.5;  // friction coefficient

  // Variables: f = (fn, ft1, ft2). Constraint: -diag(mu,1,1)*f + s = 0, s in Q^3.
  SparseMat A(3, 3);
  {
    std::vector<Triplet> triplets = {{0, 0, -mu}, {1, 1, -1.0}, {2, 2, -1.0}};
    A.setFromTriplets(triplets.begin(), triplets.end());
  }
  Vec b = Vec::Zero(3);

  SparseMat P(3, 3);
  {
    std::vector<Triplet> triplets = {{0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}};
    P.setFromTriplets(triplets.begin(), triplets.end());
  }

  ConeSpec cone_spec;
  cone_spec.soc_dims = {3};

  Settings settings;
  settings.warm_start = true;

  Solver solver;
  Vec f_des(3);
  f_des << 5.0, 1.0, 1.0;  // first "timestep": already inside the cone
  Vec q = -f_des;

  if (!solver.setup(P, q, A, b, cone_spec, settings)) {
    std::fprintf(stderr, "setup() failed\n");
    return 1;
  }

  const int num_timesteps = 4;
  for (int t = 0; t < num_timesteps; ++t) {
    // A changing desired contact force, e.g. from the dynamics solver.
    f_des << 5.0 - t, 1.0 + 2.0 * t, 1.0 + 0.5 * t;
    q = -f_des;

    if (!solver.updateData(nullptr, &q, nullptr, nullptr)) {
      // Sparsity pattern changed (not expected here since q is dense and
      // always present) -- would need a full setup() in that case.
      std::fprintf(stderr, "updateData() rejected at timestep %d\n", t);
      return 1;
    }

    const Solution& sol = solver.solve();
    std::printf("timestep %d: status=%s f_des=(%.2f,%.2f,%.2f) f=(%.4f,%.4f,%.4f) iters=%d\n", t,
                toString(sol.status), f_des[0], f_des[1], f_des[2], sol.x[0], sol.x[1], sol.x[2],
                sol.info.iterations);
  }

  return 0;
}
