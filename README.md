# ConicXX

**ConicXX** is a modern C++ solver for quadratic conic optimization problems.

It implements a primal-dual interior-point method with:
- Nesterov–Todd scaling
- Mehrotra predictor–corrector
- Support for nonnegative and second-order cones
- Symmetric KKT systems with regularization

Built on Eigen, ConicXX is designed for research, prototyping, and high-performance applications,
including as an embedded solver inside a per-timestep nonsmooth multibody simulation loop
(`setup()` once, then cheap `updateData()` + `solve()` calls that reuse the KKT sparsity pattern
and factorization symbolic analysis across timesteps).

Supported convex sets: the zero cone (equality constraints), the nonnegative orthant, and
second-order (Lorentz) cones -- no exponential/power/semidefinite cones.

## Building

Requires a C++17 compiler and CMake >= 3.21. Eigen (3.4.0) and, for tests, GoogleTest are located
via `find_package` and transparently fetched with `FetchContent` if not found locally.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Useful options: `CONICXX_BUILD_SHARED`, `CONICXX_BUILD_TESTS`, `CONICXX_BUILD_EXAMPLES`,
`CONICXX_BUILD_BENCHMARKS`, `CONICXX_WARNINGS_AS_ERRORS`, `CONICXX_INSTALL`. See
`examples/friction_contact_demo.cpp` for a minimal usage example (a QP-regularized contact force
projected onto a Coulomb friction cone, solved once per "timestep" via `Solver::updateData`).

Consume from another CMake project via `find_package(conicxx CONFIG REQUIRED)`,
`add_subdirectory`, or `FetchContent` -- all three expose the same `conicxx::conicxx` target.

## Benchmarks

`benchmarks/` contains a small, deterministic performance suite (`conicxx_benchmarks`, also
registered as a `ctest` under the `benchmark` label) covering problems drawn from the literature --
the ECOS long-only portfolio-optimization SOCP (Domahidi/Chu/Boyd, eq. 17) and the QOCO group-lasso
benchmark (Chari et al., sec. C.3) -- plus random QP/LP/SOCP instances and a domain-specific
multi-contact Coulomb-friction QP+SOC problem. Each instance is built from a fixed seed and from an
explicit strictly-feasible primal-dual pair, so it is guaranteed solvable and its reported iteration
count is exactly reproducible.

```sh
./build/benchmarks/conicxx_benchmarks     # prints a QOCO/Clarabel-style table:
                                           #   Family | Size | Trial | Status | Iters | Time | residuals
ctest --test-dir build -L benchmark       # same, as a pass/fail regression gate (nonzero exit
                                           # if any instance stops converging)
ctest --test-dir build -LE benchmark      # run everything except the benchmark suite
```

To check whether a change affected solver behavior, run the suite before and after and compare the
printed iteration counts/residuals (e.g. redirect to a file and `diff`) -- a shift there reflects a
real algorithmic difference, not sampling noise.

## Literature

- Clarabel: https://arxiv.org/abs/2405.12762
- QOCO: https://arxiv.org/abs/2503.12658
- IPM slides with self-dual homogeneous embedding (as in Clarabel): https://www.syscop.de/files/2015ss/numopt/TEMPO_NOC_ECOS.pdf