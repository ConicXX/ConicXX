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

## Problem formulation

ConicXX solves problems of the form

```
minimize    (1/2) x'Px + q'x
subject to  Ax + s = b
            s in K = Zero^p x R^l_+ x Q^(k_1) x ... x Q^(k_N)
```

| Symbol | Type | Meaning |
|---|---|---|
| `x` | `Vec`, size `n` | decision variable -- what you get back in `Solution::x` |
| `P` | `SparseMat`, `n x n` | quadratic cost; symmetric PSD, **only its upper triangle is read** |
| `q` | `Vec`, size `n` | linear cost |
| `A` | `SparseMat`, `m x n` | constraint matrix |
| `b` | `Vec`, size `m` | constraint right-hand side |
| `s` | `Vec`, size `m` | slack variable, constrained to lie in `K` (`Solution::s`) |
| `z` | `Vec`, size `m` | dual variable/Lagrange multiplier, lies in the dual cone `K*` (`Solution::z`) |

If `P = 0` the problem is a linear/conic program (LP/SOCP); a nonzero `P` makes it a QP/QCQP-style
problem with conic constraints.

### The cone product `K`

`K` is a Cartesian product of blocks, **always in this fixed order** (this is what `ConeSpec`
encodes):

1. **Zero cone**, `{0}^p` (`ConeSpec::zero_dim`) -- forces `s_i = 0` for these `p` rows, i.e. encodes
   `p` linear *equality* constraints `A_i x = b_i`. The corresponding `z_i` is an unconstrained
   (free) Lagrange multiplier, exactly like the equality multiplier in a KKT system.
2. **Nonnegative orthant**, `R^l_+` (`ConeSpec::nonneg_dim`) -- forces `s_i >= 0`, i.e. encodes
   `l` linear *inequality* constraints `A_i x <= b_i`.
3. **Second-order (Lorentz) cones**, `Q^(k) = { (u0,u1) in R x R^(k-1) : u0 >= ||u1||_2 }`
   (`ConeSpec::soc_dims`, one entry per block) -- encodes convex quadratic/norm constraints such as
   friction cones (`mu*f_n >= ||f_t||_2`) or bounding a linear expression's Euclidean norm.

### Encoding common constraints

Every constraint reduces to picking a row (or a few rows) of `A` and `b` such that
`s = b - Ax` is the quantity you want confined to a cone:

| You want | Cone block | Row(s) of `A`, `b` |
|---|---|---|
| `a'x = c` | Zero | `A_row = a`, `b_row = c` |
| `a'x <= c` | Nonnegative | `A_row = a`, `b_row = c` |
| `a'x >= c` | Nonnegative | `A_row = -a`, `b_row = -c` |
| `lb <= x_i <= ub` | Nonnegative (2 rows) | `-e_i, -lb` and `e_i, ub` |
| `\|\|Fx+g\|\|_2 <= c'x+d` | SecondOrder, dim `k+1` | rows `[-c'; -F]`, rhs `[d; g]` (so `s=(c'x+d,\ Fx+g)`) |

### Example: all three cone types together

Writing `P` and `A` as plain dense `Eigen::MatrixXd` first makes the row-by-row structure easy to
see; a `conicxx::SparseMat` for the actual solver call is then one line away.

```cpp
#include <conicxx/solver.h>

using namespace conicxx;

// Variables x = (x0, x1, x2). Minimize 0.5*||x - c||^2 with target c = (1, 1, 0.3), subject to
//   x0 + x1 = 1                (equality    -> Zero cone,        dim 1)
//   x2 >= 0                    (inequality  -> Nonnegative cone, dim 1)
//   (x2, x0, x1) in Q^3        (x2 >= sqrt(x0^2+x1^2)            -> SecondOrder, dim 3)
//
// Cone rows MUST appear in the fixed order Zero -> Nonnegative -> SecondOrder.
// The target's x2=0.3 deliberately violates the SOC bound (sqrt(0.5^2+0.5^2)=0.707 once the
// equality is respected), so this constraint is actually active at the solution.

Mat P(3, 3);  // 0.5*x'*I*x - c'*x  =>  P = I
P << 1, 0, 0,
     0, 1, 0,
     0, 0, 1;
Vec q(3);
q << -1.0, -1.0, -0.3;  // q = -c

Mat A(5, 3);            //                x0    x1    x2
A << /* row 0 (Zero):        x0 + x1        + s0 = 1 */    1,    1,    0,
     /* row 1 (Nonnegative):      -x2        + s1 = 0 */    0,    0,   -1,
     /* row 2 (SecondOrder):      -x2        + s2 = 0 */    0,    0,   -1,
     /* row 3 (SecondOrder): -x0             + s3 = 0 */   -1,    0,    0,
     /* row 4 (SecondOrder):      -x1        + s4 = 0 */    0,   -1,    0;
// (s2, s3, s4) = (x2, x0, x1), which is constrained to lie in Q^3.
Vec b(5);
b << 1.0, 0.0, 0.0, 0.0, 0.0;

ConeSpec cone_spec;
cone_spec.zero_dim = 1;
cone_spec.nonneg_dim = 1;
cone_spec.soc_dims = {3};

// Solver::setup() has overloads accepting P and/or A as dense matrices directly (handy while
// prototyping, as here) -- each one is converted to conicxx::SparseMat internally, with a warning
// since dense input carries no sparsity to exploit. For a problem solved repeatedly with changing
// data (the per-timestep use case this library targets), build the SparseMat once yourself --
// e.g. `SparseMat A_sparse = toSparse(A);` / `SparseMat P_sparse = toSparseUpperTriangular(P);`
// -- and reuse it across `Solver::updateData()` calls instead.
Solver solver;
solver.setup(P, q, A, b, cone_spec);
const Solution& sol = solver.solve();  // sol.x, sol.s, sol.z, sol.status, sol.info.iterations
// sol.x == (0.5, 0.5, 0.70710678...): the equality forces x0=x1=0.5 by symmetry, and the SOC
// constraint pulls x2 up from the target 0.3 to exactly sqrt(0.5^2+0.5^2) -- it's active.
```

See `examples/friction_contact_demo.cpp` for a second full worked example (a QP + single SOC
friction cone, built with sparse matrices directly and solved repeatedly via `Solver::updateData`
as in a simulation loop).

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

ConicXX follows [semantic versioning](https://semver.org/); the single source of truth is the
`project(conicxx VERSION ...)` call in the top-level `CMakeLists.txt`. `find_package(conicxx
1.2.3 CONFIG REQUIRED)` version-checks against it, and `<conicxx/version.h>` exposes it to C++
code as `CONICXX_VERSION_{MAJOR,MINOR,PATCH}` / `CONICXX_VERSION_STRING` and `conicxx::version()`.

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