Benchmarks and testing
======================

Benchmark suite
---------------

``benchmarks/`` contains a deterministic benchmark suite, ``conicxx_benchmarks``, built with
``-DCONICXX_BUILD_BENCHMARKS=ON``. Every instance is generated from a fixed seed and from an
explicitly constructed strictly feasible primal-dual pair, so it is guaranteed to be solvable and
its iteration count is exactly reproducible.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Family
     - Description
   * - ``RandomQP``, ``RandomLP``
     - Random QPs with a dense positive definite :math:`P` and box constraints, and random LPs.
   * - ``RandomSOCP``
     - Random pure SOCPs with several independent second-order cone blocks.
   * - ``Portfolio``
     - Long-only mean-variance portfolio optimization in the SOCP form of
       :cite:t:`domahidi2013ecos` (eq. 17), with one large cone block of dimension
       ``num_assets + 1``.
   * - ``GroupLasso``
     - :math:`\min \|Ax - b\|^2 + \lambda\sum_i\|x_i\|_2`, following the group-lasso benchmark of
       :cite:t:`chari2025qoco`; many SOC blocks of moderate size.
   * - ``FrictionChain``
     - Independent 3D Coulomb friction cones with a QP regularization and one coupling equality.
   * - ``FrictionChainXL``
     - Friction cones embedded in a banded chain of 75 equality rows per contact, sized after a
       real multibody time step; stresses the factorization at scale.
   * - ``ApexContacts``
     - Half of the contacts with the exact solution at the cone apex (degenerate, not strictly
       complementary), half strictly inside the cone, in one KKT system.

It also runs warm-started sequences of 30 slightly perturbed problems (the pattern of a
simulation) and reports iterations and factorizations for cold and warm starts.

.. code-block:: sh

   ./build/benchmarks/conicxx_benchmarks          # table: family, size, status, iterations, time, residuals
   ctest --test-dir build -L benchmark            # pass/fail gate: every instance must be Solved

Because the instances are deterministic, a change of an iteration count between two versions is
a real algorithmic difference, not noise. The development workflow compares the printed table
before and after every change to the solver.

Tests
-----

The GoogleTest suite in ``test/`` covers the cone kernels (NT scaling identities, step lengths on
the boundary and at the apex, scale invariance down to :math:`10^{-12}`), the sparsity maps and the :math:`LDL^\top` factorization
(including inertia, dynamic regularization and supernodal against simplicial), and the solver on
LPs, QPs, SOCPs, friction and apex-contact problems, infeasible and unbounded problems, redundant
and inconsistent equalities, compliant rows, warm starts, repeated ``updateData()`` calls, dense
input, and unscaled termination on badly scaled problems. One test checks with allocation hooks
that ``solve()`` does not allocate after the first call; it runs in the AddressSanitizer build.

.. code-block:: sh

   ctest --test-dir build -LE benchmark --output-on-failure

Continuous integration builds and tests with GCC and Clang in Debug and Release, runs the tests
under AddressSanitizer and UndefinedBehaviorSanitizer, runs the benchmark gate, and checks that an
installed package can be consumed with ``find_package``.
