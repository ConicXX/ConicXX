Algorithm overview
==================

ConicXX is a primal-dual interior-point method (IPM) for the problem pair :eq:`eq-primal` /
:eq:`eq-dual`. Its ingredients are standard and published; this part of the documentation
describes each of them as implemented, including the choices where ConicXX differs from the
solvers it is modeled on (ECOS :cite:p:`domahidi2013ecos`, Clarabel :cite:p:`goulart2024clarabel`,
QOCO :cite:p:`chari2025qoco`; the cone formulas follow CVXOPT :cite:p:`vandenberghe2010cvxopt`).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Choice
   * - Formulation
     - Homogeneous self-dual embedding with quadratic objective (:doc:`hsde`): one method for
       solvable, infeasible and unbounded problems, no phase I.
   * - Cones
     - Zero, nonnegative orthant, second-order; Nesterov--Todd scaling, closed-form and
       cancellation-free (:doc:`cones`).
   * - Search direction
     - Mehrotra predictor--corrector with the second-order correction of symmetric cones
       (:doc:`ipm`).
   * - Step length
     - Fraction to the boundary, then backtracking on a centrality condition evaluated exactly at
       the trial point (:doc:`ipm`).
   * - Linear algebra
     - One quasidefinite KKT matrix per iteration, AMD ordering, simplicial or supernodal
       :math:`LDL^\top` with expected pivot signs, static and dynamic regularization, iterative
       refinement against the exact matrix (:doc:`linear_system`).
   * - Scaling
     - Ruiz equilibration with cost scaling, cone-preserving (:doc:`equilibration`).
   * - Termination
     - Unscaled relative residuals and duality gap; infeasibility certificates; best-iterate
       tracking with a merit function that includes complementarity (:doc:`termination`).
   * - Warm start
     - Blend of the previous solution with the standard starting point (:doc:`warm_start`).

The iteration
-------------

With :math:`\nu` the degree of the cone (:doc:`cones`) and :math:`(x, s, z, \tau, \kappa)` the
iterate of the embedding:

.. code-block:: text

   setup():  equilibrate data, build KKT pattern, AMD ordering, symbolic factorization

   solve():
     choose starting point (warm start if its merit is not worse than the cold start's)
     loop k = 0, 1, ...:
       compute residuals r_x, r_z, r_tau and mu = (s'z + tau kappa) / (nu + 1)
       check termination (solved, infeasible, progress, limits); track best iterate
       NT scaling W of (s, z); factorize K = [P  A'; A  -W'W]                 (1 factorization)
       solve K for the constant rhs (-q, b) and the affine rhs, together      (1 paired solve)
       eliminate dtau, affine direction, affine step length alpha_aff
       sigma = (mu_aff / mu)^3
       solve K for the combined (corrector) rhs                                 (1 solve)
       alpha = 0.99 * step to boundary, then backtrack until centred
       (x, s, z, tau, kappa) += alpha * direction; rescale if tau, kappa extreme
     unscale and report solution or certificate (best iterate on early stop)

Cost per iteration
------------------

Each iteration costs one numeric factorization of the KKT matrix and three triangular solves
(two of them in one pass over the factor), plus iterative refinement and :math:`O(\text{nnz})`
vector work. The final iteration only checks termination, so a solve with :math:`k` iterations
needs :math:`k` factorizations from a cold start and :math:`k-1` from an accepted warm start. The
symbolic analysis is done once per sparsity pattern, in ``setup()``.

Which part dominates depends on the problem. On sparse contact-like KKT matrices the triangular
solves and refinement residuals cost about as much as the factorization; on denser problems (QPs
with a dense :math:`P`, large SOC blocks) the factorization dominates.

Notation
--------

.. list-table::
   :widths: 25 75

   * - :math:`n,\ m`
     - Number of variables and of constraint rows.
   * - :math:`p,\ l`
     - Dimensions of the zero cone and of the nonnegative orthant.
   * - :math:`\nu`
     - Degree of :math:`\K`: :math:`l` plus the number of second-order cone blocks.
   * - :math:`e`
     - Identity element of :math:`\K` (ones on the orthant, :math:`(1, 0, \dots, 0)` per SOC
       block, zero on the zero cone).
   * - :math:`u \circ v`
     - Jordan product (:doc:`cones`).
   * - :math:`W,\ H = W^\top W,\ \lambda`
     - Nesterov--Todd scaling, its square, and the scaled point :math:`\lambda = W z = W^{-\top} s`.
   * - :math:`\mu`
     - Complementarity measure :math:`(s^\top z + \tau\kappa)/(\nu+1)`.
   * - :math:`\hat{\cdot}`
     - Equilibrated quantities (:doc:`equilibration`); the algorithm chapters omit the hat.
