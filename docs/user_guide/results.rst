Solution, status and diagnostics
================================

``solve()`` returns a :cpp:struct:`conicxx::Solution`:

.. code-block:: cpp

   struct Solution {
     Status status;     // see below
     Vec x, s, z;       // primal, slack, dual; unscaled
     Scalar objective;  // 0.5 x'Px + q'x at the reported iterate
     Info info;         // diagnostics
     bool ok() const;   // status == Status::Solved
   };

Status
------

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - Status
     - Meaning and content of ``x``, ``s``, ``z``
   * - ``Solved``
     - All termination criteria hold (:doc:`../algorithm/termination`). ``x, s, z`` is an
       approximate primal-dual solution.
   * - ``PrimalInfeasible``
     - Certificate of primal infeasibility: ``z`` with :math:`A^\top z \approx 0`,
       :math:`z\in\K^*`, normalized to :math:`b^\top z = -1`.
   * - ``DualInfeasible``
     - Certificate of dual infeasibility (unbounded objective): ``x`` with :math:`Px \approx 0`,
       :math:`Ax + s \approx 0`, :math:`s \in \K`, normalized to :math:`q^\top x = -1`.
   * - ``AlmostSolved``
     - The solve stopped early (see the next rows), and the best iterate meets the reduced
       tolerances ``reduced_tol_*``. That iterate is returned.
   * - ``AlmostPrimalInfeasible``, ``AlmostDualInfeasible``
     - As above, for an infeasibility certificate at the reduced tolerances.
   * - ``MaxIterations``
     - ``max_iter`` reached; the best iterate is returned.
   * - ``MaxTime``
     - ``time_limit`` reached; the best iterate is returned.
   * - ``InsufficientProgress``
     - Two consecutive very short steps, or residuals that grow instead of decrease (see
       :ref:`progress-checks`); the best iterate is returned.
   * - ``NumericalError``
     - A KKT factorization or solve produced a zero or non-finite pivot or direction, the step
       length became zero, or ``solve()`` was called without a successful ``setup()``. The best
       iterate (if any) is returned.
   * - ``Unsolved``
     - ``solve()`` has not been called yet.

"Best iterate" is the iterate with the smallest merit value (defined in
:doc:`../algorithm/termination`), not the last one. :cpp:func:`conicxx::toString` returns the name
of a status.

How to react to a status in a simulation loop is application specific. A conservative pattern:

.. code-block:: cpp

   switch (sol.status) {
     case Status::Solved:
     case Status::AlmostSolved:
       accept(sol);                       // AlmostSolved: accurate to ~1e-4
       break;
     case Status::PrimalInfeasible:       // e.g. inconsistent constraints
     case Status::DualInfeasible:         // e.g. a body without mass and without constraints
       report_model_error(sol);
       break;
     default:                             // iteration/time limit, numerical trouble
       log(sol.info);
       fall_back(sol);
   }

Info
----

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``iterations``
     - Iterations of this ``solve()``. The last iteration only evaluates the termination criteria.
   * - ``primal_residual``, ``dual_residual``
     - Relative primal and dual residuals of the reported iterate (unscaled, infinity norm).
   * - ``duality_gap``
     - Absolute duality gap of the reported iterate.
   * - ``mu``
     - Complementarity measure :math:`\mu` of the reported iterate.
   * - ``merit``
     - Merit value of the reported iterate: the maximum of the residuals, the smaller of the
       absolute and relative gap, and the complementarity residual.
   * - ``setup_time_s``, ``solve_time_s``
     - Wall-clock time of the last ``setup()`` and of this ``solve()``.
   * - ``kkt_factorizations``
     - Numeric :math:`LDL^\top` factorizations in this ``solve()``.
   * - ``kkt_dynamic_reg_pivots``
     - Pivots replaced by dynamic regularization, summed over all factorizations. Nonzero values
       indicate a (nearly) singular KKT matrix.
   * - ``kkt_refinement_steps``
     - Iterative-refinement steps, summed over all KKT solves.
   * - ``kkt_refinement_residual``
     - :math:`\|\text{rhs} - K\,\text{sol}\|_\infty / \|\text{rhs}\|_\infty` of the last KKT solve,
       against the exact (unregularized) KKT matrix.
   * - ``equality_rank_deficient``
     - Some KKT solve left an equality row unsatisfied (relative residual above
       :math:`10^{-8}`): the equality constraints are inconsistent or numerically dependent.
       Consistent duplicated equalities are solved normally and do not set this flag.
   * - ``warm_start_used``
     - The solve started from the warm start rather than the cold start.
   * - ``warm_start_merit``, ``cold_start_merit``
     - Merit values of the warm starting point and of the cold start it was compared with; 0 if
       not evaluated.

Interpreting accuracy
---------------------

``Solved`` guarantees the tolerances on residuals and gap, not on the distance to the exact
solution. On problems whose solution is not strictly complementary (in contact problems: a contact
that touches without force, so both the gap :math:`s_i` and the force :math:`z_i` vanish), an
interior-point iterate is accurate only to about :math:`\sqrt{\mu}` in the affected components.
The ``merit`` includes a complementarity residual (the natural map
:math:`z - \proj_\K(z - s)`) that measures exactly this, and the progress checks keep iterating
while it improves.
