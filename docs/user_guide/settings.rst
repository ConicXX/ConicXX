Settings
========

:cpp:struct:`conicxx::Settings` is a plain aggregate; set the fields you want to change and pass
it to ``setup()`` or ``setSettings()``:

.. code-block:: cpp

   conicxx::Settings settings;
   settings.tol_feas = 1e-6;
   settings.max_iter = 50;
   settings.verbose  = 1;
   solver.setup(P, q, A, b, cones, settings);

Fields marked *setup* are read only by ``setup()``; all others take effect at the next ``solve()``.
The defaults are chosen for general problems and for the frictional contact problems ConicXX was
built for; most applications only change the tolerances, ``max_iter``, ``time_limit`` and
``verbose``.

Termination
-----------

All criteria are evaluated on the unscaled problem; see :doc:`../algorithm/termination` for the
exact definitions.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Field
     - Default
     - Meaning
   * - ``tol_feas``
     - 1e-8
     - Relative primal and dual residual tolerance (infinity norm).
   * - ``tol_gap_abs``
     - 1e-8
     - Absolute duality gap tolerance. Convergence requires the absolute **or** the relative gap
       to be below its tolerance.
   * - ``tol_gap_rel``
     - 1e-8
     - Relative duality gap tolerance.
   * - ``tol_ktratio``
     - 1e-6
     - Infeasibility certificates are tested only once :math:`\kappa/\tau > 1000/`\ ``tol_ktratio``.
       ``Solved`` requires :math:`\kappa/\tau \le 1`.
   * - ``tol_infeas_abs``
     - 1e-7
     - Absolute part of the infeasibility certificate test.
   * - ``tol_infeas_rel``
     - 1e-7
     - Relative part of the certificate test (multiplies :math:`-b^\top z` or :math:`-q^\top x`).
   * - ``max_iter``
     - 200
     - Iteration limit, ``Status::MaxIterations``.
   * - ``time_limit``
     - :math:`\infty`
     - Wall-clock limit of one ``solve()`` in seconds, ``Status::MaxTime``.
   * - ``reduced_tol_feas``
     - 1e-4
     - Reduced tolerances: when a solve stops early (iteration or time limit, insufficient
       progress, numerical error), the best iterate is reported as ``AlmostSolved`` if it meets
       these.
   * - ``reduced_tol_gap_abs``
     - 5e-5
     -
   * - ``reduced_tol_gap_rel``
     - 5e-5
     -
   * - ``reduced_tol_infeas_abs``
     - 5e-5
     - Reduced tolerances for ``AlmostPrimalInfeasible`` / ``AlmostDualInfeasible``.
   * - ``reduced_tol_infeas_rel``
     - 5e-5
     -
   * - ``reduced_tol_ktratio``
     - 1e-4
     -

Step length
-----------

See :doc:`../algorithm/ipm`.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Field
     - Default
     - Meaning
   * - ``max_step_fraction``
     - 0.99
     - Fraction of the step to the cone boundary that is taken.
   * - ``centrality_theta``
     - 1e-4
     - Centrality safeguard :math:`\theta`: a step is accepted only if every complementarity pair
       of the trial point has centrality :math:`\ge \theta\mu`.
   * - ``linesearch_backtrack``
     - 0.8
     - Backtracking factor of the centrality safeguard.
   * - ``min_terminate_step_length``
     - 1e-4
     - Two consecutive steps shorter than this end the solve with ``InsufficientProgress``.

Equilibration
-------------

See :doc:`../algorithm/equilibration`.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Field
     - Default
     - Meaning
   * - ``equilibrate``
     - true
     - *setup.* Ruiz equilibration of the data.
   * - ``equilibrate_max_iter``
     - 10
     - *setup.* Number of Ruiz iterations.
   * - ``equilibrate_min_scale``
     - 1e-4
     - *setup.* Lower bound of the cumulative scaling factors.
   * - ``equilibrate_max_scale``
     - 1e4
     - *setup.* Upper bound of the cumulative scaling factors.

Linear system
-------------

See :doc:`../algorithm/linear_system`.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Field
     - Default
     - Meaning
   * - ``supernodal``
     - true
     - *setup.* Use the supernodal factorization where it pays (at least 16 nonzeros per column
       of :math:`L` on average), the simplicial one otherwise. Both give the same factor up to
       roundoff.
   * - ``refine_max_iter``
     - 10
     - Maximum iterative-refinement steps per KKT solve; 0 disables refinement.
   * - ``refine_reltol``
     - 1e-13
     - Refinement stops when :math:`\|r\|_\infty \le` ``refine_abstol`` :math:`+` ``refine_reltol``
       :math:`\cdot\|\text{rhs}\|_\infty`.
   * - ``refine_abstol``
     - 1e-12
     -
   * - ``refine_stop_ratio``
     - 5
     - Refinement stops when a step reduces the residual by less than this factor.
   * - ``refine_affine``
     - false
     - Also refine the affine (predictor) direction.

``Settings::regularization`` (:cpp:struct:`conicxx::RegularizationSettings`):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Field
     - Default
     - Meaning
   * - ``static_P``
     - 1e-8
     - Static regularization of the pivots of the :math:`x` rows (added).
   * - ``static_nonneg``
     - 1e-8
     - ... of the orthant rows (subtracted).
   * - ``static_soc``
     - 1e-8
     - ... of the second-order cone rows (subtracted).
   * - ``static_zero``
     - 1e-8
     - ... of the equality rows (subtracted). Factorization only; refinement removes it from the
       solution.
   * - ``static_proportional``
     - :math:`\varepsilon^2`
     - Added on every row, times :math:`\max_i |K_{ii}|`.
   * - ``static_soc_proportional``
     - 1e-12
     - Added on the rows of every second-order cone block, times the largest diagonal entry of
       that block of :math:`H`.
   * - ``dynamic_enable``
     - true
     - Dynamic regularization of pivots with the wrong sign or too small magnitude.
   * - ``dynamic_eps``
     - 1e-13
     - A pivot :math:`d_k` with :math:`\operatorname{sign}_k d_k <` ``dynamic_eps`` is replaced ...
   * - ``dynamic_delta``
     - 2e-7
     - ... by :math:`\operatorname{sign}_k\cdot` ``dynamic_delta``.

:math:`\varepsilon = 2.2\cdot 10^{-16}` is the machine epsilon of ``double``.

Warm start
----------

See :doc:`../algorithm/warm_start`.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Field
     - Default
     - Meaning
   * - ``warm_start``
     - true
     - Start the next ``solve()`` from the previous solution (or from ``setWarmStart()``) if it
       is at least as good as the cold start.
   * - ``warm_lambda``
     - 0.99
     - Weight :math:`\lambda` of the previous solution in the warm starting point.

Diagnostics
-----------

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Field
     - Default
     - Meaning
   * - ``verbose``
     - 0
     - 0: silent; 1: one summary line before and after the solve; 2: also one line per iteration.
       Output goes to stdout.
   * - ``validate_inputs``
     - true
     - ``setup()`` and ``updateData()`` reject data with non-finite values. Costs one pass over the
       data.
