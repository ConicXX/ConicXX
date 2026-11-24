Frictional contact problems
===========================

ConicXX was written as the per-time-step solver of a nonsmooth multibody simulation. This chapter
shows how a time step with joints, springs and frictional contacts maps onto the standard form
and what to watch for when the solver is called thousands of times per simulation. For the
mechanics background see :cite:t:`acary2008nonsmooth`.

One time step as a conic QP
---------------------------

Consider a velocity-level time step with generalized velocities :math:`u` (after the step) and
:math:`u^-` (before), mass matrix :math:`M`, step size :math:`h` and generalized forces :math:`f`.
Bilateral constraints (joints) have Jacobian :math:`W_g`, contact :math:`i` has a normal and two
tangential directions collected in :math:`W_i \in \R^{n\times 3}` and friction coefficient
:math:`\mu_i`. The convex relaxation of Coulomb friction of :cite:t:`anitescu2006optimization` replaces
the normal condition :math:`v_n \ge 0` by :math:`v_n \ge \mu\|v_t\|`; it turns the time step into
the cone complementarity problem

.. math::

   \begin{aligned}
   M(u - u^-) - h f &= W_g \lambda_g + \textstyle\sum_i W_i \lambda_i, \\
   W_g^\top u + \beta_g &= 0, \\
   \lambda_i \in \mathcal F_i,\quad v_i = W_i^\top u + \beta_i \in \mathcal F_i^*,\quad
   \lambda_i^\top v_i &= 0,
   \end{aligned}

with friction cone :math:`\mathcal F_i = \{\lambda : \mu_i\lambda_n \ge \|\lambda_t\|\}`, its dual
:math:`\mathcal F_i^* = \{v : v_n \ge \mu_i \|v_t\|\}`, and stabilization or restitution terms
:math:`\beta`. These are exactly the optimality conditions :eq:`eq-kkt` of a ConicXX problem with

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ConicXX
     - Time step
   * - :math:`x`
     - :math:`u`
   * - :math:`P`, :math:`q`
     - :math:`M`, :math:`-(M u^- + h f)`
   * - zero-cone rows
     - :math:`A_E = W_g^\top`, :math:`b_E = -\beta_g`, multipliers :math:`z_E = -\lambda_g`
   * - one :math:`\Q^3` block per contact
     - :math:`A_i = -\diag(1, \mu_i, \mu_i)\, W_i^\top`, :math:`b_i = \diag(1,\mu_i,\mu_i)\,\beta_i`

Then :math:`s_i = (v_n, \mu_i v_t) \in \Q^3` is :math:`v_i \in \mathcal F_i^*`, and
:math:`z_i = (\lambda_n, \lambda_t/\mu_i) \in \Q^3` is :math:`\lambda_i \in \mathcal F_i`. This
scaling has no division by :math:`\mu_i`; a frictionless contact (:math:`\mu_i = 0`) is a
:math:`\Q^3` block whose tangential rows are zero, or simply a nonnegative row.

Springs and dampers acting along bilateral constraints become compliant rows
(:ref:`compliant-rows`): the compliance of a row is that of the force
element in the discretized equations, :math:`z_E = -\pi` is its force, and a stiff element is
approached continuously as :math:`c_i \to 0`.

What makes these problems hard
------------------------------

* **Degenerate contacts.** A contact that touches without transmitting force has
  :math:`v_i` and :math:`\lambda_i` both at the apex of their cones. Such pairs are not strictly
  complementary; interior-point methods converge more slowly on them, and the iterate is accurate
  to about :math:`\sqrt\mu` in these components. ConicXX measures their complementarity directly
  (the natural-map residual in the merit function, :doc:`../algorithm/termination`), handles the
  near-apex scaling with cancellation-free formulas (:doc:`../algorithm/cones`) and keeps the
  factorization of nearly singular cone blocks defined with a block-relative regularization
  (:doc:`../algorithm/linear_system`).
* **Redundant joints.** Closed kinematic loops and hyperstatic supports give linearly dependent
  equality rows. Static regularization makes the KKT matrix quasidefinite so that it always
  factorizes; refinement against the exact matrix still solves consistent redundant equalities to
  full accuracy. Inconsistent or numerically dependent equalities are reported through
  ``Info::equality_rank_deficient``.
* **Scale.** Impulses at small time steps are small numbers; masses and inertias differ by orders
  of magnitude. Equilibration and the unscaled, relative termination criteria make the iteration
  count largely independent of the units.

Recommendations
---------------

* **Keep the pattern, avoid** ``setup()``. ``setup()`` typically costs about as much as one or two
  iterations. If the contact set changes from step to step, consider reserving rows for candidate
  contacts from the broad phase (stored zeros in :math:`A`, an inactive contact with a large gap in
  :math:`b`). Then ``updateData()`` succeeds and the solver's own warm start survives.
* **Warm-start explicitly after** ``setup()``. ``setup()`` discards the stored warm start. Map the
  previous contact impulses onto the new contact set and call ``setWarmStart(u, s, z)``; contacts
  without a predecessor can get zeros (the warm start is projected onto the cones and blended with
  the standard starting point). ``Info::warm_start_used`` tells whether it was accepted.
* **Choose tolerances for the simulation, not for the optimizer.** The defaults (1e-8) are those of
  general-purpose conic solvers. Contact simulations usually need far less; relax
  ``tol_feas``/``tol_gap_*`` to the accuracy your integrator needs and set ``max_iter`` to a hard
  per-step budget. ``AlmostSolved`` then signals "usable but outside tolerance".
* **Never ignore the status.** ``PrimalInfeasible`` on a contact problem usually means
  inconsistent constraints (for example a joint pulling through a contact); ``DualInfeasible``
  means a direction of free motion with negative cost, such as a body with zero mass.
* **Freeze failing instances.** Simulations are chaotic; to compare settings or to report a
  failure, save the exact :math:`(P, q, A, b, \mathcal K)` of the failing step and replay it,
  instead of comparing two simulation runs.
