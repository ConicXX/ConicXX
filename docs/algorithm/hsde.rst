Homogeneous self-dual embedding
===============================

Optimality and duality
----------------------

A point :math:`(x, s, z)` solves the primal :eq:`eq-primal` and dual :eq:`eq-dual` problems if and
only if it satisfies :eq:`eq-kkt`. For any primal feasible :math:`(x, s)` and dual feasible
:math:`(x, z)` (with the same :math:`x`), the duality gap is

.. math::

   \Big(\tfrac12 x^\top P x + q^\top x\Big) - \Big(-\tfrac12 x^\top P x - b^\top z\Big)
   = x^\top P x + q^\top x + b^\top z = s^\top z \;\ge\; 0,

which follows from :math:`s^\top z = (b - Ax)^\top z = b^\top z + x^\top (P x + q)`. Optimality is
therefore primal feasibility, dual feasibility and complementarity :math:`s^\top z = 0`.

A problem may also have no solution. Two alternatives can be certified:

* **Primal infeasibility.** If :math:`z \in \K^*` with :math:`A^\top z = 0` and
  :math:`b^\top z < 0` exists, no :math:`(x, s)` with :math:`Ax + s = b`, :math:`s \in \K` exists,
  because :math:`0 \le s^\top z = (b - Ax)^\top z = b^\top z < 0` would follow.
* **Dual infeasibility.** If :math:`x` with :math:`Px = 0`, :math:`-Ax \in \K` and
  :math:`q^\top x < 0` exists, then every feasible point :math:`x_0` can be moved along :math:`x`
  without leaving the feasible set, and the objective
  :math:`\tfrac12 (x_0 + t x)^\top P (x_0 + t x) + q^\top(x_0 + t x)` decreases without bound as
  :math:`t \to \infty`. The dual problem is then infeasible.

The embedding
-------------

The homogeneous self-dual embedding :cite:p:`ye1994homogeneous,andersen2003conic` treats all three
cases with one system. It introduces two scalars :math:`\tau, \kappa \ge 0` and asks for
:math:`(x, s, z, \tau, \kappa)` with

.. math::
   :label: eq-hsde

   \begin{aligned}
   P x + A^\top z + q\tau &= 0, \\
   A x + s - b\tau &= 0, \\
   q^\top x + b^\top z + \tfrac1\tau x^\top P x + \kappa &= 0, \\
   (s, z) \in \K \times \K^*, \quad \tau, \kappa &\ge 0 .
   \end{aligned}

The quadratic term :math:`\tfrac1\tau x^\top P x` is the extension to quadratic objectives used by
Clarabel :cite:p:`goulart2024clarabel` and QOCO :cite:p:`chari2025qoco`; for :math:`P = 0` it
reduces to the classical linear embedding (as in ECOS and SCS
:cite:p:`domahidi2013ecos,odonoghue2016scs`).

Substituting the first two equations into the third shows that every solution satisfies

.. math::

   s^\top z + \tau\kappa = \tau(b^\top z + q^\top x) + x^\top P x
   - x^\top P x - \tau q^\top x - \tau b^\top z = 0,

and since both terms are nonnegative, :math:`s^\top z = 0` and :math:`\tau\kappa = 0`. This gives
the interpretation of a solution of :eq:`eq-hsde`:

* :math:`\tau > 0,\ \kappa = 0`: :math:`(x, s, z)/\tau` solves :eq:`eq-kkt`.
* :math:`\tau = 0,\ \kappa > 0`: then :math:`Px = 0` (the quadratic term stays bounded),
  :math:`A^\top z = 0`, :math:`Ax + s = 0` and :math:`q^\top x + b^\top z = -\kappa < 0`. If
  :math:`b^\top z < 0`, :math:`z` certifies primal infeasibility; if :math:`q^\top x < 0`,
  :math:`x` certifies dual infeasibility (at least one of the two holds).

The system is homogeneous: multiplying a solution by :math:`t > 0` gives another solution. The
interior-point method approaches one of these limits along the central path, and the ratio
:math:`\kappa/\tau` tells which: it tends to 0 for solvable problems and to :math:`\infty` for
infeasible ones. Because of the homogeneity, ConicXX divides the whole iterate by
:math:`\max(\tau,\kappa)` whenever that value leaves :math:`[10^{-8}, 10^{8}]`, which keeps the
numbers in range without changing the method.

Residuals
---------

The interior-point method works with the residuals of :eq:`eq-hsde`,

.. math::
   :label: eq-residuals

   \begin{aligned}
   r_x &= -(P x + A^\top z + \tau q), \\
   r_z &= A x + s - \tau b, \\
   r_\tau &= q^\top x + b^\top z + \tfrac1\tau x^\top P x + \kappa,
   \end{aligned}

and drives them to zero together with the complementarity measure

.. math::

   \mu = \frac{s^\top z + \tau\kappa}{\nu + 1}.

Each Newton step reduces the linear residuals :math:`r_x, r_z` by exactly the factor
:math:`1 - \alpha(1 - \sigma)` (step length :math:`\alpha`, centering parameter :math:`\sigma`, see
:doc:`ipm`); the method is an infeasible path-following method that never needs a feasible
starting point.

Compliant rows in the embedding
-------------------------------

With compliant equality rows (:eq:`eq-compliant`) the term :math:`-C z_E` joins the zero-cone rows
of the primal residual, and the quadratic form includes the eliminated variables
:math:`\pi = -z_E`:

.. math::

   r_{z,E} = A_E x - C z_E - \tau b_E, \qquad
   x^\top P x \;\to\; x^\top P x + z_E^\top C z_E

in :math:`r_\tau`, in the objectives and in the elimination of :math:`\Delta\tau`
(:doc:`ipm`). These are exactly the iterates the method would produce on the augmented problem
with explicit variables :math:`\pi`: every Newton step of the augmented problem keeps
:math:`\pi = -z_E`, because the :math:`\pi` rows of its KKT system read
:math:`C(\Delta\pi + \Delta z_E) = 0`.
