Problem formulation
===================

The standard form
-----------------

ConicXX solves

.. math::
   :label: eq-primal

   \begin{array}{ll}
   \text{minimize}   & \tfrac12 x^\top P x + q^\top x \\
   \text{subject to} & A x + s = b, \\
                     & s \in \K,
   \end{array}

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Symbol
     - C++ type
     - Meaning
   * - :math:`x`
     - ``Vec``, size :math:`n`
     - Decision variable, returned in ``Solution::x``.
   * - :math:`P`
     - ``SparseMat``, :math:`n\times n`
     - Symmetric positive semidefinite cost matrix. **Only the upper triangle is read**, including
       the diagonal. :math:`P = 0` (an empty sparse matrix) gives a linear or second-order cone
       program.
   * - :math:`q`
     - ``Vec``, size :math:`n`
     - Linear cost.
   * - :math:`A`
     - ``SparseMat``, :math:`m\times n`
     - Constraint matrix.
   * - :math:`b`
     - ``Vec``, size :math:`m`
     - Constraint right-hand side.
   * - :math:`s`
     - ``Vec``, size :math:`m`
     - Slack, constrained to :math:`\K`, returned in ``Solution::s``.
   * - :math:`z`
     - ``Vec``, size :math:`m`
     - Dual variable (Lagrange multiplier of :math:`Ax + s = b`), lies in the dual cone
       :math:`\K^*`, returned in ``Solution::z``.

The dual problem is

.. math::
   :label: eq-dual

   \begin{array}{ll}
   \text{maximize}   & -\tfrac12 x^\top P x - b^\top z \\
   \text{subject to} & P x + A^\top z + q = 0, \\
                     & z \in \K^* ,
   \end{array}

and a primal-dual solution satisfies the optimality conditions

.. math::
   :label: eq-kkt

   P x + A^\top z + q = 0, \qquad A x + s = b, \qquad s \in \K,\; z \in \K^*,\; s^\top z = 0 .

The sign convention for :math:`z` is the one of Clarabel, SCS and ECOS: for an inequality
:math:`a^\top x \le c` the multiplier is :math:`z_i \ge 0`.

The cone
--------

:math:`\K` is a Cartesian product of blocks, **always in this order**:

.. math::

   \K = \{0\}^{p} \times \R^{l}_{+} \times \Q^{k_1} \times \dots \times \Q^{k_N}.

The block sizes are described by :cpp:struct:`conicxx::ConeSpec`:

.. code-block:: cpp

   conicxx::ConeSpec cones;
   cones.zero_dim   = p;               // equality rows
   cones.nonneg_dim = l;               // inequality rows
   cones.soc_dims   = {k1, k2, ...};   // one entry per second-order cone block

The rows of :math:`A` and :math:`b` must be ordered the same way: first the :math:`p` zero-cone
rows, then the :math:`l` orthant rows, then the second-order cone blocks in the order of
``soc_dims``. ``cones.totalDim()`` must equal :math:`m`.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Block
     - Set
     - Dual cone
   * - Zero cone :math:`\{0\}^p`
     - :math:`s = 0`
     - :math:`\R^p` (free multipliers)
   * - Nonnegative orthant :math:`\R^l_+`
     - :math:`s_i \ge 0`
     - :math:`\R^l_+` (self-dual)
   * - Second-order cone :math:`\Q^k`
     - :math:`\{(t, u) \in \R\times\R^{k-1} : t \ge \|u\|_2\}`
     - :math:`\Q^k` (self-dual)

A second-order cone of dimension 1 is the half-line :math:`t \ge 0`; dimension 2 is
:math:`t \ge |u|`. Exponential, power and semidefinite cones are not supported.

Encoding common constraints
---------------------------

Every constraint is written as rows of :math:`A` and :math:`b` such that :math:`s = b - Ax` is the
quantity you want in the cone.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Constraint
     - Cone
     - Rows of :math:`A` / :math:`b`
   * - :math:`a^\top x = c`
     - zero
     - :math:`a^\top` / :math:`c`
   * - :math:`a^\top x \le c`
     - nonnegative
     - :math:`a^\top` / :math:`c`
   * - :math:`a^\top x \ge c`
     - nonnegative
     - :math:`-a^\top` / :math:`-c`
   * - :math:`\ell \le x_i \le u`
     - nonnegative, 2 rows
     - :math:`-e_i^\top` / :math:`-\ell` and :math:`e_i^\top` / :math:`u`
   * - :math:`\|F x + g\|_2 \le c^\top x + d`
     - second-order, dim :math:`1+\operatorname{rows}(F)`
     - :math:`\begin{bmatrix}-c^\top\\ -F\end{bmatrix}` / :math:`\begin{bmatrix} d\\ g\end{bmatrix}`
   * - :math:`\mu f_n \ge \|f_t\|_2` (Coulomb friction)
     - second-order, dim 3
     - :math:`-\diag(\mu,1,1)` on :math:`(f_n, f_t)` / :math:`0`
   * - :math:`\|x\|_2 \le t` with :math:`t` a variable
     - second-order
     - :math:`-I` on :math:`(t, x)` / :math:`0`

Quadratic constraints :math:`x^\top Q x \le r` with :math:`Q = F^\top F` become
:math:`\|F x\|_2 \le \sqrt r`. A rotated cone :math:`2uv \ge \|w\|^2,\ u,v\ge 0` is the
second-order cone :math:`\|(\sqrt2\,w,\ u - v)\|_2 \le u + v`.

.. _compliant-rows:

Compliant equality rows
-----------------------

The zero-cone rows can optionally carry a diagonal compliance :math:`C = \diag(c)`,
:math:`c \ge 0`, passed to :cpp:func:`conicxx::Solver::setup` and
:cpp:func:`conicxx::Solver::updateData` as ``zero_compliance`` (one entry per zero-cone row). The
problem then becomes

.. math::
   :label: eq-compliant

   \begin{array}{ll}
   \text{minimize}   & \tfrac12 x^\top P x + q^\top x + \tfrac12 z_E^\top C z_E \\
   \text{subject to} & A x - \begin{bmatrix} C z_E \\ 0 \end{bmatrix} + s = b, \quad s \in \K,
   \end{array}

where :math:`z_E` are the multipliers of the zero-cone rows. This is the elimination of extra
variables :math:`\pi = -z_E` with cost :math:`\tfrac12\pi^\top C\pi` and rows
:math:`A_E x + C\pi = b_E`, without adding them to the problem. In multibody dynamics,
:math:`\pi` are the forces (or impulses) of springs and dampers acting along the constraint and
:math:`C` their compliance; :math:`c_i = 0` is an ordinary rigid equality, and the solution depends
continuously on :math:`c_i \to 0`. See :doc:`examples` for a complete program and
:doc:`../algorithm/linear_system` for how the compliance enters the KKT matrix.

Infeasible and unbounded problems
---------------------------------

ConicXX detects both cases and returns a certificate:

* **Primal infeasible** (no :math:`x` satisfies the constraints): a :math:`z \in \K^*` with
  :math:`A^\top z = 0` and :math:`b^\top z < 0`, normalized to :math:`b^\top z = -1` and returned in
  ``Solution::z``.
* **Dual infeasible** (the objective is unbounded below): an :math:`x` with :math:`Px = 0`,
  :math:`Ax + s = 0` for some :math:`s \in \K`, and :math:`q^\top x < 0`, normalized to
  :math:`q^\top x = -1` and returned in ``Solution::x`` (and ``Solution::s``).

The precise tests are in :doc:`../algorithm/termination`.

Scaling and units
-----------------

ConicXX equilibrates the problem internally (:doc:`../algorithm/equilibration`) and evaluates
all termination criteria on the original, unscaled data, so tolerances refer to your units.
Residuals are relative to the size of the data, but the absolute duality gap tolerance is not:
formulating a problem with very large or very small objective values (for example impulses in
kg·m/s at a small time step) shifts which of the two gap criteria decides convergence.
