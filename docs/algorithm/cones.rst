Cones and Nesterov--Todd scaling
================================

The nonnegative orthant and the second-order cone are *symmetric* cones: self-dual and
homogeneous. Interior-point methods for them are conveniently written in the language of Euclidean
Jordan algebras :cite:p:`alizadeh2003socp,vandenberghe2010cvxopt`. The zero cone is handled
separately: its slack is identically zero and its dual variable is free, so it contributes no
complementarity condition.

Jordan algebra
--------------

For each cone block the Jordan product :math:`u \circ v`, the identity :math:`e` and the degree
are

.. list-table::
   :header-rows: 1
   :widths: 22 40 18 20

   * - Block
     - :math:`u \circ v`
     - :math:`e`
     - degree
   * - :math:`\R^l_+`
     - :math:`(u_1 v_1, \dots, u_l v_l)`
     - :math:`(1,\dots,1)`
     - :math:`l`
   * - :math:`\Q^k`
     - :math:`\big(u^\top v,\; u_0 v_1 + v_0 u_1\big)`
     - :math:`(1, 0, \dots, 0)`
     - :math:`1`

where :math:`u = (u_0, u_1) \in \R\times\R^{k-1}` for the second-order cone. The degree
:math:`\nu` of :math:`\K` is the sum over all blocks. Complementarity of :math:`s \in \K` and
:math:`z\in\K` is equivalent to :math:`s \circ z = 0`, and the central path of the method is
:math:`s \circ z = \mu e`.

For :math:`\Q^k` the following quantities are used throughout:

.. math::

   \det(u) = u_0^2 - \|u_1\|^2 = (u_0 - \|u_1\|)(u_0 + \|u_1\|), \qquad
   \text{margin}(u) = u_0 - \|u_1\|.

:math:`u` is interior if and only if its margin is positive. ConicXX always evaluates the
determinant in the factored form on the right. The expanded form :math:`u_0^2 - \|u_1\|^2`
cancels catastrophically near the cone boundary, which is exactly where active frictional contacts
live.

The **inverse product** :math:`u \backslash w`, the solution :math:`v` of :math:`u \circ v = w` for
interior :math:`u`, is

.. math::

   v_0 = \frac{u_0 w_0 - u_1^\top w_1}{\det(u)}, \qquad
   v_1 = \frac{w_1 - v_0 u_1}{u_0},

for the second-order cone, and :math:`v_i = w_i / u_i` for the orthant. No clamping of
:math:`\det(u)` or :math:`u_0` is applied: :math:`u` is always strictly interior when this is
called, and an absolute floor would break the scale invariance of the method for problems whose
solutions are small (impulses at small step sizes).

Nesterov--Todd scaling
----------------------

Each iteration linearizes the complementarity condition in a scaled space. The Nesterov--Todd (NT)
scaling :cite:p:`nesterov1997selfscaled,nesterov1998primaldual` is a symmetric positive definite
:math:`W`, block diagonal over the cones and mapping each cone onto itself, that maps the current
:math:`z` and :math:`s` to the same point,

.. math::

   \lambda = W z = W^{-\top} s .

In the scaled variables the complementarity condition :math:`s \circ z = \mu e` becomes
:math:`\lambda\circ\lambda = \mu e`, and :math:`\lambda^\top\lambda = s^\top z`. The symmetry of the
scaled formulation is what makes the Newton system symmetric (:doc:`linear_system`).

**Orthant.** :math:`W = \diag(w)` with :math:`w_i = \sqrt{s_i / z_i}`, so
:math:`H = W^\top W = \diag(s_i / z_i)` and :math:`\lambda_i = \sqrt{s_i z_i}`.

**Second-order cone.** With :math:`a = \sqrt{\det s}`, :math:`b = \sqrt{\det z}` and

.. math::

   \gamma = \sqrt{\frac{1 + s^\top z / (a b)}{2}}, \qquad
   \bar w = \frac{1}{2\gamma}\left(\frac{s}{a} + J\frac{z}{b}\right), \qquad
   \eta = \sqrt{a / b},

where :math:`J = \diag(1, -1, \dots, -1)`, the scaling is

.. math::

   W = \eta \begin{bmatrix} \bar w_0 & \bar w_1^\top \\
   \bar w_1 & I + \dfrac{\bar w_1 \bar w_1^\top}{1 + \bar w_0} \end{bmatrix}, \qquad
   W^{-1} = \frac{1}{\eta^2} J W J, \qquad
   H = W^2 = \eta^2 \big(2 \bar w \bar w^\top - J\big),

with :math:`\bar w^\top J \bar w = 1`. ConicXX never forms :math:`W` as a matrix. Products with
:math:`W`, :math:`W^{-1}` and :math:`H` are applied in :math:`O(k)` operations from
:math:`(\eta, \bar w)`:

.. math::

   W x = \eta\Big(\bar w^\top x,\;\; x_1 + \big(x_0 + \tfrac{\bar w_1^\top x_1}{1 + \bar w_0}\big)\bar w_1\Big),
   \qquad
   H x = \eta^2 \big(2 (\bar w^\top x)\, \bar w - J x\big).

Only the KKT matrix needs :math:`H` explicitly. For a second-order cone block it is a dense
:math:`k\times k` block of the KKT matrix; for the orthant it is diagonal, and for the zero cone
it is zero (or the compliance :math:`C`, :doc:`linear_system`). There are no dense blocks for
orthant or zero-cone rows, and no sparse expansion of SOC blocks: for the small blocks of contact
problems (:math:`k = 3`) the dense block is the cheapest representation.

Batched kernels
---------------

All cone operations (scaling update, products with :math:`W`, :math:`W^{-1}`, :math:`H`, Jordan
product and inverse product, step length, centrality, projection) run as one loop over contiguous
arrays without virtual calls. If every second-order block has dimension 3, as in frictional
contact, the loop uses kernels with the dimension fixed at compile time.

Step to the boundary
--------------------

The maximal step :math:`\alpha` with :math:`x + \alpha\,\Delta x \in \K` for interior :math:`x` is
computed per block and minimized over all blocks:

* **Orthant:** :math:`\alpha = \min_{i:\,\Delta x_i < 0} -x_i / \Delta x_i`.
* **Second-order cone:** :math:`\det(x + \alpha\Delta x) = a\alpha^2 + b\alpha + c` with
  :math:`a = \det(\Delta x)`, :math:`b = 2(x_0\Delta x_0 - x_1^\top\Delta x_1)` and
  :math:`c = \det(x)`, all evaluated with factored determinants. The smallest positive root at
  which :math:`x_0 + \alpha\Delta x_0 \ge 0` limits the step. The roots are computed with the
  numerically stable form :math:`q = -\tfrac12(b + \operatorname{sign}(b)\sqrt{b^2 - 4ac})`,
  :math:`r_1 = q/a`, :math:`r_2 = c/q`. A linear equation is used only when :math:`a` is exactly
  zero up to the smallest normal numbers; a larger threshold would misclassify ordinary directions
  of large or small scale.

Centrality
----------

The step-length safeguard (:doc:`ipm`) needs the centrality of a pair :math:`(s, z)`: the smallest
eigenvalue of :math:`\lambda\circ\lambda` for the NT point :math:`\lambda` of :math:`(s, z)`. It is
computed directly from :math:`s` and :math:`z`, without forming a scaling:

* **Orthant:** :math:`\min_i s_i z_i`.
* **Second-order cone:** from :math:`\lambda^\top\lambda = s^\top z` and
  :math:`\det\lambda = \sqrt{\det s\,\det z}`,

  .. math::

     (\lambda_0 - \|\lambda_1\|)^2 = \frac{\det s \,\det z}{s^\top z + \sqrt{(s^\top z)^2 - \det s\,\det z}},

  which avoids the cancellation of the direct form :math:`s^\top z - \sqrt{(s^\top z)^2 - \det s\det z}`.

Projection
----------

The Euclidean projection onto :math:`\K` is used for the complementarity residual
(:doc:`termination`) and to make a user-supplied warm start interior (:doc:`warm_start`). For
:math:`\Q^k` with :math:`u = (t, u_1)`:

.. math::

   \proj_{\Q}(u) =
   \begin{cases}
   u & \|u_1\| \le t, \\
   0 & \|u_1\| \le -t, \\
   \tfrac12\big(t + \|u_1\|\big)\Big(1,\ \dfrac{u_1}{\|u_1\|}\Big) & \text{otherwise,}
   \end{cases}

and :math:`\max(u_i, 0)` componentwise for the orthant.
