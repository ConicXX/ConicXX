Predictor--corrector iteration
==============================

This chapter describes one ``solve()`` from the starting point to the update of the iterate. The
termination tests at the top of every iteration are described in :doc:`termination`, the linear
algebra in :doc:`linear_system`.

Cold start
----------

The cold start follows Clarabel :cite:p:`goulart2024clarabel`. It solves one or two linear
systems with the KKT matrix at the identity scaling, :math:`H = I` on the orthant and second-order
cone rows and :math:`H = 0` (or the compliance) on the zero-cone rows:

* **LP and SOCP** (:math:`P = 0`, no compliance): two solves give the least-norm slack and dual,

  .. math::

     \min_{x,s}\ \tfrac12\|s\|^2 \ \text{ s.t. } Ax + s = b, \qquad
     \min_{z}\ \tfrac12\|z\|^2 \ \text{ s.t. } A^\top z + q = 0 .

* **QP**: one solve of

  .. math::

     \min_{x,s}\ \tfrac12 x^\top P x + q^\top x + \tfrac12\|s\|^2 \ \text{ s.t. } Ax + s = b,

  with :math:`z = -s` from its multipliers.

The zero-cone part of :math:`s` is then set to zero, and :math:`s` and :math:`z` are moved into
the interior separately: with :math:`\alpha` the smallest margin over all cone blocks
(:doc:`cones`) and

.. math::

   t = \max\Big(1,\ \tfrac{1}{10\,\nu}\textstyle\sum_\text{blocks} \max(\text{margin}, 0)\Big),

the vector is shifted along :math:`e` by :math:`t - \alpha` if :math:`\alpha < t`, so that its
smallest margin becomes :math:`t`. The cold start costs one factorization. Finally
:math:`\tau = \kappa = 1`.

The cold start is computed only if it is needed: a warm start whose merit is not worse than that
of the last cold start since ``setup()`` is used directly (:doc:`warm_start`).

Newton system
-------------

Linearizing the embedding :eq:`eq-hsde` and the scaled central-path conditions
:math:`\lambda\circ\lambda = \sigma\mu e`, :math:`\tau\kappa = \sigma\mu` at the current iterate
gives the system for the direction :math:`(\Delta x, \Delta s, \Delta z, \Delta\tau, \Delta\kappa)`

.. math::
   :label: eq-newton

   \begin{aligned}
   P\Delta x + A^\top\Delta z + q\Delta\tau &= d_x, \\
   A\Delta x + \Delta s - b\Delta\tau &= -d_z, \\
   (q + 2P\xi)^\top\Delta x + b^\top\Delta z - \xi^\top P\xi\,\Delta\tau + \Delta\kappa &= -d_\tau, \\
   \lambda\circ(W\Delta z + W^{-\top}\Delta s) &= -d_s, \\
   \kappa\Delta\tau + \tau\Delta\kappa &= -d_\kappa,
   \end{aligned}

with :math:`\xi = x/\tau`. The third row is the linearization of :math:`r_\tau` including its
quadratic term :math:`\tfrac1\tau x^\top P x`. The right-hand sides of the two directions computed
per iteration are

.. list-table::
   :header-rows: 1
   :widths: 16 32 52

   * -
     - Affine (predictor)
     - Combined (corrector)
   * - :math:`d_x`
     - :math:`r_x`
     - :math:`(1-\sigma)\,r_x`
   * - :math:`d_z`
     - :math:`r_z`
     - :math:`(1-\sigma)\,r_z`
   * - :math:`d_\tau`
     - :math:`r_\tau`
     - :math:`(1-\sigma)\,r_\tau`
   * - :math:`d_s`
     - :math:`\lambda\circ\lambda`
     - :math:`\lambda\circ\lambda + \eta\,(W^{-\top}\Delta s_a)\circ(W\Delta z_a) - \sigma\mu e`
   * - :math:`d_\kappa`
     - :math:`\tau\kappa`
     - :math:`\tau\kappa + \eta\,\Delta\tau_a\Delta\kappa_a - \sigma\mu`

where :math:`r_x, r_z, r_\tau` are the residuals :eq:`eq-residuals`, the index :math:`a` marks
the affine direction, and :math:`\eta` scales Mehrotra's second-order correction
:cite:p:`mehrotra1992implementation`: :math:`\eta = 1`, except in the first iteration, where
:math:`\eta = \alpha_a` (the affine step length) because the starting point, cold or warm, may be
badly centred (as in Clarabel). The second-order term
:math:`(W^{-\top}\Delta s_a)\circ(W\Delta z_a)` is the symmetric-cone form of Mehrotra's
correction :cite:p:`vandenberghe2010cvxopt`.

Reduction to the KKT system
---------------------------

Because :math:`W` is symmetric, the fourth row gives

.. math::

   \Delta s = -W^\top\big(\lambda\backslash d_s\big) - H\Delta z, \qquad H = W^\top W,

and the fifth row :math:`\Delta\kappa = -(d_\kappa + \kappa\Delta\tau)/\tau`. Substituting
:math:`\Delta s` into the second row leaves

.. math::
   :label: eq-reduced

   \begin{bmatrix} P & A^\top \\ A & -H \end{bmatrix}
   \begin{bmatrix} \Delta x \\ \Delta z \end{bmatrix}
   =
   \begin{bmatrix} d_x \\ W^\top(\lambda\backslash d_s) - d_z \end{bmatrix}
   + \Delta\tau \begin{bmatrix} -q \\ b \end{bmatrix} .

The solution is affine in :math:`\Delta\tau`: with the KKT matrix :math:`K` of :eq:`eq-reduced`,

.. math::

   K \begin{bmatrix} x_c \\ z_c \end{bmatrix} = \begin{bmatrix} -q \\ b \end{bmatrix}, \qquad
   K \begin{bmatrix} x_v \\ z_v \end{bmatrix} =
   \begin{bmatrix} d_x \\ W^\top(\lambda\backslash d_s) - d_z \end{bmatrix},

the direction is :math:`(\Delta x, \Delta z) = (x_v, z_v) + \Delta\tau\,(x_c, z_c)`. Inserting it
together with :math:`\Delta\kappa` into the third row of :eq:`eq-newton` gives

.. math::
   :label: eq-dtau

   \Delta\tau = \frac{d_\tau - d_\kappa/\tau + (q + 2P\xi)^\top x_v + b^\top z_v}
                     {\kappa/\tau + \xi^\top P\xi - (q + 2P\xi)^\top x_c - b^\top z_c} .

The constant solve :math:`(x_c, z_c)` depends only on :math:`K`, so it is computed once per
iteration and used for both directions. It is solved together with the affine right-hand side in
one pass over the factor (the affine direction is not refined by default, because it only sets
:math:`\sigma` and the second-order correction; ``Settings::refine_affine``). One iteration
therefore costs one factorization, one paired solve and one single solve.

For the affine direction, :math:`W^\top(\lambda\backslash d_s) = W^\top\lambda = s`, so its
right-hand side is simply :math:`(r_x,\ s - r_z)`.

Centering parameter
-------------------

After the affine direction, its maximal step :math:`\alpha_a \le 1` to the boundary of the cones
and of :math:`\tau,\kappa \ge 0` is computed, and

.. math::

   \mu_a = \frac{(s + \alpha_a\Delta s_a)^\top(z + \alpha_a\Delta z_a)
   + (\tau + \alpha_a\Delta\tau_a)(\kappa + \alpha_a\Delta\kappa_a)}{\nu + 1}, \qquad
   \sigma = \min\Big(1,\ \big(\max(\mu_a/\mu, 0)\big)^3\Big) .

This is Mehrotra's heuristic. Clarabel uses :math:`\sigma = (1 - \alpha_a)^3` instead; the two
agree when the affine step reduces :math:`\mu` in proportion to the step length.

Step length and centrality safeguard
------------------------------------

For the combined direction, the maximal step :math:`\alpha_\text{max}` to the boundary is computed
for :math:`s`, :math:`z` (:doc:`cones`), :math:`\tau` and :math:`\kappa`, capped at 1, and scaled
by ``max_step_fraction`` (0.99). Starting from :math:`\alpha = \alpha_\text{max}`, the step is then
reduced by ``linesearch_backtrack`` (0.8) until the trial point

.. math::

   (s, z, \tau, \kappa) + \alpha(\Delta s, \Delta z, \Delta\tau, \Delta\kappa)

is strictly interior and well centred: every cone block :math:`i` and the pair
:math:`(\tau,\kappa)` satisfy

.. math::

   \text{centrality}_i(s^+, z^+) \ge \theta\,\mu^+, \qquad \tau^+\kappa^+ \ge \theta\,\mu^+,

with :math:`\theta` = ``centrality_theta`` (:math:`10^{-4}`) and :math:`\mu^+` the complementarity
measure of the trial point. The centrality is the smallest eigenvalue of
:math:`\lambda^+\circ\lambda^+` for the NT point of the trial pair itself, computed exactly
(:doc:`cones`), not through the scaling of the current point. This is the
negative-infinity neighbourhood of the central path :cite:p:`wright1997primaldual`; with the small
:math:`\theta` it only rejects steps that leave some pair badly uncentred. After 64 reductions the
last (tiny) step is taken; the progress check then ends the solve if this repeats.

The iterate is updated, :math:`(x, s, z, \tau, \kappa) \mathrel{+}= \alpha\,\Delta`, and
rescaled by :math:`1/\max(\tau,\kappa)` if :math:`\max(\tau,\kappa)` leaves
:math:`[10^{-8}, 10^{8}]` (:doc:`hsde`).

Numerical failure
-----------------

The iteration stops with ``NumericalError`` if a factorization meets a zero or non-finite pivot
(only possible with non-finite data or with dynamic regularization disabled), if a direction or
:math:`\Delta\tau` is not finite, if the denominator of :eq:`eq-dtau` is zero, or if the step
length is not positive. In every case the best iterate found so far is reported, possibly as
``AlmostSolved`` (:doc:`termination`). There are no fallbacks that change the problem.
