Termination
===========

The termination criteria follow Clarabel :cite:p:`goulart2024clarabel`. They are checked at the
top of every iteration, before the KKT matrix is factorized, so the final iteration costs no
factorization. All quantities are evaluated for the **unscaled** problem
(:doc:`equilibration`).

Quantities
----------

With :math:`(x, s, z) = (x, s, z)_\text{iterate}/\tau` in original units:

.. math::

   \begin{aligned}
   \text{primal residual} \quad r_p &= \frac{\|Ax + s - b\|_\infty}{\max\big(1,\ \|b\|_\infty + \|x\|_\infty + \|s\|_\infty\big)}, \\
   \text{dual residual} \quad r_d &= \frac{\|Px + A^\top z + q\|_\infty}{\max\big(1,\ \|q\|_\infty + \|x\|_\infty + \|z\|_\infty\big)}, \\
   \text{primal objective} \quad c_p &= \tfrac12 x^\top P x + q^\top x, \\
   \text{dual objective} \quad c_d &= -\tfrac12 x^\top P x - b^\top z, \\
   \text{gaps} \quad g_\text{abs} &= |c_p - c_d|, \qquad
   g_\text{rel} = \frac{g_\text{abs}}{\max\big(1,\ \min(|c_p|, |c_d|)\big)} .
   \end{aligned}

With compliant rows, :math:`Ax` includes :math:`-Cz_E` and :math:`x^\top P x` includes
:math:`z_E^\top C z_E` (:doc:`hsde`). The ratio :math:`\kappa/\tau` is scale-invariant and needs no
unscaling.

Optimality
----------

The iterate is ``Solved`` when

.. math::

   r_p < \texttt{tol\_feas}, \qquad r_d < \texttt{tol\_feas}, \qquad
   \big(g_\text{abs} < \texttt{tol\_gap\_abs} \ \text{ or }\ g_\text{rel} < \texttt{tol\_gap\_rel}\big),
   \qquad \kappa/\tau \le 1 .

The gap test is an *or* of the absolute and relative test, and the relative gap uses the
*smaller* of the two objective magnitudes, as in Clarabel. The bound on :math:`\kappa/\tau` is a
sanity check that excludes iterates on their way to an infeasibility certificate.

Infeasibility
-------------

Certificates are evaluated on the unnormalized iterate (not divided by :math:`\tau`, which tends to
zero for infeasible problems), and only once

.. math::

   \kappa/\tau > 1000 / \texttt{tol\_ktratio} \quad (= 10^9 \text{ by default}).

**Primal infeasibility** is declared when

.. math::

   b^\top z < -\epsilon_\text{abs}
   \quad\text{and}\quad
   \frac{\|A^\top z\|_\infty}{\max(1, \|z\|_\infty)} < -\epsilon_\text{rel}\, b^\top z,

and **dual infeasibility** when

.. math::

   q^\top x < -\epsilon_\text{abs}
   \quad\text{and}\quad
   \max\left(\frac{\|Px\|_\infty}{\max(1, \|x\|_\infty)},\
             \frac{\|Ax + s\|_\infty}{\max(1, \|x\|_\infty + \|s\|_\infty)}\right)
   < -\epsilon_\text{rel}\, q^\top x,

with :math:`\epsilon_\text{abs}` = ``tol_infeas_abs`` and :math:`\epsilon_\text{rel}` =
``tol_infeas_rel``. With compliance, :math:`\|Cz_E\|_\infty` enters both :math:`\|A^\top z\|` and
:math:`\|Px\|` (the columns of the eliminated variables). The returned certificate is scaled so
that :math:`b^\top z = -1` (resp. :math:`q^\top x = -1`). In units of the original problem,
:math:`A^\top z` is computed as :math:`-(r_x + Px + \tau q)` and :math:`Ax + s` as
:math:`r_z + \tau b` from the residuals that are already available, without extra matrix-vector
products.

Merit function and best iterate
-------------------------------

Every iterate is scored by

.. math::

   \text{merit} = \max\Big(r_p,\ r_d,\ \min(g_\text{abs}, g_\text{rel}),\ r_c\Big), \qquad
   r_c = \big\|z - \proj_\K(z - s)\big\|_\infty \ \text{over the orthant and SOC rows},

where :math:`r_c` is the natural-map residual of complementarity. It vanishes exactly when
:math:`s \in \K`, :math:`z\in\K` and :math:`s^\top z = 0` blockwise. The gap alone does not
measure this: on problems whose solution is not strictly complementary (contacts touching without
force) the gap falls like the square of the distance to complementarity, so iterates with a small
gap can still be far from complementary pairs. The solver keeps a copy of the iterate with the
smallest merit.

.. _progress-checks:

Progress checks
---------------

The solve stops with ``InsufficientProgress`` when

* two consecutive steps are shorter than ``min_terminate_step_length`` (:math:`10^{-4}`), or
* from the third iteration on, a residual grows (:math:`r_p` or :math:`r_d` larger than in the
  previous iteration), and either

  - **stalled:** the previous gap already met its tolerance, :math:`\kappa/\tau < 100\varepsilon`,
    and the current merit is worse than the best one; or
  - **diverging:** :math:`\kappa/\tau < 1` and a residual exceeds
    :math:`\max(100\,\texttt{tol\_feas}, \sqrt\varepsilon)` after growing by more than a factor 100.

The :math:`\sqrt\varepsilon` floor matters for tight tolerances: below it, a growing residual is
roundoff of an accurate iterate and not a sign of divergence. The merit condition matters where
the relative gap is measured against an objective that is much larger than the complementarity
terms (heavy bodies with small contact impulses): there the gap converges long before the contact
complementarity does, and the solve should continue as long as the merit improves.

Early stops and reduced tolerances
----------------------------------

On ``MaxIterations``, ``MaxTime``, ``InsufficientProgress`` and ``NumericalError`` the best
iterate is restored and tested once more with the reduced tolerances ``reduced_tol_*``. If it
passes the optimality test, the status becomes ``AlmostSolved``; otherwise, if
:math:`\kappa/\tau > 1000/\texttt{reduced\_tol\_ktratio}` and a certificate test passes,
``AlmostPrimalInfeasible`` or ``AlmostDualInfeasible``. Otherwise the original status is kept,
and the best iterate is still returned with its residuals in ``Info``.

Additions to Clarabel's rules
-----------------------------

The optimality and certificate tests are Clarabel's. ConicXX adds the complementarity residual
:math:`r_c` to the merit function, and its progress check uses that merit and the
:math:`\sqrt\varepsilon` floor described above. Both were introduced for frictional contact
problems, where the gap-based rules either stopped early at inaccurate contact forces or ended
accurate solves because of roundoff.
