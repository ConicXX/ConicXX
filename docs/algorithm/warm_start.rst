Warm start
==========

In a time-stepping simulation consecutive problems are close, and the previous solution is a much
better guess than the cold start. Interior-point methods cannot use it directly, however: a
converged point lies on the boundary of the cones, where the method cannot start, and its
complementarity pairs are tiny and badly centred for the new data.

Construction of the starting point
----------------------------------

ConicXX uses the warm start of :cite:t:`skajaa2013warmstart` for the homogeneous embedding. With a
previous solution :math:`(x^*, s^*, z^*)` (in equilibrated units, divided by its :math:`\tau`) and
the standard starting point :math:`(0, e, e, 1, 1)`, the starting point is

.. math::

   x_0 = \lambda x^*, \qquad s_0 = \lambda s^* + (1 - \lambda) e, \qquad
   z_0 = \lambda z^* + (1 - \lambda) e, \qquad \tau_0 = 1, \qquad \kappa_0 = 1 - \lambda,

with :math:`\lambda` = ``warm_lambda`` (0.99). Its residuals are about :math:`\lambda` times those
of the previous solution for the new data plus :math:`(1-\lambda)` times those of the standard
point, and every complementarity pair is at least :math:`(1-\lambda)^2`, so the point is interior
and not badly centred. Zero-cone slacks stay zero.

Before the blend, :math:`s^*` and :math:`z^*` are projected onto the cones. This is a no-op for the
solver's own solutions and makes a user-supplied warm start (``setWarmStart()``) usable even if it
is infeasible, for example impulses mapped from the previous contact set.

The blend treats all variables alike. Shifting only :math:`(s^*, z^*)` into the interior and
rescaling them to a target :math:`\mu` would leave :math:`x` and :math:`\tau` unchanged and
multiply the residuals :math:`Ax + s - \tau b` and :math:`Px + A^\top z + \tau q` by the rescaling
factor, which near a converged point is large. Such a point loses against the cold start even when
the data did not change.

Acceptance
----------

The warm point is used only if its merit (:doc:`termination`) is not larger than that of the cold
start. To avoid computing the cold start in every step, the merit of the last cold start since
``setup()`` is cached:

1. Compute the warm point and its merit (residuals only, no factorization).
2. If a cold-start merit is cached and the warm merit is not larger, start from the warm point.
3. Otherwise compute the cold start (one factorization), cache its merit, and start from the better
   of the two.

An accepted warm start therefore saves the cold-start factorization in addition to iterations. The
first ``solve()`` after ``setup()`` always computes the cold start. ``Info::warm_start_used``,
``Info::warm_start_merit`` and ``Info::cold_start_merit`` report the decision.

When to expect a benefit
------------------------

The warm start helps most when the data change little and the active set (which contacts touch,
stick or slide) changes little. When many equality rows have a right-hand side that changes from
step to step (for example position-level stabilization of stiff constraints), the cold start solves
these linear equations exactly while the warm point carries the change, and the warm point tends to
lose the merit comparison; the solver then simply starts cold. The value of ``warm_lambda`` was
chosen on sequences of perturbed benchmark problems: 0.9 is too weak, and values above 0.99 help
contact problems slightly but hurt others.
