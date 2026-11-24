Equilibration
=============

Interior-point methods are invariant to many transformations in exact arithmetic, but not in
floating point: badly scaled rows and columns make the KKT matrix ill-conditioned long before the
iterate reaches the cone boundary. ``setup()`` therefore rescales the data (unless
``Settings::equilibrate`` is false) with diagonal matrices :math:`D \in \R^{n\times n}`,
:math:`E\in\R^{m\times m}` and a scalar :math:`c > 0`:

.. math::

   \hat P = c\,DPD, \qquad \hat q = c\,Dq, \qquad \hat A = EAD, \qquad \hat b = Eb, \qquad
   \hat C = \frac1c E C E .

The scaled problem is solved, and the solution is recovered as

.. math::

   x = D\hat x, \qquad s = E^{-1}\hat s, \qquad z = \frac1c E\hat z .

Ruiz iteration
--------------

:math:`D` and :math:`E` are computed by a modified Ruiz equilibration :cite:p:`ruiz2001scaling` of
the symmetric matrix

.. math::

   M = \begin{bmatrix} P & A^\top \\ A & 0 \end{bmatrix},

as in OSQP and Clarabel :cite:p:`stellato2020osqp,goulart2024clarabel`. Starting from
:math:`D = I`, :math:`E = I`, :math:`c = 1`, each of ``equilibrate_max_iter`` (10) iterations does

1. Compute the infinity norms of the columns of the current scaled :math:`M`: :math:`\nu_x` for the
   :math:`n` variable columns (from :math:`\hat P` and :math:`\hat A`) and :math:`\nu_z` for the
   :math:`m` constraint columns (from :math:`\hat A`).
2. **Cone-preserving rows.** On every second-order cone block, replace the entries of
   :math:`\nu_z` by their maximum over the block. A cone block must be scaled by a single factor,
   because :math:`\diag(e_1,\dots,e_k)\Q^k \ne \Q^k` in general. (Orthant rows can be scaled
   individually.)
3. Factors :math:`\delta_i = 1/\sqrt{\nu_i}`, with norms below ``equilibrate_min_scale`` treated as
   1 (empty rows and columns are not scaled) and norms above ``equilibrate_max_scale`` clipped.
4. **Clipping of the cumulative factors.** Each :math:`\delta_i` is clipped so that the cumulative
   :math:`d_i\delta_i` (resp. :math:`e_i\delta_i`) stays in
   :math:`[\texttt{min\_scale}, \texttt{max\_scale}] = [10^{-4}, 10^{4}]`. The stored data are
   then exactly :math:`cDPD`, :math:`EAD`, so no correction is needed afterwards.
5. Scale :math:`\hat P`, :math:`\hat A`, :math:`\hat q` and update :math:`D`, :math:`E`.
6. **Cost scaling.** With :math:`\gamma = \max\big(\operatorname{mean}_j \|\hat P_{:,j}\|_\infty,\
   \|\hat q\|_\infty\big)`, scale the cost by :math:`1/\gamma` (limited as in step 3 and clipped so
   that :math:`c` stays in the same bounds), unless :math:`\gamma = 0`.

The compliance follows the scaling (:math:`\hat C = ECE/c`) but does not enter the norms: like
:math:`H` it belongs to the (2,2) block of the KKT matrix, which the equilibration of :math:`M`
leaves out.

Including :math:`\|q\|_\infty` in the cost scale makes the iteration count largely independent of
the magnitude of :math:`q`. In a measurement with a box-constrained QP with :math:`P = I`, it needed
12, 8 and 8 iterations for :math:`\|q\|_\infty = 1, 10^3, 10^6`, against 12, 12 and 15 with a cost
scale based on :math:`P` alone.

``updateData()`` applies the scaling computed in ``setup()`` to the new values; it does not
recompute :math:`D`, :math:`E` and :math:`c`. If the data change by orders of magnitude during a
simulation, a new ``setup()`` re-equilibrates.

Working in scaled units
-----------------------

All iterations work with the scaled data. The termination criteria, however, are evaluated in the
original units, so that the tolerances mean what the user expects. ConicXX does not reconstruct
the unscaled data for this. Every quantity needed is obtained from the scaled residuals and the
cached factors (:doc:`termination`):

* infinity norms of unscaled vectors are weighted norms of scaled ones, for example
  :math:`\|Ax + s - b\|_\infty = \|E^{-1}\hat r_z\|_\infty/\tau`;
* inner products of conjugate pairs are invariant up to the cost scale, for example
  :math:`b^\top z = \tfrac1c\,\hat b^\top\hat z`, because :math:`E^{-1}` and :math:`E` cancel.

The returned solution, residuals and objective are always unscaled.
