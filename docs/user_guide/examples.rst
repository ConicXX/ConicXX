Examples
========

All programs below are in ``examples/`` and are built with ``-DCONICXX_BUILD_EXAMPLES=ON`` (the
default for a top-level build). The output shown is that of the actual programs.

Three cone types
----------------

``examples/quickstart.cpp``, discussed in detail in :doc:`../getting_started/quickstart`.

Friction cone projection in a time loop
---------------------------------------

A single contact with Coulomb friction coefficient :math:`\mu = 0.5`. A desired contact impulse
:math:`f_\text{des}` that changes every time step is projected onto the friction cone,

.. math::

   \min_f \tfrac12\|f - f_\text{des}\|^2 \quad\text{s.t.}\quad (\mu f_n, f_{t,1}, f_{t,2}) \in \Q^3 .

Only :math:`q = -f_\text{des}` changes, so each step is ``updateData()`` plus a warm-started
``solve()``.

.. literalinclude:: ../../examples/friction_contact_demo.cpp
   :language: cpp
   :lines: 15-

.. code-block:: text

   timestep 0: status=Solved f_des=(5.00,1.00,1.00) f=(5.0000,1.0000,1.0000) iters=8
   timestep 1: status=Solved f_des=(4.00,3.00,1.50) f=(4.5416,2.0311,1.0156) iters=9
   timestep 2: status=Solved f_des=(3.00,5.00,2.00) f=(4.5541,2.1141,0.8457) iters=7
   timestep 3: status=Solved f_des=(2.00,7.00,2.50) f=(4.5732,2.1534,0.7691) iters=6

In steps 1 to 3 the desired impulse lies outside the cone and the result lies on its boundary,
:math:`\mu f_n = \|f_t\|`.

Infeasibility certificates
--------------------------

Two linear programs without solution: :math:`x \ge 1,\ x \le 0` is primal infeasible, and
:math:`\min -x` s.t. :math:`x \ge 0` is unbounded (dual infeasible).

.. literalinclude:: ../../examples/infeasibility.cpp
   :language: cpp
   :lines: 8-

.. code-block:: text

   [primal] status = PrimalInfeasible, z = (1.000000, 1.000000), A'z = -6.11e-10, b'z = -1.000000
   [dual]   status = DualInfeasible, x = 1.000000, s = 1.000000, q'x = -1.000000

The primal certificate :math:`z = (1, 1)` proves infeasibility: adding the two constraint rows
with these nonnegative weights gives :math:`0 \cdot x \le -1`. The dual certificate :math:`x = 1`
is a feasible direction along which the objective decreases without bound.

Compliant equality rows
-----------------------

:math:`\min \tfrac12(x_0^2 + x_1^2)` s.t. :math:`x_0 - x_1 = 1` with compliance :math:`c`. The
exact solution is :math:`x_0 = -x_1 = 1/(2 + c)`; the compliance only changes through
``updateData()``.

.. literalinclude:: ../../examples/compliant_rows.cpp
   :language: cpp
   :lines: 10-

.. code-block:: text

   c =  0.0e+00: status = Solved, x = (+0.50000000, -0.50000000), exact x0 = 0.50000000, z = -0.50000000
   c =  5.0e-01: status = Solved, x = (+0.40000000, -0.40000000), exact x0 = 0.40000000, z = -0.40000000
   c =  2.0e+00: status = Solved, x = (+0.25000000, -0.25000000), exact x0 = 0.25000000, z = -0.25000000
   c =  1.0e+03: status = Solved, x = (+0.00099800, -0.00099800), exact x0 = 0.00099800, z = -0.00099800

The multiplier :math:`z = -1/(2+c)` is the negative of the spring force :math:`\pi`.

Larger problems
---------------

The benchmark generators in ``benchmarks/problem_generators.cpp`` build complete instances of
several problem classes and are a good template for larger models: a long-only portfolio
optimization (second-order cones from a factor model), group lasso (many SOC blocks), random
QPs, LPs and SOCPs, and frictional contact problems with thousands of friction cones coupled by
equality chains. See :doc:`../benchmarks`.
