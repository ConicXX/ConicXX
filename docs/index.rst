ConicXX
=======

**ConicXX** is a C++17 interior-point solver for convex quadratic conic programs,

.. math::

   \begin{array}{ll}
   \text{minimize}   & \tfrac12 x^\top P x + q^\top x \\
   \text{subject to} & A x + s = b, \\
                     & s \in \K,
   \end{array}

with decision variable :math:`x \in \R^n`, slack :math:`s \in \R^m`, a symmetric positive
semidefinite :math:`P`, and a cone :math:`\K` that is a product of the zero cone (equality
constraints), the nonnegative orthant (linear inequalities) and second-order cones (norm
constraints, e.g. Coulomb friction cones).

It is built on `Eigen <https://eigen.tuxfamily.org>`_ and has no other runtime dependency. The
method is a primal-dual interior-point method on the homogeneous self-dual embedding with
Nesterov--Todd scaling and Mehrotra's predictor--corrector, in the line of ECOS, Clarabel and QOCO.
Each iteration solves one quasidefinite KKT system with a built-in sparse :math:`LDL^\top`
factorization (simplicial or supernodal).

ConicXX was written as the per-timestep contact solver of a nonsmooth multibody simulation
engine. That use shapes its design:

* **Equality constraints are never perturbed.** Joints and bilateral constraints are satisfied
  to the accuracy of the KKT solves; regularization only conditions the factorization and is
  removed by iterative refinement against the exact KKT matrix.
* **Robust on degenerate problems.** Contacts at the apex of the friction cone, pairs that are
  not strictly complementary, redundant equalities, and scenes with 10\ :sup:`4` variables and
  10\ :sup:`4` cone blocks.
* **Honest failure reporting.** Every solve ends with a definite :cpp:enum:`conicxx::Status`
  and diagnostics in :cpp:struct:`conicxx::Info`; there are no silent rescues.
* **Cheap repeated solves.** ``setup()`` once, then ``updateData()`` and ``solve()`` per time step,
  reusing the KKT sparsity pattern, the ordering and symbolic factorization, and a warm start.

.. code-block:: cpp

   conicxx::Solver solver;
   solver.setup(P, q, A, b, cones);           // once: ordering, symbolic factorization
   const conicxx::Solution& sol = solver.solve();
   if (sol.ok()) use(sol.x, sol.z);

The documentation is split into a user guide, which explains how to formulate and solve problems,
and a description of the algorithm, which covers every step the solver takes, with the formulas as
they are implemented.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/problem
   user_guide/usage
   user_guide/settings
   user_guide/results
   user_guide/examples
   user_guide/contact

.. toctree::
   :maxdepth: 2
   :caption: Algorithm

   algorithm/index
   algorithm/hsde
   algorithm/cones
   algorithm/ipm
   algorithm/linear_system
   algorithm/equilibration
   algorithm/termination
   algorithm/warm_start

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api/library_root
   benchmarks
   citing
   references
