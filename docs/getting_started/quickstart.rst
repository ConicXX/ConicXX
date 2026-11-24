Quickstart
==========

This example solves a three-variable problem that uses all three cone types:

.. math::

   \begin{array}{ll}
   \text{minimize}   & \tfrac12 \|x - c\|_2^2, \qquad c = (1, 1, 0.3), \\
   \text{subject to} & x_0 + x_1 = 1, \\
                     & x_2 \ge 0, \\
                     & x_2 \ge \|(x_0, x_1)\|_2 .
   \end{array}

Expanding the objective gives :math:`P = I` and :math:`q = -c` (the constant
:math:`\tfrac12\|c\|^2` is dropped). Each constraint becomes rows of :math:`A` and :math:`b` such
that the slack :math:`s = b - Ax` lies in the right cone:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Constraint
     - Cone block
     - Slack
   * - :math:`x_0 + x_1 = 1`
     - zero, dimension 1
     - :math:`s_0 = 1 - x_0 - x_1 = 0`
   * - :math:`x_2 \ge 0`
     - nonnegative, dimension 1
     - :math:`s_1 = x_2 \ge 0`
   * - :math:`x_2 \ge \|(x_0,x_1)\|`
     - second-order, dimension 3
     - :math:`(s_2, s_3, s_4) = (x_2, x_0, x_1) \in \Q^3`

The rows must appear in the order zero, nonnegative, second-order. The complete program
(``examples/quickstart.cpp``):

.. literalinclude:: ../../examples/quickstart.cpp
   :language: cpp
   :lines: 9-

With ``verbose = 2`` the solver prints one line per iteration:

.. code-block:: text

   conicxx: n = 3, m = 5 (1 zero, 1 nonneg, 1 soc blocks), nnz(L) = 11
   iter     pcost          dcost         gap      pres     dres     k/t      mu       step    sigma
      0  -7.750000e-01  -2.550000e-01  5.20e-01 5.01e-01 1.13e+00 1.00e+00 2.11e+00  0.0000  0.0000
      1  +5.797887e-02  -1.817592e+00  1.88e+00 1.19e-02 3.43e-02 4.08e-01 2.93e-01  0.9900  0.0087
      2  -5.883405e-01  -9.776179e-01  3.89e-01 3.78e-04 1.04e-03 9.97e-02 4.24e-02  0.9900  0.0087
      3  -7.065923e-01  -7.489852e-01  4.24e-02 5.64e-06 1.47e-05 1.32e-02 3.84e-03  0.9900  0.0013
      4  -7.120435e-01  -7.126096e-01  5.66e-04 5.97e-08 1.53e-07 1.90e-04 5.19e-05  0.9900  0.0004
      5  -7.121311e-01  -7.121368e-01  5.70e-06 5.98e-10 1.53e-09 1.92e-06 5.22e-07  0.9900  0.0000
      6  -7.121320e-01  -7.121321e-01  5.70e-08 6.40e-12 1.53e-11 1.92e-08 5.22e-09  0.9900  0.0000
      7  -7.121320e-01  -7.121320e-01  5.69e-10 5.40e-13 1.53e-13 1.92e-10 5.23e-11  0.9900  0.0000
   conicxx: Solved after 8 iterations, 8 factorizations, 0 dynamic pivots, 0.060 ms
   status    = Solved
   x         = (0.50000000, 0.50000000, 0.70710678)
   objective = -0.71213203
   iters     = 8

The columns are the primal and dual objective, the absolute duality gap, the relative primal and
dual residuals, :math:`\kappa/\tau`, the complementarity measure :math:`\mu`, the step length of
the previous iteration and the centering parameter :math:`\sigma`. All of them are defined in
:doc:`../algorithm/termination` and :doc:`../algorithm/ipm`. The printed objective omits the
dropped constant :math:`\tfrac12\|c\|^2`.

Building the matrices densely
-----------------------------

For small problems it can be easier to write :math:`P` and :math:`A` as dense
``Eigen::MatrixXd`` (``conicxx::Mat``). :cpp:func:`conicxx::Solver::setup` has overloads for dense
input; they convert to ``SparseMat`` and print a warning to stderr, because a dense matrix gives the
solver no sparsity to exploit:

.. code-block:: cpp

   Mat P = Mat::Identity(3, 3);
   Mat A(5, 3);
   A <<  1,  1,  0,
         0,  0, -1,
         0,  0, -1,
        -1,  0,  0,
         0, -1,  0;
   solver.setup(P, q, A, b, cones);  // P: only the upper triangle is read

There is no dense overload of ``updateData()``. Convert once with
:cpp:func:`conicxx::toSparse` / :cpp:func:`conicxx::toSparseUpperTriangular` and reuse the
``SparseMat``.

Next steps
----------

* :doc:`../user_guide/problem` explains how to encode constraints as cone rows.
* :doc:`../user_guide/usage` covers repeated solves, warm starts and compliance.
* :doc:`../user_guide/examples` has further complete programs.
