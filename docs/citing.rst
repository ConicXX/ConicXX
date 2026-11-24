License and citing
==================

License
-------

ConicXX is free software, licensed under the
`GNU General Public License v3.0 <https://www.gnu.org/licenses/gpl-3.0.html>`_.

The elimination tree and the up-looking :math:`LDL^\top` factorization follow the algorithm of
`QDLDL <https://github.com/osqp/qdldl>`_ (Apache License 2.0); see the ``NOTICE`` file in the
repository. ConicXX depends on Eigen (Mozilla Public License 2.0); GoogleTest (BSD 3-Clause) is
used for the tests only.

Citing
------

If you use ConicXX in academic work, please cite the repository:

.. code-block:: bibtex

   @software{conicxx,
     author = {Breuling, Jonas},
     title  = {{ConicXX}: an interior-point solver for quadratic conic programs},
     url    = {https://github.com/ConicXX/ConicXX},
     year   = {2026}
   }

Please also cite the methods ConicXX builds on, in particular Clarabel
:cite:p:`goulart2024clarabel` for the quadratic homogeneous embedding and the termination rules.

Related solvers
---------------

* `Clarabel <https://clarabel.org>`_ -- interior-point solver for conic programs with quadratic
  objectives (Rust, Julia) :cite:p:`goulart2024clarabel`.
* `QOCO <https://qoco-org.github.io/qoco/>`_ -- quadratic objective conic optimizer with custom
  solver generation (C) :cite:p:`chari2025qoco`.
* `ECOS <https://github.com/embotech/ecos>`_ -- embedded SOCP solver (C)
  :cite:p:`domahidi2013ecos`.
* `SCS <https://www.cvxgrp.org/scs/>`_ -- first-order conic solver based on operator splitting
  and the homogeneous embedding (C) :cite:p:`odonoghue2016scs`.
* `OSQP <https://osqp.org>`_ -- first-order QP solver (C) :cite:p:`stellato2020osqp`.
