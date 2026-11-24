Using the solver
================

Everything goes through one class, :cpp:class:`conicxx::Solver`. Its life cycle is

.. code-block:: text

   setup()  ──►  solve()  ──►  updateData()  ──►  solve()  ──►  updateData() ...
     ▲                              │ returns false: the sparsity pattern changed
     └──────────────────────────────┘

None of the methods throws. Invalid input is reported by a ``false`` return value of ``setup()``
or ``updateData()``, and every ``solve()`` ends with a definite :cpp:enum:`conicxx::Status`.

Setup
-----

.. code-block:: cpp

   bool setup(const SparseMat& P, const Vec& q, const SparseMat& A, const Vec& b,
              const ConeSpec& cone_spec, const Settings& settings = Settings{},
              const Vec* zero_compliance = nullptr);

``setup()`` copies and equilibrates the data, builds the sparsity pattern of the KKT matrix,
computes the fill-reducing ordering and the complete symbolic factorization, and allocates every
work vector. It does not factorize. It returns ``false`` (and leaves the solver unusable until the
next successful ``setup()``) if

* ``cone_spec`` is invalid (negative sizes, a second-order cone of dimension 0),
* the dimensions of ``P``, ``q``, ``A``, ``b`` and ``cone_spec.totalDim()`` do not match,
* ``Settings::validate_inputs`` is set and some value is not finite,
* ``zero_compliance`` has the wrong size or a negative or non-finite entry.

``P`` must contain the upper triangle (entries below the diagonal are ignored). Its diagonal
need not be stored; missing diagonal entries are zero.

Repeated solves with new data
-----------------------------

.. code-block:: cpp

   bool updateData(const SparseMat* P, const Vec* q, const SparseMat* A, const Vec* b,
                   const Vec* zero_compliance = nullptr);

``updateData()`` overwrites the numerical values and keeps everything that depends only on the
sparsity pattern. Pass ``nullptr`` for data that did not change. It returns ``false`` and changes
nothing if a matrix does not have **exactly** the pattern given to ``setup()``: the same column
pointers and row indices of the compressed storage. In that case call ``setup()`` again.

To keep the pattern stable:

* Store explicit zeros instead of dropping entries that happen to be zero in one step. A contact
  that may become active can be reserved in the pattern with stored zeros.
* Pass compressed matrices (``makeCompressed()``; ``setFromTriplets()`` already compresses).
* Do not use ``sparseView()`` on changing dense data: it drops exact zeros.
  :cpp:func:`conicxx::toSparse` keeps every entry.

The typical simulation loop:

.. code-block:: cpp

   conicxx::Solver solver;
   solver.setup(P, q, A, b, cones, settings);
   for (int step = 0; step < num_steps; ++step) {
     assemble(P, q, A, b);                                   // same pattern, new values
     if (!solver.updateData(&P, &q, &A, &b)) {               // pattern changed?
       solver.setup(P, q, A, b, cones, settings);            // full rebuild
     }
     const conicxx::Solution& sol = solver.solve();          // warm-started automatically
     if (!sol.ok()) handle(sol.status, sol.info);
   }

Solving
-------

``solve()`` runs the interior-point method and returns a reference to the solver's
:cpp:struct:`conicxx::Solution`. The reference stays valid until the next call of ``solve()`` or
``setup()``; copy the vectors if you need them longer. After the first ``solve()`` on a given
problem size, ``solve()`` does not allocate memory.

Results are always in the units of the original problem: ``x``, ``s``, ``z``, the objective and
all residuals in :cpp:struct:`conicxx::Info` are unscaled. See :doc:`results` for the meaning of
each status and diagnostic.

Warm starts
-----------

With ``Settings::warm_start = true`` (the default) every ``solve()`` that ends ``Solved`` stores its
solution, and the next ``solve()`` starts from it after ``updateData()``. You can also supply a
starting point yourself, for example the contact forces of the previous time step mapped onto the
new contact set:

.. code-block:: cpp

   solver.setup(P, q, A, b, cones, settings);   // clears any stored warm start
   solver.setWarmStart(x0, s0, z0);              // original units, sizes n, m, m
   solver.solve();

``s0`` and ``z0`` need not lie in the cones; they are projected onto them first. A warm start is
used only if it is at least as good as the cold start by the solver's merit function; otherwise the
cold start is used and ``Info::warm_start_used`` is ``false``. ``setup()`` discards a stored warm
start, so call ``setWarmStart()`` after it. The construction of the starting point is described in
:doc:`../algorithm/warm_start`.

Changing settings
-----------------

:cpp:func:`conicxx::Solver::setSettings` replaces the settings. Most settings take effect at the
next ``solve()``. Three groups are used only by ``setup()`` and need a new ``setup()`` to change:

* ``equilibrate`` and the ``equilibrate_*`` parameters,
* ``supernodal``,
* ``validate_inputs`` for ``setup()`` itself (``updateData()`` reads the current value).

Compliance
----------

For compliant equality rows (:ref:`problem formulation <compliant-rows>`),
pass a vector ``c`` of size ``cone_spec.zero_dim`` with :math:`c_i \ge 0`:

.. code-block:: cpp

   Vec c = compliance_of_rows();                     // size zero_dim, >= 0
   solver.setup(P, q, A, b, cones, settings, &c);
   ...
   solver.updateData(&P, &q, &A, &b, &c);           // c may change every step

In ``setup()`` a ``nullptr`` means :math:`c = 0`; in ``updateData()`` it means "unchanged". The
compliance is data, not structure: changing it never requires a new ``setup()``.

Dense input and other formats
-----------------------------

:cpp:func:`conicxx::Solver::setup` has overloads that accept ``P`` and/or ``A`` as dense
``conicxx::Mat``. They are meant for prototyping and print a warning to stderr on every call. ``<conicxx/convert.h>``
provides the conversions explicitly:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Function
     - Purpose
   * - ``toSparse(const Mat&)``
     - Dense to ``SparseMat``, keeping every entry (including zeros) so the pattern is stable.
   * - ``toSparseUpperTriangular(const Mat&)``
     - Dense symmetric to the upper-triangular ``SparseMat`` expected for ``P``; warns if the input
       is not symmetric.
   * - ``toSparse(const SparseMatRowMajor&)``
     - Row-major (CSR) to column-major (CSC). Do this once, not every time step.

Each takes optional ``name`` and ``warn`` arguments to label or suppress the warning.

Threading
---------

A :cpp:class:`conicxx::Solver` holds all of its workspace and has no global state. Distinct
solver objects can be used concurrently from different threads; one object must not be used from
two threads at the same time. ``Solver`` is movable but not copyable.
