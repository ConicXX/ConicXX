Installation
============

ConicXX is a CMake project. It needs

* a C++17 compiler (continuous integration tests GCC and Clang on Linux),
* CMake 3.21 or newer,
* `Eigen <https://eigen.tuxfamily.org>`_ 3.4.

If Eigen is not found with ``find_package(Eigen3)``, it is fetched automatically with
``FetchContent``. GoogleTest, needed only for the unit tests, is handled the same way.

Building from source
--------------------

.. code-block:: sh

   git clone https://github.com/ConicXX/ConicXX.git
   cd ConicXX
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j
   ctest --test-dir build -LE benchmark --output-on-failure

Always build in ``Release`` (or ``RelWithDebInfo``) for anything time-critical: Eigen's
expression templates are slow without optimization.

CMake options
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Option
     - Default
     - Effect
   * - ``CONICXX_BUILD_SHARED``
     - ``OFF``
     - Build a shared instead of a static library.
   * - ``CONICXX_BUILD_TESTS``
     - top level
     - Build the GoogleTest unit tests.
   * - ``CONICXX_BUILD_EXAMPLES``
     - top level
     - Build the programs in ``examples/``.
   * - ``CONICXX_BUILD_BENCHMARKS``
     - top level
     - Build the benchmark suite ``conicxx_benchmarks`` (see :doc:`../benchmarks`).
   * - ``CONICXX_WARNINGS_AS_ERRORS``
     - ``OFF``
     - Treat compiler warnings as errors.
   * - ``CONICXX_INSTALL``
     - top level
     - Generate install and export rules.

"top level" means ``ON`` when ConicXX is the top-level CMake project and ``OFF`` when it is
consumed through ``add_subdirectory`` or ``FetchContent``.

Using ConicXX from another CMake project
----------------------------------------

All three ways below provide the same target, ``conicxx::conicxx``.

**Installed package.**

.. code-block:: sh

   cmake --install build --prefix /path/to/prefix

.. code-block:: cmake

   find_package(conicxx 0.0.1 CONFIG REQUIRED)   # version check is optional
   target_link_libraries(my_app PRIVATE conicxx::conicxx)

Pass ``-DCMAKE_PREFIX_PATH=/path/to/prefix`` when configuring the consumer if the prefix is not a
system location.

**FetchContent.**

.. code-block:: cmake

   include(FetchContent)
   FetchContent_Declare(conicxx
     GIT_REPOSITORY https://github.com/ConicXX/ConicXX.git
     GIT_TAG        main)            # better: a fixed commit or tag
   FetchContent_MakeAvailable(conicxx)
   target_link_libraries(my_app PRIVATE conicxx::conicxx)

**Subdirectory.**

.. code-block:: cmake

   add_subdirectory(external/ConicXX)
   target_link_libraries(my_app PRIVATE conicxx::conicxx)

Public headers
--------------

Only the following headers are installed; everything else in ``src/conicxx/`` is internal.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Header
     - Contents
   * - ``<conicxx/solver.h>``
     - :cpp:class:`conicxx::Solver`; includes all headers below.
   * - ``<conicxx/settings.h>``
     - :cpp:struct:`conicxx::Settings`, :cpp:struct:`conicxx::RegularizationSettings`
   * - ``<conicxx/solution.h>``
     - :cpp:struct:`conicxx::Solution`, :cpp:struct:`conicxx::Info`, :cpp:enum:`conicxx::Status`
   * - ``<conicxx/cone_spec.h>``
     - :cpp:struct:`conicxx::ConeSpec`
   * - ``<conicxx/types.h>``
     - ``Scalar`` (``double``), ``Index`` (``int``), ``Vec``, ``Mat``, ``SparseMat``, ``Triplet``
   * - ``<conicxx/convert.h>``
     - Conversion of dense and row-major sparse matrices to ``SparseMat``
   * - ``<conicxx/version.h>``
     - ``CONICXX_VERSION_{MAJOR,MINOR,PATCH}``, ``CONICXX_VERSION_STRING``, ``conicxx::version()``

Versioning
----------

ConicXX follows `semantic versioning <https://semver.org/>`_. The single source of the version is
the ``project(conicxx VERSION ...)`` call in the top-level ``CMakeLists.txt``.

Building this documentation
---------------------------

The documentation is built with Sphinx; the API reference is extracted from the public headers by
Doxygen, Breathe and Exhale.

.. code-block:: sh

   sudo apt install doxygen               # or your platform's package manager
   pip install -r docs/requirements.txt
   make -C docs html                      # output in docs/_build/html
