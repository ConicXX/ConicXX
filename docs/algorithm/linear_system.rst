KKT system and factorization
============================

Almost all of the work of an iteration is the solution of the reduced system :eq:`eq-reduced`,

.. math::

   K = \begin{bmatrix} P & A^\top \\ A & -H \end{bmatrix}, \qquad
   H = \operatorname{blockdiag}\big(C \text{ on zero rows},\ \diag(s_i/z_i) \text{ on orthant rows},\
   H_1, \dots, H_N \text{ on SOC blocks}\big),

of dimension :math:`n + m`, with the compliance :math:`C` (zero for ordinary equalities) and the
dense NT blocks :math:`H_k = W_k^2` of the second-order cones (:doc:`cones`). Only the lower
triangle is stored. Between iterations only :math:`H` changes; between time steps
(``updateData()``) only the values of :math:`P` and :math:`A`; the sparsity pattern changes only
with ``setup()``.

Quasidefiniteness
-----------------

:math:`P` is positive semidefinite and :math:`H` positive definite on the cone rows, but
:math:`P` may be singular and :math:`H` is zero on equality rows. With a small static
regularization :math:`\delta > 0`,

.. math::

   \tilde K = \begin{bmatrix} P + \delta I & A^\top \\ A & -(H + \delta I) \end{bmatrix}

is *quasidefinite*: its (1,1) block is positive and its (2,2) block negative definite. Every
symmetric permutation of a quasidefinite matrix has an :math:`LDL^\top` factorization with diagonal
:math:`D`, and :math:`D` has exactly :math:`n` positive and :math:`m` negative entries
:cite:p:`vanderbei1995quasidefinite`. This allows a fill-reducing ordering chosen once from the
sparsity pattern alone, with no numerical pivoting, so the symbolic factorization is reused across
all iterations and time steps. The same approach is used by OSQP, ECOS, Clarabel and QOCO
:cite:p:`stellato2020osqp,domahidi2013ecos,goulart2024clarabel,chari2025qoco`.

Symbolic analysis (``setup()``)
-------------------------------

1. **Pattern.** The pattern of :math:`K` is assembled from :math:`P` (upper triangle in, lower
   triangle out, with a diagonal slot for every row), :math:`A`, and :math:`H`: a dense lower
   triangle for each SOC block and only a diagonal for zero and orthant rows. Index maps from every
   stored value of :math:`P`, :math:`A` and :math:`H` to its slot in :math:`K` make the later value
   updates a scatter, without searching.
2. **Ordering.** Approximate minimum degree (AMD) :cite:p:`amestoy1996amd` on the pattern of
   :math:`K` (Eigen's implementation).
3. **Elimination tree and column counts** of :math:`L` for the permuted matrix, and from them the
   complete row structure of :math:`L` :cite:p:`davis2006direct`.
4. **Supernodes** (if used, see below).
5. **Workspace.** All arrays for the numeric factorization, the solves and the refinement are
   allocated here; the hot path does not allocate.

Numeric factorization
---------------------

Two implementations compute the same factor up to roundoff.

**Simplicial (up-looking).** Row :math:`k` of :math:`L` is computed by a sparse triangular solve
with the rows computed before, as in QDLDL. Because the row structure of :math:`L` is known from
the symbolic phase, each row is a loop over a precomputed index list, without elimination-tree
walks or marker arrays.

**Supernodal (left-looking).** The elimination tree is postordered (same fill) and fundamental
supernodes are formed: column :math:`j+1` joins the supernode of column :math:`j` if it is the
parent of :math:`j` and its pattern is that of :math:`j` without row :math:`j+1`. The columns of a
supernode then form a dense block. A supernode gathers the updates
:math:`L_d D_d L_d^\top` of all earlier supernodes :math:`d` that reach into its columns, then
factorizes its diagonal block in panels of 32 columns (left-looking within a panel, one matrix
product per panel for the trailing columns). Updates from supernodes narrower than 4 columns use
scalar loops, wider ones dense matrix products. Pivot signs and regularization are handled exactly
as in the simplicial code.

The supernodal path is used if ``Settings::supernodal`` is set and :math:`L` has at least 16
entries per column on average. On very sparse factors, typical of contact problems with small
blocks, the per-supernode overhead outweighs the dense kernels and the simplicial path is faster.

Regularization
--------------

Regularization makes the factorization well defined; it never changes the system that is solved
(see *Iterative refinement* below).

**Static regularization** shifts each pivot in the direction of its expected sign
(:math:`+1` for the :math:`n` rows of :math:`x`, :math:`-1` for the :math:`m` cone rows): the
elimination of row :math:`i` starts from :math:`K_{ii} + \operatorname{sign}_i\,\delta_i` with

.. math::

   \delta_i = \delta_{\text{block}(i)} + \varepsilon_\text{prop}\max_j |K_{jj}|
   \;\big[ + \varepsilon_\text{soc}\max_{j \in \text{block } k}|(H_k)_{jj}| \ \text{ on the rows of SOC block } k \big],

where :math:`\delta_\text{block}` is ``static_P``, ``static_zero``, ``static_nonneg`` or
``static_soc`` (all :math:`10^{-8}`), :math:`\varepsilon_\text{prop}` = ``static_proportional``
(:math:`\varepsilon^2`) and :math:`\varepsilon_\text{soc}` = ``static_soc_proportional``
(:math:`10^{-12}`). The shift is applied to the pivots only and is never written into the matrix
used for refinement.

The **block-relative SOC term** handles the nearly singular SOC blocks of contact problems. Near
the cone boundary, the eigenvalues of :math:`H_k` spread like :math:`1/\mu^2`; eliminating the
block then loses digits relative to its largest entry, and the remaining pivots of the block become
pure roundoff. Without the block-relative term they eventually collapse (many dynamically
regularized pivots in one factorization, non-finite refinement, ``NumericalError``) once
:math:`\mu` is around :math:`10^{-15}`. Regularizing relative to the block's own scale keeps them
defined; values between :math:`10^{-15}` and :math:`10^{-12}` behave alike, larger ones perturb the
Newton directions noticeably.

**Dynamic regularization** :cite:p:`goulart2024clarabel` is a backstop for roundoff and exact
cancellation: a pivot :math:`d_k` with :math:`\operatorname{sign}_k d_k <` ``dynamic_eps``
(:math:`10^{-13}`) is replaced by :math:`\operatorname{sign}_k\cdot` ``dynamic_delta``
(:math:`2\cdot10^{-7}`). The replacement happens inside the numeric pass, before the pivot is used
for the remaining rows, so the result is the exact factorization of :math:`K + \diag(e)` with
:math:`e_k \ne 0` only at replaced pivots, and the inertia is always :math:`(n, m)`.
(Replacing pivots *after* the factorization would not be a diagonal perturbation: the entries of
:math:`L` below pivot :math:`k` were computed with the old value.) The number of replaced pivots is
reported in ``Info::kkt_dynamic_reg_pivots``. With the static regularization above it is normally
zero; nonzero counts point to a nearly singular KKT matrix.

Iterative refinement
--------------------

Every solve is refined against the exact, unregularized matrix :math:`K`:

.. code-block:: text

   x = solve(L D L', b)
   r = b - K x
   repeat at most refine_max_iter times while ||r||_inf > abstol + reltol * ||b||_inf:
       x_trial = x + solve(L D L', r)
       r_trial = b - K x_trial
       if ||r_trial|| >= ||r||: stop              (keep x: refinement must not make it worse)
       accept x_trial
       if ||r|| / ||r_trial|| < stop_ratio: stop  (slow convergence)

with ``refine_abstol`` = :math:`10^{-12}`, ``refine_reltol`` = :math:`10^{-13}`,
``refine_stop_ratio`` = 5 and ``refine_max_iter`` = 10 (Clarabel's values). Residuals are formed
in the permuted ordering, so no permutation is applied inside the loop. The factorization of
:math:`\tilde K` is thereby used only as a preconditioner; the converged solution solves the
system with :math:`K`, so the regularization does not perturb the Newton direction. In practice
about one refinement step per solve is needed.

Equality constraints
--------------------

The first priority of ConicXX is that equality (zero-cone) rows are not perturbed. Two properties
together guarantee this:

* The equality rows of :math:`K` have the exact entries :math:`A_E` and :math:`-C` (zero for rigid
  equalities). The static regularization ``static_zero`` is needed: without it, an ordering that
  eliminates an equality row before its neighbours would meet an exactly zero pivot. But it lives
  only in the factorization, and refinement removes its effect, so every Newton direction satisfies
  the linearized equalities :math:`A_E\Delta x - C\Delta z_E - b_E\Delta\tau = -d_{z,E}` to the
  refinement tolerance.
* The interior-point method itself never relaxes equality rows: their slack is identically zero,
  and their residual is reduced by the exact factor :math:`1 - \alpha(1-\sigma)` per step
  (:doc:`hsde`).

If a refined solve still leaves an equality row with a residual above
:math:`10^{-8}\max(1, \|\text{rhs}\|_\infty)`, the equalities are inconsistent or numerically
dependent, and ``Info::equality_rank_deficient`` is set. Consistent duplicated equalities (the
common case of redundant joints) are solved normally.

Compliant rows
--------------

A compliance :math:`C` on the zero-cone rows enters :math:`K` as :math:`-C` on their diagonal.
This is the result of eliminating the extra variables :math:`\pi` of the augmented formulation
(:ref:`compliant-rows`) from its KKT matrix:

.. math::

   \begin{bmatrix} P & 0 & A_E^\top & \cdot \\ 0 & C & C & 0 \\ A_E & C & 0 & 0 \\ \cdot & 0 & 0 & -H \end{bmatrix}
   \;\longrightarrow\;
   \begin{bmatrix} P & A_E^\top & \cdot \\ A_E & -C & 0 \\ \cdot & 0 & -H \end{bmatrix} .

The compact form has fewer variables and a better conditioned pivot for stiff elements: in the
augmented form the pivot of :math:`\pi_i` is :math:`C_i` plus the static regularization, which
dominates for :math:`C_i \lesssim 10^{-8}`.

Cost and allocation
-------------------

``setup()`` does all symbolic work and all allocation. ``updateData()`` scatters new values of
:math:`P` and :math:`A` into :math:`K` through the index maps and checks that the patterns are
unchanged by comparing the compressed index arrays. Each iteration scatters the new :math:`H`,
computes the static regularization (one pass over the diagonal), and runs one numeric
factorization. After the first ``solve()`` the hot path performs no heap allocation, which the test
suite verifies with allocation hooks.
