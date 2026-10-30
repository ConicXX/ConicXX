# Design notes

This file collects rationale that's too long for a source comment, per `CONICXX_AGENT_TASKS.md`'s
T0.5 (moving long design commentary out of source files) and T4.1 (documenting deviations from
Clarabel's reference behavior). It covers Phase 4 (termination), Phase 9 (KKT factorization) and the background for Phases 11 and 12.

## Termination and infeasibility (Phase 4, T4.1/T4.2)

### Where the formulas come from

`CONICXX_AGENT_TASKS.md`'s T4.1/T4.2 give simplified, plain-English formulas and explicitly say
to compare them against Clarabel's actual source
(`Clarabel.rs/src/solver/implementations/default/info.rs` and `residuals.rs`) and document any
deviation. Two real deviations were found and resolved in Clarabel's favor (confirmed with the
maintainer before implementing):

1. **`tol_ktratio`'s role.** The task text says "convergence requires κ/τ ≤ `tol_ktratio`"
   (default 1e-6). Clarabel's actual `check_convergence` does not gate `Solved` on `tol_ktratio`
   at all -- it requires a fixed `ktratio <= 1.0` sanity bound instead. `tol_ktratio` is used only
   to decide when infeasibility certificates are even considered:
   `ktratio > (1/tol_ktratio) * 1000`. Implemented here to match Clarabel exactly
   (`SolverImpl::solve()`'s `infeas_ktratio_gate`, and the mirrored `reduced_tol_ktratio` gate for
   the "almost" statuses) -- the task's literal reading would make `Solved` roughly 10⁶× stricter
   on that one gate than any production solver actually is, and isn't something a paraphrase can
   safely stand in for on the single most safety-critical part of the solver.
2. **Gap check structure.** The task's formula is a single additive check,
   `gap_abs <= tol_gap_abs + tol_gap_rel * min(|pobj|, |dobj|)`. Clarabel actually uses an OR of
   two independent checks: `(gap_abs < tol_gap_abs) OR (gap_rel < tol_gap_rel)`, where
   `gap_rel = gap_abs / max(1, min(|pobj|, |dobj|))`. Implemented Clarabel's OR form
   (`SolverImpl::isSolved`). Both agree that the denominator uses `min`, not `max` -- the
   pre-Phase-4 code used `max`, which was a real bug (made the relative-gap tolerance easier to
   satisfy than intended whenever `|pobj|` and `|dobj|` differ).

Everything else (infinity-norm residuals, the `max(1, ...)` normalization denominators, the
`rx_inf`/`Px`/`rz_inf`-based infeasibility certificate residuals, `tol_infeas_abs`/`tol_infeas_rel`
as a pair) matches the structure of both the task's formulas and Clarabel's actual code.

### Unscaling without reconstructing the problem

`KktSystem`/`SolverImpl` work entirely in equilibrated ("hat") units after `setup()` (see
`equilibration.h`): `P_hat = c*D*P*D`, `q_hat = c*D*q`, `A_hat = E*A*D`, `b_hat = E*b`, with
`x = D*x_hat`, `s = E^-1*s_hat`, `z = (1/c)*E*z_hat`. T4.1 requires evaluating termination in
*unscaled* units. Rather than reconstruct unscaled `P`/`A`/`q`/`b` every iteration (defeating much
of the point of equilibrating in the first place), every unscaled quantity needed for termination
is obtained algebraically from the already-computed equilibrated residual cache
(`rx_`, `rz_`, `Px_`, `dot_qx_`, `dot_bz_`, `dot_xPx_`) and the cached equilibration weights
(`d_eff_`, `e_eff_`, `dinv_eff_`, `einv_eff_`, `cinv_` -- always valid, identity when
`Settings::equilibrate` is off). The derivation, worked from conicxx's own documented `D`/`E`/`c`
convention (not assumed to match Clarabel's internal Rust variable-by-variable scaling, which
wasn't independently verifiable from source -- see below):

- `||v||_inf` for any unscaled vector `v` that's `D`-, `E`-, or `E^-1`-weighted from its stored hat
  form is `max_i |weight_i * v_hat_i|` (`SolverImpl::weightedInfNorm`) -- no need to materialize
  `v` itself.
- Unscaled dual residual `r_d = Px + A'z + q`: since `rx_ = -(A'z + Px + tau*q)` is already exactly
  `-tau * D * r_d_hat`-related, `||r_d||_inf = cinv * weightedInfNorm(rx_, dinv) / tau`.
- Unscaled primal residual `r_p = Ax + s - b`: `rz_ = Ax_hat + s_hat - tau*b_hat` directly gives
  `||r_p||_inf = weightedInfNorm(rz_, einv) / tau` (no `cinv` factor -- verified, see below).
- Unscaled dot products `b'z` and `q'x` are simply `cinv * dot_bz_` and `cinv * dot_qx_`: the row
  scaling `E` (resp. `D`) cancels exactly in a dot product against its own conjugate variable
  (`(E^-1 b_hat)' * (E z_hat) = b_hat' z_hat` since `E^-1 E = I`), leaving only the objective scale
  `c`. This is *not* a per-component reweighting like the norms above -- easy to get wrong by
  reaching for the same `weightedInfNorm` pattern out of habit; don't.
- Infeasibility-certificate residuals (`res_primal_inf`, `res_dual_inf`) use the same weights but
  evaluate `x_`/`z_` directly (not `x_/tau`, `z_/tau`): a certificate is meaningful exactly when
  `tau -> 0`, where tau-normalizing would blow up for no reason. `rx_inf = -A'z = rx_ + Px_ +
  tau*q_` and `rz_inf = Ax+s = rz_ + tau*b_` are recovered from the already-computed residual
  cache rather than a fresh matrix-vector product.

**On `Px`'s `cinv` factor specifically:** an initial WebFetch-based read of Clarabel's
`info.rs` appeared to show `res_dual_inf`'s `Px` term *without* a `cinv` multiplication, while this
derivation (and the analogous `res_dual` derivation, which does the same unscaling to a
`Px`-containing sum) says it needs one. Given a fetched/summarized read of Rust source is not
reliable enough evidence to trust over a from-scratch derivation grounded in this codebase's own
documented convention -- and getting an infeasibility certificate's scaling wrong is exactly the
"false certificate" failure mode the task's own accept criteria rule out -- the derivation above
was implemented and validated empirically instead of copying the fetched formula as-is:
`test/solver/test_termination_unscaled.cpp` reconstructs `res_primal`/`res_dual` directly from the
public API (original unscaled `P`/`A`/`q`/`b`, returned `Solution::x/s/z`) on a deliberately
badly-scaled problem and confirms they match the internally-computed values -- this exercises the
identical `cinv`/`dinv` weighting `Px_` gets in `res_dual` (since `rx_` contains `-Px_` as a
summand), which is strong indirect evidence for `res_dual_inf`'s identical treatment of `Px_`
alone. A direct test of `res_dual_inf`'s `Px` term specifically wasn't constructed, because dual
infeasibility requires `P` to be exactly singular along the unbounded direction by definition --
any instance built to trigger the certificate has `Px ~ 0` in exactly the direction being tested,
regardless of whether the `cinv` scaling is right or wrong.

### Best iterate and "almost" statuses (T4.3)

`SolverImpl::updateBestIterate()` snapshots `(x_, s_, z_, tau_, kappa_)` plus that iterate's
`Metrics` and `mu` whenever `merit = max(res_primal, res_dual, |gap_abs|)` improves. On
`MaxIterations`/`MaxTime`/`InsufficientProgress`, the best snapshot (not the last iterate) is
restored before finalizing, and upgraded to `AlmostSolved`/`AlmostPrimalInfeasible`/
`AlmostDualInfeasible` if it meets `Settings::reduced_tol_*`.

## KKT factorization (Phase 9)

### Algorithm

`QuasiDefiniteLdlt` (`src/conicxx/kkt/quasidefinite_ldlt.{h,cpp}`) factorizes the KKT matrix

```
K = [ P   A' ]      Hs = blockdiag(0 on zero cones, W'W on orthant and SOC blocks)
    [ A  -Hs ]
```

once per IPM iteration with QDLDL's up-looking LDL^T and an AMD ordering. Everything that depends only
on the sparsity pattern is computed once in `analyzePattern()`: the ordering, the permuted upper
triangle and the map from K's entries into it, the elimination tree, the column counts and the complete
row structure of L. `factorize()` is one numeric pass: it scatters K's values into the permuted copy
and computes row k of L by a sparse triangular solve over the stored row pattern (columns ascending),
without elimination-tree walks, marker arrays or writes to L's index arrays.

The interior-point extensions are those of Clarabel's and QOCO's modified QDLDL:

1. **Expected pivot signs.** +1 for P rows, -1 for cone rows. A pivot with `D*sign < dynamic_eps`
   (1e-13) is replaced by `sign*dynamic_delta` (2e-7), Clarabel's rule. The inertia is therefore always
   (n, m); `Info::kkt_dynamic_reg_pivots` counts the replacements.
2. **Static regularization on the pivots only.** Row i starts from `K_ii + sign_i*reg_i` with
   `reg_i = static_<block> + static_proportional*max|diag(K)|`. It is never written into the matrix copy
   used for refinement. With it on every row the factorized matrix is quasidefinite, so every symmetric
   ordering factorizes with the expected signs (Vanderbei 1995); dynamic regularization is only a
   backstop for roundoff and exact cancellation.
3. **Iterative refinement against the exact K** in the infinity norm with Clarabel's tolerances (abs
   1e-12, rel 1e-13, stop ratio 5, at most 10 steps). A correction is kept only if it reduces the
   residual; the previous loop added every correction before checking, so a diverging refinement could
   make the solution worse. Residuals are formed with the permuted copy, so no permutation is applied
   inside the refinement loop.

### Equality rows

Phase 2 gave zero-cone rows zero static regularization so that equalities are never perturbed. Every
static regularization in ConicXX was already factor-only (refinement targets the exact K): it never
perturbed the equalities of a Newton direction, it only decides how good a preconditioner the
factorization is. Without it, an AMD ordering that eliminates an equality row before its x-neighbours
meets an exactly zero pivot. The old code then ran a retry ladder of 18 refactorizations that bumped only
the P, orthant and SOC diagonals, which cannot change that pivot, followed by a zero-row fallback: 19
factorizations per KKT update on every problem with equality rows (133 for 5 iterations on
FrictionChainXL, 152 for 6 on ApexContacts).

Now `static_zero = 1e-8` (as in Clarabel and QOCO) is applied in the factorization only.
`KktSystem.NewtonDirectionSatisfiesLinearizedEqualitiesExactly` checks `A_E dx = r_E` to 1e-12.
`Info::equality_rank_deficient` is set when refinement leaves an equality row with a residual above
`1e-8*max(1, ||rhs||_inf)`: inconsistent or numerically dependent equalities. Consistent duplicated
equalities are solved normally.

### Removed in Phase 9

Eigen::SimplicialLDLT, QDLDL and RegularizedLdlt backends; `Settings::linear_solver`;
`backend_comparison.h`; the K_exact/K_fact split; the retry ladder (`factorizeWithRetry()`, up to about
2e10) and `escalateAndResolve()` (x1000 per attempt); `static_zero_factor_only`; `dynamic_on_zero_rows`;
the QDLDL dependency. The last one also fixes `find_package(conicxx)` for installed builds, which
referenced the non-exported target `qdldl::qdldlstatic`. `SparsityMap` builds K with a counting sort
instead of two `std::map`s.

### Measurements

Same machine, `main` before Phase 9 against Phase 9, minimum of 5 runs, warm start off, times in ms.

| Problem | setup | solve | updateData + solve |
|---|---|---|---|
| FrictionChainXL 264 | 82.9 -> 32.2 | 83.5 -> 52.8 | 89.1 -> 49.1 |
| ApexContacts 64 | 0.58 -> 0.16 | 0.75 -> 0.42 | 0.78 -> 0.42 |
| FrictionChain 64 | 0.48 -> 0.17 | 0.51 -> 0.48 | 0.52 -> 0.50 |
| RandomQP 200 | 110.7 -> 16.8 | 144.0 -> 116.0 | 153.7 -> 115.1 |
| RandomSOCP 50x4 | 41.3 -> 3.8 | 43.1 -> 33.7 | 49.0 -> 33.7 |
| GroupLasso 60x5 | 36.0 -> 6.0 | 62.8 -> 51.1 | 67.0 -> 51.9 |

Factorizations per solve: before, iterations + 2 without equality rows and 19 x (iterations + 2) with
them; now iterations + 1 everywhere. The iteration counts of all benchmark instances are unchanged.

## Cost of an IPM iteration and "one factorization per time step"

### Where the time goes

One IPM iteration is one factorization and three KKT solves (constant, affine, combined). In the IPM a
solve needs about one refinement step (0.6 to 1.4 per solve across the benchmark families). Per call,
identity scaling:

| Problem | N | nnz(L) | factorize | solve, no refinement | solve + 1 refinement step | factorization share |
|---|---|---|---|---|---|---|
| FrictionChainXL 264 | 40,392 | 160,502 | 1.25 ms | 0.90 ms | 1.75 ms | 19 % |
| FrictionChain 64 | | | 0.008 ms | 0.006 ms | 0.010 ms | 20 % |
| RandomQP 200 | 600 | 99,900 | 8.30 ms | 0.33 ms | 0.67 ms | 81 % |
| RandomSOCP 50x4 | 400 | 59,900 | 5.14 ms | 0.18 ms | 0.34 ms | 83 % |
| GroupLasso 60x5 | 720 | 47,010 | 3.98 ms | 0.17 ms | 0.45 ms | 75 % |

Factorization share = factorize / (factorize + 3 x solve with one refinement step). On sparse,
contact-like KKT matrices the triangular solves and refinement residuals dominate, not the
factorization; on denser problems the factorization dominates.

### Why the full KKT cannot be factorized only once per time step

Hs changes in every IPM iteration, in every cone block, and a Newton direction needs a solve with the
current K. A factorization of an older K is only a preconditioner, and it gets worse as the method
converges: the scalings of active cones go to 0 or infinity, exactly where the solves must be accurate.
Exactly one factorization per time step therefore needs a different solver class (ADMM or
Douglas-Rachford with a fixed penalty, OSQP-style), which reaches moderate accuracy with many cheap
solves.

### What can be done (Phase 11 in CONICXX_AGENT_TASKS.md)

1. **Remove wasted work.** Check termination before refactorizing (one factorization and one solve per
   `solve()`); use a warm start without also computing the cold start (one factorization and two
   solves); refine only as accurately as the current iteration needs (inexact Newton); solve the
   constant and the affine right-hand side together.
2. **Static/dynamic split.** Order every row that is constant within a time step first (P, zero-cone
   rows, later compliant rows: block S), the cone rows last (block C). Only `K_CC = -(Hs + delta)`
   changes between iterations. In the up-looking factorization the rows of S never change, and the
   contribution of the S columns to every row of C is constant: it is `G = K_CS K_SS^-1 K_SC` on the
   trailing pattern. Factorize S and form G once per time step, then factorize only the contact-space
   matrix `-(Hs + delta + G)` per iteration. G is the generalized Delassus operator (contact Jacobian
   times the inverse of the joint-constrained, compliant mass matrix times its transpose).
   - It wins when the static part is expensive and contacts couple locally: rigid bodies with joints,
     piles, the domino scene.
   - It loses when many contacts share one flexible body, because G is then dense on those contacts. In
     FrictionChainXL the banded equality chain couples all 792 contact rows: a dense 792 x 792
     factorization (about 1.7e8 flops) per iteration against 1.25 ms for the sparse KKT.
   - Both orderings' flop counts follow from the symbolic analysis, so `analyzePattern()` can choose
     automatically from a cost model.
3. **Factorization reuse with a Krylov method** (GMRES/MINRES preconditioned by an older factorization,
   refactorize when the iteration count grows). Research item: weak close to the solution, and on sparse
   contact matrices it trades a cheap factorization for more of the expensive solves.
4. **Gondzio's multiple centrality correctors** reuse the factorization (one solve each) to save
   iterations; they pay off only where the factorization dominates.

### One factorization per IPM iteration, not per solve and not per time step

The question comes up regularly, because the QOCO and Clarabel papers speak of a fixed KKT structure.

- **Within one iteration the matrix is constant.** All right-hand sides of an iteration (the constant
  one (-q, b) for the tau elimination, the affine predictor, the combined corrector, and further
  correctors if any) are solved with one factorization of `K_k = [P + delta, A'; A, -(W_k'W_k + delta)]`.
- **Across iterations it is not.** W_k is the NT scaling of the current (s_k, z_k), so the (2,2) block
  changes in every iteration and every cone block. What is constant is the sparsity pattern: ordering,
  elimination tree and the row structure of L are computed once per pattern (`analyzePattern()`), and
  every iteration is one numeric pass. "Constant" in those papers refers to the pattern.
- **Why not keep the values.** The Newton direction linearizes `s o z = sigma mu e` at the current
  point. Near the solution the scaling of an active cone goes to 0 and that of an inactive one to
  infinity; an old factorization is a poor approximation exactly where accuracy matters. A frozen
  matrix turns the method into a different one (see the alternatives below).
- **Where "many factorizations per update" came from.** Before Phase 9 a zero pivot on an equality row
  started a retry ladder that refactorized with larger regularization, up to 19 times per update. That
  was a defect, not part of the method. Now: one factorization for the cold start (identity scaling),
  one per iteration, none in the converged iteration; an accepted warm start saves the first one.
- **Dynamic regularization is part of the single numeric pass.** In the up-looking factorization the
  pivot d_k is replaced by `sign_k * delta` before D_k and 1/D_k are stored and used for the remaining
  rows. The result is the exact factorization of a diagonally perturbed matrix,
  `L D L' = K + diag(e)` with e_k != 0 only at replaced pivots, and refinement against K removes the
  perturbation from the solution. Changing D only after the factorization would be something else:
  the entries of L below pivot k were computed with the old d_k, so L D L' would differ from K by a
  dense update of the trailing block, not by a diagonal perturbation, and the inertia guarantee would
  be lost.

### Alternatives to one factorization per IPM iteration

**Rank-one / rank-k updates (Davis-Hager update/downdate) do not help inside an IPM.** They pay when few
entries change. Between two iterations the 3x3 scaling block of every cone changes, and none settles: a
separating contact has z -> 0 with s fixed (its block grows with 1/mu), a sticking one s -> 0 (its block
goes to 0). That is a rank-3*(#contacts) change per iteration, about 40,000 for the domino scene; each
rank-1 update costs about one column path of L, so the updates cost far more than refactorizing.
"Partial updating" from LP IPMs (refresh only entries whose ratio changed by more than a factor and
repair the error with a Krylov method) gives up the exact Newton direction and only pays in early
iterations. Low-rank updates make sense only for the part that is constant within a time step (the
contact-space matrix of a static/dynamic split, when contacts appear or disappear), and even there the
Jacobians of jointed rigid bodies move with the configuration in every step.

**Methods with truly one factorization per time step** leave the IPM class:

1. **ADMM / Douglas-Rachford (OSQP, SCS).** The matrix `[P + sigma I, A'; A, -I/rho]` contains no cone
   scaling: one factorization, then per iteration a triangular solve and a cheap SOC projection. Linear
   convergence, tens to hundreds of iterations for 1e-4 to 1e-6 depending on rho and conditioning;
   adapting rho needs a (rare) refactorization; warm starts are fully usable because there is no
   central path. Anderson acceleration (SCS) helps. CardilloCxx's projected Jacobi solver (bilateral
   system factorized once, projected iteration on the contacts) belongs to this family.
2. **Hybrid / crossover:** a few ADMM iterations from the warm start, then one to three Newton steps
   (IPM or semismooth) for the last digits; factorizations only in the Newton phase.
3. **Frozen factorization as preconditioner** (inexact IPM with MINRES/GMRES): one factorization per
   step, Krylov iterations per IPM iteration. Weak near convergence for the reason above; research.

**Faster IPMs that stay robust:**

1. **No re-setup when the contact set changes** (caller side): a superset pattern (contact rows of the
   broadphase candidates; inactive contacts as rows with stored zeros or a large gap) keeps the
   symbolic analysis, saves the cold start computed for the warm-start comparison and keeps the
   solver's own warm start.
2. **Reduced systems** (T8.1, T11.5). With P = M diagonal the cone rows can be eliminated block-wise:
   body space `M + sum_i A_i' H_i^-1 A_i` (6 dofs per body, sparsity of the contact graph, positive
   definite, Cholesky without pivot signs; ill-conditioned near convergence as in LP normal equations,
   handled by regularization and refinement), or contact space (Delassus operator plus H, T11.5) when
   contacts are few relative to the dofs. The symbolic analysis can choose by flop count. A
   constrained ordering (contact rows first) of the full KKT produces the body-space factor as its
   trailing block; whether AMD already finds this order is not measured yet.
3. **Supernodal or multifrontal factorization** (BLAS-3 on dense column blocks; CHOLMOD, MA57, Pardiso,
   cuDSS out of scope). `QuasiDefiniteLdlt` is simplicial, i.e. scalar code.
4. **Gondzio's multiple centrality correctors**: one extra solve each with the existing factorization;
   pays where the factorization dominates the iteration.

**A non-IPM alternative for time stepping: semismooth Newton** on the natural map or Fischer-Burmeister
(Alart-Curnier in Siconos, so-bogus, non-smooth Newton in PhysX/Warp). It refactorizes every iteration
too, but has no central path, so the warm start from the previous step is used in full (typically 2 to 5
iterations are reported), and it can solve exact (non-associated) Coulomb friction, which no convex
conic method can. Needs globalization (line search, or a fallback to projected Gauss-Seidel/Jacobi).

## Compliant rows (Phase 12 in CONICXX_AGENT_TASKS.md)

### Current CardilloCxx formulation

CardilloCxx (documentation, chapter "Interior-Point Solvers") keeps the percussions of springs and
dampers as primal variables, x = (u, pi_g, pi_gamma):

```
P    = blkdiag(M, C_g, C_gamma)
A_eq = [ W_g'      C_g   0       ]      (zero cone)
       [ W_gamma'  0     C_gamma ]
```

with the scaled compliances C (diagonal) and pi = -y at the solution. Perfect constraints (C_i = 0) also
get a pi_i, which then has a zero row and column in K and is not determined.

### Elimination inside the KKT

ConicXX's KKT for this problem (regularization omitted):

```
[ M     0     W_g    G_u' ]
[ 0     C     C      0    ]      pi rows
[ W_g'  C     0      0    ]      equality rows
[ G_u   0     0     -Hs   ]
```

Eliminating the pi rows (pivot block C) subtracts C C^-1 C = C from the equality diagonal:

```
[ M     W_g    G_u' ]
[ W_g'  -C     0    ]
[ G_u   0     -Hs   ]
```

A compliant row is an equality row with s_i = -C_i z_i instead of s_i = 0; C_i -> 0 recovers a perfect
constraint continuously.

### What it saves

- **Factorization: little.** Minimum degree eliminates the degree-1 pi nodes first without fill; what
  remains is the compact matrix.
- **Variables:** n drops by the number of bilateral rows, perfect constraints included; all O(n) vector
  work and the refinement products shrink accordingly.
- **Conditioning:** in the augmented form the pi pivot is C_i + static_P. For stiff elements with C_i of
  the order of static_P (1e-8) or below, the regularization dominates the factorized compliance and
  refinement must recover a quantity divided by C_i. In the compact form the pivot of a compliant row is
  dominated by the Schur complement W M^-1 W' + C.
- **Data:** P = M and A_eq = [W_g'; W_gamma'], no extra columns to assemble.
- **Static/dynamic split:** compliant rows belong to the static block S.

### Implementation (Phase 12)

Every Newton step of the augmented problem keeps pi = -z_E: the pi rows of the KKT read
C (d_pi + d_z_E) = 0 for every right-hand side the IPM uses, because their dual residual vanishes on
that manifold. The compact method therefore produces the same iterates:

1. **API:** `Solver::setup(..., settings, const Vec* zero_compliance)` and
   `Solver::updateData(P, q, A, b, const Vec* zero_compliance)`; c has size `zero_dim`, entries >= 0,
   nullptr means 0 in `setup()` and "unchanged" in `updateData()`. The compliance is data, not structure,
   so `ConeSpec` is unchanged (CardilloCxx's C depends on the step size and theta).
2. **KKT:** `K_ii = -C_i` on the zero-cone rows in the exact matrix (`KktSystem::setZeroCompliance()`);
   static regularization stays factor-only on top.
3. **Residuals:** `rz_E = A_E x - C z_E - tau b_E` (s_E stays 0); `C z_E` is cached per iteration.
4. **Quadratic forms:** `dot_xPx_` includes `z_E'C z_E` (rtau, objectives, metrics); the tau elimination
   gains `2 zeta'C z_vE` in the numerator and `zeta'C zeta - 2 zeta'C z1_E` in the denominator
   (`zeta = z_E/tau`; the `z1_E'C z1_E` terms cancel); the QP branch of the cold start is used when some
   C_i > 0; the infeasibility residuals include `||C z_E||` (A'z and Px of the eliminated columns).
5. **Equilibration:** `C_hat = E C E / c`. |C_hat_i| does not count in the row norm: like Hs it belongs
   to the (2,2) block of K, which the Ruiz scaling of `[P A'; A 0]` leaves out (as Clarabel). With it
   in the norm, slinky needed 12.8 iterations per step; without it 11.9 (augmented form: 12.1).

Tests (`test_compliant_rows.cpp`): the compact solution satisfies the KKT conditions of the augmented
problem to 1e-8 and matches its solution to 1e-4 (an IPM iterate is accurate to about sqrt(mu) at
degenerate apex contacts, so a tighter comparison of two solutions measures the stopping rule, not the
formulation), with iteration counts within 2; C = 0 is identical to no compliance; C = 1e-12 converges;
`updateData()` with a new C matches a fresh `setup()`; negative or mis-sized C is rejected. Without
compliance the benchmark iteration counts are unchanged.

## IPM safeguards and fewer factorizations per solve (Phase 10a)

### Centrality measured at the trial point

The step-length safeguard accepts a trial point only if every complementarity pair satisfies
`centrality >= centrality_theta * mu_trial`. Before, the cone pairs were measured through
`lambda = W z_trial` with the scaling W of the current point, which is not a property of the trial
point. Now `ConeBase::centrality(s, z)` returns the smallest eigenvalue of `lambda o lambda` for the NT
scaling point of the trial pair itself, computed from s and z:

- orthant: `min_i s_i z_i`;
- SOC: `lambda'lambda = s'z` and `det(lambda) = sqrt(det(s) det(z))` give
  `(lambda0 - ||lambda1||)^2 = det(s)det(z) / (s'z + sqrt((s'z)^2 - det(s)det(z)))` with
  `det(u) = (u0 - ||u1||)(u0 + ||u1||)`, without scaling and without cancellation.

tau*kappa is checked the same way. This is the negative-infinity neighbourhood of the central path; with
theta = 1e-4 it only rejects steps that leave some pair badly uncentred.

### Clarabel safeguards

- **Reduced Mehrotra correction in the first iteration:** the second-order term of the combined step is
  scaled by the affine step length, because the starting point, cold or warm, may be badly centred.
- **Progress check:** from the third iteration on, if a residual grows, the solve stops with
  `InsufficientProgress` when the iterate is already accurate (kappa/tau < 100 eps with a converged gap)
  or when a residual grows by 100x above 100 tol_feas. The best iterate is reported, upgraded to
  `AlmostSolved` if it meets the reduced tolerances.

### Fewer factorizations

- **Termination before refactorization:** residuals and termination checks need no factorization, so the
  converged iteration does not factorize any more. A cold solve with k iterations takes k factorizations
  (cold start plus k - 1), before k + 1.
- **Warm start without cold start:** the recentered warm point is used directly if its merit is not above
  the merit of the last cold start (cached since `setup()`). Only otherwise is the cold start computed (one
  factorization, two solves) and the better point kept. An accepted warm start takes k - 1
  factorizations for k iterations.

### Scale invariance

`SecondOrderCone::inverseProduct()` clamped `det(lambda)` and `lambda0` at an absolute 1e-12. For problems
whose solution has small magnitudes (impulses at small step sizes) this triggered close to convergence and
changed the corrector. lambda is interior by construction; the determinant is now computed as
`(u0 - ||u1||)(u0 + ||u1||)` without clamps (`IpmSafeguards.SocInverseProductIsScaleInvariant` checks
scales 1, 1e-8 and 1e-12).

### Measurements

Benchmark suite (71 instances, all Solved): 561 -> 563 iterations and 632 -> 563 factorizations in total.
ApexContacts needs 5 instead of 6 iterations on all 9 instances; 15 random, portfolio and group-lasso
instances change by 1 or 2 iterations (net +11).

Warm-started sequences of 30 steps with 0.5 % changes of q and b per step (the CardilloCxx pattern),
Phase 9 -> Phase 10a:

| Problem | iterations | factorizations | time |
|---|---|---|---|
| FrictionChainXL 66 | 180 -> 150 | 210 -> 150 | 347 -> 263 ms |
| ApexContacts 64 | 180 -> 150 | 210 -> 150 | 10.4 -> 8.4 ms |
| FrictionChain 64 | 180 -> 180 | 210 -> 150 | 9.4 -> 8.4 ms |
| RandomSOCP 20x4 | 201 -> 185 | 231 -> 185 | 101 -> 84 ms |

(The RandomSOCP steps that do not end Solved are correctly reported `DualInfeasible`: a pure SOCP,
P = 0, becomes unbounded when q is perturbed. The sequence benchmark therefore uses GroupLasso.)

**Finding (T11.7, fixed in Phase 11):** on FrictionChainXL and ApexContacts the warm point lost the merit
comparison in every step. See "Warm start" below for the cause and the fix.

## Phase 11: warm start, equilibration, cheaper solves, batched cones

### Warm start (T11.7)

**Cause of the lost merit comparison.** The previous recentering shifted the stored solution (s, z) into
the interior and then scaled s and z, but not x and tau, by rho = sqrt(warm_mu0 deg / s'z) so that
mu = warm_mu0. Near a converged point s'z is of order deg * 1e-9, so rho is about 300. Scaling s alone
multiplies the inactive slacks in `Ax + s - tau b` by 300, scaling z multiplies the multipliers, including
the free equality multipliers z_E, in `A'z + Px + tau q` by 300. The residuals of the warm point were
therefore large by construction, independent of the data change, and it lost against the cold start.
(Scaling x, s, z, tau and kappa together would leave everything unchanged, the embedding is homogeneous.)

**Fix.** The warm start of Skajaa, Andersen and Ye ("Warmstarting the homogeneous and self-dual interior
point method for linear and conic quadratic problems", Math. Prog. Comp. 2013): with the previous solution
(x*, s*, z*) (divided by its tau) and the standard start (0, e, e, 1, 1),

```
x0 = lambda x*,  s0 = lambda s* + (1 - lambda) e,  z0 = lambda z* + (1 - lambda) e,
tau0 = 1,        kappa0 = 1 - lambda.
```

The residuals change by about lambda times the data change plus (1 - lambda) times those of the
standard start, and every complementarity pair is at least (1 - lambda)^2, so the point is interior and
not badly centred. Zero-cone s stays 0, z_E = lambda z*_E. `Settings::warm_lambda` (default 0.99)
replaces `warm_mu0`. A user warm start (`setWarmStart()`) need not be feasible: s and z are projected
onto the cones first (`ConeSet::projectOntoCones()`, a no-op for the solver's own iterates). The rule
"use the warm point if its merit is not above the cached cold-start merit" is kept.

Sequences of 30 steps, q and b perturbed by up to 0.5 % per entry and step (`conicxx_benchmarks`, end of
the output; iterations / factorizations):

| Problem | cold | warm, before | warm, lambda = 0.99 |
|---|---|---|---|
| FrictionChain 64 | 240 / 240 | 182 / 153 | 153 / 124 |
| FrictionChainXL 66 | 150 / 150 | 150 / 150 | 121 / 92 |
| ApexContacts 64 | 150 / 150 | 150 / 150 | 121 / 92 |
| GroupLasso 30x5 | 270 / 270 | 213 / 184 | 178 / 149 |

(Cold and "before" with the old equilibration, which the next section changes.) lambda = 0.9 is too
weak (XL 179 iterations), 0.999 is as good on the contact families but worse on GroupLasso (202), 0.9999
is worse everywhere; 0.99 is the default.

### Equilibration as in Clarabel (T10.2)

Each Ruiz iteration clips its factors against the cumulative bounds (`delta_i in [min/d_i, max/d_i]`),
so the stored data equal `c D P D`, `E A D` exactly; norms below `min_scale` leave a row or column
unscaled. The cost is rescaled inside the loop by `1 / max(mean column inf-norm of P, ||q||_inf)`,
skipped only if both are 0.

Including ||q||_inf makes the iteration count independent of the scale of q: a box QP with P = I needs
12 / 8 / 8 iterations for ||q||_inf = 1 / 1e3 / 1e6, against 12 / 12 / 15 with a P-only cost scale.
The price on the generated suite is about one iteration per instance: 563 -> 635 iterations in total
(P-only cost scaling inside the loop: 570; no cost scaling: 577). The random families have
||q||_inf above the mean norm of P, so their P is scaled down and the problems look more LP-like to the
method. Scale invariance is kept as the default because it is what a general-purpose solver needs; the
cost on the CardilloCxx scenes is reported in "CardilloCxx scenes" below.

### Cheaper KKT solves (T11.3, T11.4)

- The constant right-hand side (-q, b) and the affine one (rx, s - rz) are solved in one pass over L
  (`QuasiDefiniteLdlt::solvePair()`, two interleaved columns).
- The affine (predictor) direction is not refined by default (`Settings::refine_affine`): it only sets
  sigma and the second-order correction. Refinement of the constant and the combined solves is
  unchanged, so the Newton directions and the equality rows are as accurate as before.
- With `refine_max_iter = 0` no residual is computed at all.

Iteration counts are unchanged on all 71 instances. FrictionChainXL 264 (cold, 6 iterations):
19.1 -> 16.4 ms; without refinement 14.1 -> 11.1 ms.

T11.3 (adaptive refinement tolerance) was not implemented: a whole solve needs about 9 refinement steps
in total (FrictionChainXL 264), so a looser tolerance saves almost nothing. The cost is the first
residual of every refined solve, which the predictor change removes where it is safe.

### Batched cone operations (T11.8)

Measured share of the cone operations (all `ConeSet` calls) in the solve time of FrictionChain with
10^4 (10^3) contacts: 32 % (34 %), plus per-block virtual calls inside the KKT update.

`ConeSet` now owns the scaling data of all blocks in contiguous arrays (orthant w; SOC eta per block and
w at the block's rows) and runs every operation as one loop without virtual calls. If all SOC blocks are
3-dimensional the loop uses kernels with a compile-time dimension. The math lives once, in
`soc_kernels.h`, and the per-cone classes call the same kernels. `setup()` no longer allocates one cone
object and three vectors per block. `ConeSet::block()` is replaced by `blockType()`/`blockDim()`/
`blockOffset()`/`numHsEntries()`/`writeHsLowerTriangle()`.

Also fixed in passing: `maxStep()` computed `det(x) = x0^2 - ||x1||^2` with cancellation; now
`(x0 - ||x1||)(x0 + ||x1||)` as everywhere else.

Iteration counts unchanged; FrictionChain 10^4 contacts 38.8 -> 32.3 ms per solve; total solve time of
the suite 520 -> 490 ms.

### Static/dynamic split (T11.5): deferred

Not implemented (maintainer decision). With warm starts a CardilloCxx step needs few iterations, each a
sparse factorization of a matrix with an inexpensive pattern. The split would trade it for one static
factorization per step plus a factorization of the Delassus-like contact matrix per iteration. Whether
that pays off depends on how dense G is for the scene; the numbers in "CardilloCxx scenes" are the
baseline for that decision.

### CardilloCxx scenes

First steps of the domino scene (about 13,500 frictional contacts, no bilateral rows, 20 steps) and the
slinky scene (rod with 4950 compliant bilateral rows, 146 -> 2961 contacts, 40 steps), configs
`scene_conicxx.config`, per step. The contact set changes in every step of both scenes, so every step
runs `setup()`. "Old" is ConicXX `cddaa5d` as pinned before (10 resp. 20 steps), Clarabel 0.11 on the
identical config:

| Scene | Solver | iterations | solve | setup |
|---|---|---|---|---|
| domino | old ConicXX | 14.5 | 1240 ms | 353 ms |
| domino | ConicXX, warm start after setup | 10.2 | 702 ms | 34 ms |
| domino | ConicXX, cold after setup | 19.4 | 1347 ms | 32 ms |
| domino | Clarabel | 16.4 | 1519 ms | 35 ms |
| slinky | old ConicXX | 12.9 | 311 ms | 125 ms |
| slinky | ConicXX, compliant rows | 11.9 | 125 ms | 16 ms |
| slinky | ConicXX, augmented (pi as variables) | 12.1 | 111 ms | 17 ms |
| slinky | Clarabel | 12.4 | 200 ms | 17 ms |

No dynamically regularized pivot occurred. "Warm start after setup" is CardilloCxx's
`setWarmStart()` from the current velocities, the previous percussions and the contact impulses
matched across steps; on domino it is accepted in every step and halves the iterations against the
cold start.

Observations and open items:

- **Cold start on domino** needs more iterations than with the old equilibration (19.4 against about
  14.5). With the P-only cost scale it needs 15.5 in the first 6 steps, but the warm-started default
  then needs 14.0 instead of 11.7, and slinky 9.0 instead of 10.3 in its first 20 steps. The
  Clarabel rule is kept (see "Equilibration as in Clarabel").
- **Warm start on equality-dominated scenes.** On slinky the warm point loses the merit comparison
  in every step (merit about 0.3 against 0.004): the cold start solves the linear equations exactly,
  while the warm point carries the change of the position-level right-hand side of 4950 stiff rows.
  Correcting the warm point onto the linear equations with the cold start's factorization (one extra
  solve) makes its dual residual about 100x and its gap 10-100x smaller than the cold start's, at about
  1e-2 of its mu, but its primal residual stays about 1.5x larger (projection onto the cones), so it
  still loses the max-merit test. Forced, it saves 11 % of the iterations on slinky (12.8 -> 11.4) and
  costs some on domino, where the plain warm point is better. The correction was therefore not kept.
  The next lever is an acceptance rule that accounts for mu (in the embedding the residuals shrink in
  proportion to mu, so a point with a slightly larger residual but a much smaller mu needs fewer
  iterations).
- **Compliant vs augmented rows on slinky:** 11.9 vs 12.1 iterations per step and the same solution,
  but 125 vs 110 ms per step: the compact form needs about twice the refinement steps (44 vs 23 per
  solve) at equal fill (nnz(L) about 2.1e5 to 2.4e5 in both). Reducing the factor-only static
  regularization of compliant rows by C_i did not change that on slinky and made a well-conditioned
  test stall at 1e-10, so it was not kept. Open: why refinement converges more slowly with the exact
  -C_i on the diagonal than with the pi pivots (stringTest, stiffer, is faster in the compact form:
  4.0 vs 4.6 iterations, 5.4 vs 9.9 ms per solve).
- **Cost per iteration** on domino is about 70 ms (KKT dimension about 42,000): T11.5 (contact-space
  Schur complement) is the next structural lever once the iteration count is settled.

### Profile of the CardilloCxx target scenes

Measured with temporary timers (removed again), default `scene_conicxx.config`, one run each: domino 10
steps (n = 10,176, about 13,500 SOC3 contacts, m = 40,300, nnz(L) = 1.65e6 to 1.71e6), slinky 40 steps
(n = 4,956, 4,950 compliant rows, 0 -> 6,518 contacts, nnz(L) up to 8.6e5). Every step runs `setup()`.

| | domino | slinky |
|---|---|---|
| solve per step | 847 ms | 134 ms |
| iterations per step | 10.0 | 10.9 |
| one numeric factorization | **60 ms** | 5.9 ms |
| numeric factorization | **78 %** | **53 %** |
| triangular solves (incl. refinement corrections) | 14 % | 25 % |
| refinement residuals | 2 % | 11 % |
| cone operations (NT scaling, Hs write, Jordan ops, step length) | 2.3 % | 4.7 % |
| residuals and metrics | 1 % | 3 % |
| other vector work | 2 % | 3 % |
| setup per step (outside solve) | 46 ms | 18 ms |
| setup / whole step | 5 % | 10 % |
| cold starts computed only for the warm-start comparison | 9 % of factorizations | 8 % |

Setup splits roughly evenly into equilibration, KKT pattern, AMD and symbolic analysis (symbolic largest
on domino: 18 ms).

Consequences, in order of expected benefit:

1. **Domino is factorization-bound.** 60 ms for nnz(L) = 1.7e6 is the speed of scalar simplicial code.
   Levers: the ordering (check whether AMD eliminates the contact rows first; a constrained ordering is
   the body-space reduction without new code paths), a supernodal factorization, the body-space
   Cholesky (T8.1). All solves of one iteration together cost about 18 % of one factorization, so
   Gondzio correctors are cheap here relative to a saved iteration.
2. **No re-setup** (caller side, superset pattern): saves setup plus the extra cold-start
   factorization, about 14 % on domino and 18 % on slinky, and keeps the internal warm start.
3. **Slinky** spends 36 % in solves and refinement residuals; the compact compliant rows need about
   twice the refinement steps of the augmented form (44 vs 23 per solve), worth 10-15 % there.
   Supernodal factorization helps less than on domino.
4. **Cone code** (2-5 %) and rank updates are not worth further work.
5. **ADMM break-even on domino:** a triangular solve pass costs a few ms against 60 ms per
   factorization, so the budget of a 10-iteration IPM solve is about 150-200 ADMM iterations. A
   warm-started ADMM at 1e-5 can stay within that, but not reliably.

### Ordering experiment on the target scenes

KKT matrices dumped in the middle of a solve (domino: n = 10,176, 40,596 cone rows; slinky: n = 4,956,
4,950 compliant rows, 18,471 cone rows), factorized with `QuasiDefiniteLdlt` under different orderings
(`QuasiDefiniteLdlt::analyzePattern(K, signs, perm)`):

| Ordering | domino nnz(L) | domino flops | domino factorize | slinky nnz(L) | slinky flops | slinky factorize |
|---|---|---|---|---|---|---|
| A: AMD on the full KKT (current) | 1.74e6 | 3.4e8 | 64 ms | 7.9e5 | 3.9e7 | 9.4 ms |
| B: cone rows first, AMD on the rest (body space) | 1.79e6 | 4.0e8 | 73 ms | 8.1e5 | 4.3e7 | 10.2 ms |
| C: cone and compliant rows first (full body space) | - | - | - | 8.0e5 | 4.0e7 | 9.6 ms |
| D: bodies and zero rows first, cones last (contact space, T11.5) | 1.6e7 | 1.3e10 | 3.9 s | 2.6e8 | 3.1e12 | (skipped) |

(flops = sum of squared column counts of L; refined solves reach 1e-12 in all cases, no dynamic pivots.)

- **AMD already finds an ordering as good as the body-space reduction**, although it interleaves
  body and contact pivots. A body-space Cholesky (T8.1) would not save flops on these scenes; its
  only advantage would be a Cholesky without pivot signs.
- **The contact-space split (T11.5) is ruled out for these scenes**: the Delassus-like matrix is
  essentially dense (domino 60x slower, slinky infeasible).
- The simplicial factorization runs at 5.3 GFlop/s (domino) and 4.2 GFlop/s (slinky).
- **Supernodes** (fundamental, AMD order): domino 14,901 for 50,772 columns, but 79 % of the flops are
  in supernodes of width >= 32 (a dense trailing block among the bodies); slinky 11,815 for 28,377
  columns with 59 % of the flops in widths 2 to 7 and 24 % in widths >= 32.

Conclusion: the remaining lever on the factorization is a **supernodal LDL^T** (dense BLAS-3 kernels on
the wide supernodes, dynamic regularization inside the dense diagonal-block factorization). Expected:
large on domino, where the factorization is 78 % of the solve, modest on slinky.

### Supernodal factorization

`QuasiDefiniteLdlt` has a second numeric path (`Settings::supernodal`, default on, read at `setup()`):

- **Supernodes.** After the ordering, the elimination tree is postordered (same fill) and fundamental
  supernodes are formed: column j + 1 continues j's supernode if it is j's parent and its pattern is
  j's without row j + 1. Every column's pattern is then the tail of its supernode's row list, so the
  supernode is a dense column-major block whose columns map one to one onto the compressed L. The
  triangular solves, the paired solve and refinement are unchanged.
- **Numerics, left-looking.** A supernode assembles its columns of C (the pivots starting from the
  static regularization), gathers the updates `L_d D_d L_d(cols)'` of every earlier supernode d
  reaching into its columns (linked lists keyed by the next unapplied row), then factorizes its block:
  panels of 32 columns, left-looking within a panel, one matrix product per panel for the trailing
  columns. Pivot signs, static and dynamic regularization are exactly those of the simplicial path
  (`setPivot()`); dynamic regularization still happens inside the numeric pass.
- **Performance switches** (results do not depend on them): updates from supernodes narrower than 4
  columns use scalar loops over the contiguous columns, wider ones a matrix product. A product with
  inner dimension 1 cost more than it saved (RandomQP 200: 8.7 ms per factorization against 0.36 ms
  with the scalar path). The supernodal path is taken only if L has at least 16 entries per column on
  average; on very sparse factors (FrictionChainXL: 4 per column) the per-supernode overhead outweighs
  the contiguous access (1.7 ms against 0.62 ms simplicial).

Per factorization (identity scaling, `-march=native`):

| Problem | nnz(L) | simplicial | supernodal |
|---|---|---|---|
| domino (dumped mid-solve) | 1.74e6 | 60 ms | 17.4 ms |
| slinky (dumped mid-solve) | 7.9e5 | 9.1 ms | 6.8 ms |
| RandomQP 200 | 9.99e4 | 3.7 ms | 0.35 ms |
| GroupLasso 60x5 | 4.70e4 | 1.74 ms | 0.36 ms |
| RandomSOCP 50x4 | 5.99e4 | 2.09 ms | 1.95 ms |
| FrictionChainXL 264 | 1.61e5 | 0.62 ms | (falls back) |

Benchmark suite (Release, without `-march=native`): iterations unchanged (635), solve time
481 -> 262 ms; RandomQP/LP 200 2.8-3.9x, GroupLasso 60 1.95x, Portfolio 120 1.34x faster, the contact
families unchanged (fallback or tiny), RandomSOCP about 10 % slower (open). CardilloCxx, same machine
and binary apart from the switch: domino 831 -> 315 ms per step (10 steps), slinky 149 -> 115 ms per
step (40 steps), iterations unchanged.

With the factorization 2.6x faster on domino, the triangular solves (7.4 ms per pass against 17.4 ms
per factorization) become the next cost to watch; a supernodal solve (dense blocks with BLAS-2/3)
would follow the same structure.

## Exact Coulomb friction (experimental)

`ConeSpec::soc_coulomb` marks second-order cones as Coulomb cones. The problem is then the
discrete Coulomb friction problem of Acary, Armand, Nguyen (CMAME 461, 2026) in the unit-cone
scaling (rows `(n, mu t1, mu t2)`, done by the caller):

    Px + q + A'z = 0,   Ax + s - b - C z_E - E l(s) = 0,   s o z = 0,   s, z in K,

with `l_i(s) = ||s_i,bar||` the De Saxce term and `E` putting it into the first row of cone `i`. `s`
is the modified relative velocity, `u = s - E l(s) = b - Ax` the relative velocity (the tangential
parts of `s` and `u` agree). This is not the KKT system of an optimization problem (non-associated
friction), so `solve()` runs `CoulombIpm` instead of the homogeneous embedding; every other part of
the solver (cones, NT scaling, KKT pattern and ordering, equilibration, refinement, warm start) is
shared. Without Coulomb cones nothing changes.

### Newton matrix

With `B^-1 = I - theta E L`, `L_i = [0, s_i,bar' / l_i]`, the linearized primal equation is
`A dx + B^-1 ds = -r_p`, and the NT elimination `ds = -(Hs dz + ds_const)` gives

    [ P   A'          ] [dx]   [ -r_d                      ]
    [ A   -B^-1 Hs    ] [dz] = [ -r_p + B^-1 ds_const      ].

Since `L_i e_0 = 0`, `(E L)^2 = 0` and `B = I + theta E L`. Of the two equivalent reductions,
`[P A'; B A -Hs]` (with the (2,1) block row-mixed) and the one above, this one changes neither `A` nor
`A'`: only row 0 of every Coulomb block, `N_i = Hs_i - theta e_0 (Hs_i L_i')'`. The KKT pattern,
ordering and symbolic analysis are those of the convex problem (no padding of A to the union of a
cone's row patterns). `N_i(0,0) = eta^2 (2 w_0^2 - 1 - 2 theta w_0 w_bar' l_bar) >=
eta^2 (w_0 - sqrt(w_0^2 - 1))^2 > 0` (`|l_bar| < 1`, `w'Jw = 1`), so a cone pivot eliminated first
keeps its sign.

For a sliding contact near the solution, with Jordan frame `c_1,2 = (1, +-d)/2`, `Hs ~ alpha c_1 c_1'`
(alpha ~ 1/mu) and `N ~ alpha/2 (0, d) c_1'`: the large NT stiffness moves from the normal row into
the tangential direction. The symmetric matrix of the convex problem therefore differs from the
Coulomb one at the leading order on every sliding contact, so GMRES preconditioned with the
symmetric factorization (the guide's path A) would need about one iteration per sliding contact.
The Newton matrix is factorized directly instead.

### Unsymmetric LDU on the quasidefinite structure

`QuasiDefiniteLdlt::factorizeUnsymmetric()` computes `K = L D U` on the analyzed (symmetric) pattern
and ordering: simplicial (up-looking, row k of L and column k of U from two triangular solves over
the same reach) and supernodal (`sn_xu_` holds the U^T blocks in the layout of the L blocks, every
update done twice with the roles swapped). `U` is stored as `U^T` in the pattern of L; solves and
refinement use both value arrays. `KktSystem::updateScalingAndFactorize(cones, coulomb_blocks, row0)`
writes the modified rows: the lower triangle keeps `-Hs(a,0)`, the diagonal and the transposed
entries `(0,a)` get `-row0_a`.

Pivot rule: the matrix is not quasidefinite. Eliminating a body column after its cones gives the
pivot `P + A' Hs^-1 B A`, and `Hs^-1 B ~ (2/sigma_2) c_2 e_0'` for a sliding contact is not definite,
so a pivot of the unexpected sign is legitimate. Forcing the quasidefinite sign (replacing it by
`+delta`) destroyed the factor in the hyperstatic box test (refinement residual 4e31). The LDU only
replaces pivots with `|d| < dynamic_eps`, keeping their sign. No dynamically regularized pivot
occurred on domino and slinky.

### Smoothing

`l_i = sqrt(||s_i,bar||^2 + (c mu_k)^2)` with `mu_k` the complementarity measure of the iteration
(frozen within it) and `c = CoulombSettings::smoothing / z_ref`, `z_ref` the median normal component
of z at the start. The error `c mu_k` has the order of the complementarity, `L` stays well defined at
`s_bar = 0` (sticking), and the Jacobian is smooth. The termination test uses the unsmoothed term.
`smoothing = 0` is the paper's unsmoothed variant (L = 0 at `s_bar = 0`); the sticking test passes
with 0, 0.1 and 1.

### The loop

Infeasible-start Mehrotra predictor-corrector without the embedding (no infeasibility certificates:
existence follows from the Slater condition, which is not checked): residuals `r_d`, `r_p` with the
smoothed term, `mu = s'z / nu`, affine direction, `sigma = (mu_aff/mu)^3`, corrector with the
second-order term, step to the boundary times `max_step_fraction`, backtracking with the centrality
safeguard of the convex loop. Start: as the convex loop (recentered warm start if its merit is not
above the cold start's); a warm-started solve that fails is repeated from the cold start.
Termination: `res_primal` (unsmoothed term), `res_dual` below `tol_feas` and, for every cone, the
complementarity in velocity units (user units)

    v_i = ||s_i o z_i|| / (||z_i|| + m_i ||s_i||)  <=  tol_gap_abs + tol_gap_rel ||s_i||,

with `m_i = 1 / max_rows (A P^-1 A')_rr` the effective mass of the contact (diagonal of P). For an
active contact `v_i` is its velocity error, for a separating one the velocity its residual impulse
induces. The measure is local: an earlier version normalized with global scales
(`V = max(||x||, ||s||, ||q|| / max P_ii)`, `Z = max(||q||, ||z||)`) and let every contact keep an
error of `tol V`, which on domino added up to 20-40x the energy injection of the convexified solve
(exact_coulomb_findings.md).

**Why not `||s o z||`.** The first version tested `max_i ||s_i o z_i|| <= tol_gap_abs` (velocity x
impulse). On domino (impulses 1e-4 to 1e-2 N s) that accepted normal velocities of `tol / z_i`:
active contacts separated with 3 mm/s on average and up to 8 cm/s, and the dominos bounced. On
interior points every active Coulomb contact separates (`s_i in int L <=> u_i0 > 0`), so the error
always injects energy; the convex solver has the same weakness but overshoots mu by about 100x in its
last iteration. The floors `max(1, .)` of the residual normalizations are not usable either: with
`Z = 1`, every contact with `z_i < tol` counts as separated whatever its velocity. `||b||` is not used
in V: the position-level bias of stiff bilateral rows (slinky: 250) is no contact velocity.

**Complementarity floor.** In the NT neighborhood (`lambda_min(lambda o lambda) >= theta mu`) a pair
near the boundary can keep its tangential parts misaligned: with misalignment phi,
`s'z ~ ab phi^2` but `||s o z|| ~ ab phi`, i.e. `||s o z|| ~ sqrt(mu)`. The NT direction does not
target `s o z` itself (the paper's Arw(u)/Arw(r) linearization does), and Mehrotra's sigma ~ 0 drives
mu far below what the tolerance needs: on the hyperstatic box mu reached 1e-17 with
`||s o z|| = 5e-8`, the KKT solves lost all accuracy and the run ended AlmostSolved; from a warm start
(previous sliding direction) the complementarity fell only 5x per iteration. The centering target is
therefore bounded below, `max(sigma mu, CoulombSettings::mu_floor * tol_comp)`, and without the
second-order term once the bound is active: the steps then converge to the central point at that mu,
where `||s o z|| = mu < tol`. With the local measure the bound is
`mu_floor * min_i tol_i (||z_i|| + m_i ||s_i||)` (scaled units). The box solves to 1e-9 in 11
iterations.

**Blocked steps.** After two steps with `alpha < 0.1` the loop restores the last iterate with
centrality `>= 0.1 mu` and runs a pseudo-arclength continuation of the central path (`continuePath`:
bordered corrector from two solves with one factorization, local relative metric), at most 5 times;
otherwise `InsufficientProgress`. This fallback does not get through the domino failures (the last
well-centred iterate is often the start), see exact_coulomb_findings.md.

**Path monitor** (`verbose >= 3`): the tangent `J dy/dmu = -dF/dmu` with the factorization of the
iteration gives per-cone orders `mu ||ds_i/dmu|| / ||s_i||` (0: stays, 1: vanishes like mu, 1/2:
not strictly complementary, >> 1: sharp turn of the path) and the number of pivots of unexpected sign.

**Contact states** (`Solution::contact_states`) come from the spectral values of `s_i`, `z_i` in
equilibrated units with threshold `sqrt(mu)` (the vanishing ones are O(mu), the others O(1)).

### Measurements

CardilloCxx, committed `scene_conicxx.config` (domino 1e-5, slinky 1e-6), same binary, convexified law
(frozen shift, embedding, supernodal LDL^T) against `conicxx.exact_coulomb = on` (supernodal LDU),
mean per step; `xi_N` is the normal velocity of contacts with a percussion above 1e-3 of the largest:

The last column used the intermediate global measure (V, Z above), which was still too loose. With the
local measure the exact solver fails in step 1 of domino already at 1e-5; the cause (a late sharp
turn of the central path driven by degenerate contacts) and the cost of following it are analysed in
exact_coulomb_findings.md.

| Scene | convexified | exact, `||s o z||` test | exact, global per-cone measure |
|---|---|---|---|
| domino, 30 steps | 10.0 it, 258 ms, mean xi_N 1e-5 m/s | 5.3 it, 183 ms, mean xi_N 3e-3 m/s | 7.9 it, 292 ms, mean xi_N 3e-5 m/s |
| slinky, 40 steps | 11.8 it, 112 ms | 7.9 it, 80 ms | fails after about 30 steps (AlmostSolved) |

The fast first results were at a much lower accuracy. With the per-cone measure:

- domino at 1e-5 is accurate (largest xi_N 0.03 m/s against 0.10 convexified) and costs about as
  much as the convexified solve; at 1e-6 the first step fails.
- slinky at 1e-6 stalls in some step with the steps to the boundary collapsing (1e-4) at mu about
  2e-9 (scaled) while the KKT solves are accurate (refinement residual 1e-11). The limiting cone is a
  degenerate stick-slip contact (`s ~ 0`, `z` near the cone boundary, `s_bar` and `z_bar` 25 degrees
  apart, `dz` 100x larger than `z`): the Newton model of the De Saxce term (curvature `1/||s_bar||`)
  is useless there. Neither smoothing 0 or 1, nor the floor 0 or 1, nor a pure centering step after
  a blocked step, nor 30 refinement steps helps. The open measure is an end game that does not rely
  on the central path there (active-set or semismooth Newton polishing, see below).

The simplicial LDU alone made domino much slower (459 ms per step, `||s o z||` test). The spinning-ball
benchmark (`test_convexified_friction`, exact-ConicXX variant): `xi_N = 0` to 2e-8 in every active
step, no penetration, effective restitution e.

### Not implemented (relative to the implementation guide)

- P_mu scaling inside ConicXX: the caller passes unit-cone data (CardilloCxx does already).
- Slater check, theta homotopy, ANM. Pseudo-arclength continuation exists only as the fallback
  above; a working one would cost hundreds of factorizations per step (exact_coulomb_findings.md).
- Semismooth Newton (or active-set) polishing.
- GMRES with the symmetric factorization as preconditioner (see "Newton matrix").
- FCLIB harness.
