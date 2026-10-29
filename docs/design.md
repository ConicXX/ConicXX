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

`QuasiDefiniteLdlt` (`src/conicxx/quasidefinite_ldlt.{h,cpp}`) factorizes the KKT matrix

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
