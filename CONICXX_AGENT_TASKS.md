# ConicXX — Refactor Tasks for a Coding Agent

Goal: turn ConicXX (`github.com/ConicXX/ConicXX`, reviewed at commit `784e5cd`) into a robust
drop-in replacement for the Clarabel C++ bindings in a frictional-contact multibody engine.

**Top priorities, in order:** (1) never perturb equality (zero-cone) constraints, (2) robustness
on degenerate frictional contact (contacts at the cone apex, redundant/hyperstatic equalities),
(3) Clarabel-compatible API, (4) performance.

---

## 0. Ground rules for the agent

- Work phase by phase, one PR-sized change at a time. Do not mix phases in one commit.
- After every change:
  ```sh
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
  ctest --test-dir build -LE benchmark --output-on-failure
  ./build/benchmarks/conicxx_benchmarks > bench_after.txt
  ```
- Before Phase 1, record `bench_baseline.txt` from the unmodified code (after the T0.1 build fix).
  Every later change must report the iteration-count diff against it. Iteration counts may only
  get worse with an explicit justification in the commit message.
- Keep the public types (`Vec`, `SparseMat` with `int` indices, Eigen 3.4, C++17).
- No new mandatory dependencies. Optional ones (e.g. SuiteSparse CAMD) behind a CMake option.
- Do not silently "rescue" numerics (clamps, identity fallbacks, huge regularization bumps).
  If something fails, report it through `Status` and `Info`.
- Long explanatory comments belong in `docs/design.md`, not in source files. Keep code comments short.
- If a task below is ambiguous or conflicts with what you find in the code, stop and ask the
  maintainer rather than guessing. Open questions are collected in Section 10.

## 1. Repository orientation

| Path | Role |
|---|---|
| `include/conicxx/{solver,settings,solution,cone_spec,types}.h` | public API |
| `src/conicxx/solver_impl.{h,cpp}` | HSDE IPM loop (init, residuals, affine/combined step, termination, warm start) |
| `src/conicxx/cones/*` | `ZeroCone`, `NonnegativeCone`, `SecondOrderCone`, `ConeSet` |
| `src/conicxx/kkt/kkt_system.{h,cpp}` | KKT assembly, regularization, retry/escalation, refinement |
| `src/conicxx/kkt/{qdldl_ldlt,regularized_ldlt}.cpp` | LDLᵀ backends (QDLDL and a modified per-pivot-regularized QDLDL) |
| `src/conicxx/kkt/equilibration.cpp` | Ruiz scaling |
| `benchmarks/` | deterministic suite incl. `FrictionChain`, `FrictionChainXL` |

Problem form (keep): `min ½xᵀPx + qᵀx  s.t. Ax + s = b, s ∈ K`, with P given as its upper triangle.

KKT matrix (lower triangle stored):
```
K = [ P + δ_P I     Aᵀ          ]
    [ A           −(H + δ_K)    ]    H = blockdiag(0 on zero cone, W²_nonneg, W²_soc)
```

---

## Phase 0 — Hygiene (small, do first)

**T0.1 ~~Fix the build.~~ Resolved.** `src/conicxx/backend_comparison.cpp` and
`include/conicxx/backend_comparison.h` now exist and are wired into `CMakeLists.txt`; a fresh
clone builds. Remaining gap, low priority: `compareLinearSolverBackends()` has no test or call
site anywhere in the tree (`grep` finds none) — it's currently only reachable by writing a
throwaway `main()`, the way it was used to diagnose the domino-scene report. Add a small smoke
test, or a `tools/compare_backends` CLI that loads a P/A/q/b/cone dump from disk, so it doesn't
bit-rot silently.

**T0.2 ~~License.~~ Resolved.** GPL-3.0 (matches CardilloCxx, the primary downstream consumer;
plain GPL over LGPL since static linking is the intended embedding pattern and no non-GPL
consumer is known). `LICENSE` (verbatim GPL-3.0 text) and `NOTICE` (crediting QDLDL, Apache-2.0,
both the unmodified link and the adapted numeric loop in `regularized_ldlt.cpp`) added, plus a
README section pointing at both.

**T0.3 Remove local tooling files.** Delete `.claude/settings.json` (contains local absolute
paths) and add `.claude/` to `.gitignore`.

**T0.4 CI.** Add a GitHub Actions workflow: Ubuntu, GCC and Clang, Release and Debug. Add one
job with `-fsanitize=address,undefined`. Run `ctest -LE benchmark`, plus the benchmark label as a
separate non-blocking job. Note: Eigen is fetched from gitlab.com if not found; install
`libeigen3-dev` in CI instead.

**T0.5 Move design commentary.** Collect the long rationale comments (kkt_system.h,
second_order_cone.cpp, regularized_ldlt.cpp) into `docs/design.md`. Leave one-line comments in
the code. Do this after Phases 1–3, since those phases delete much of the commented code.

---

## Phase 1 — Structured cone scaling (critical performance bug)

**Problem.** `ZeroCone` and `NonnegativeCone` store `Hs_` as a dense `dim × dim` `Mat`.
`ConeSet::updateScaling` copies every block into `scaling_blocks_` each iteration, and
`ConeSet::mulHs` does dense matvecs. Measured: `FrictionChainXL contacts=264` (19,800 zero-cone
rows) takes about 1.6 s per IPM iteration although the LDLᵀ has only ~160k nonzeros. The
66-contact case already uses ~196 MB RSS; the 264-contact case needs ~3 GB for the zero block alone.

**T1.1 Replace the dense-block interface.** Change `ConeBase`:
- Remove `scalingBlock()` returning `const Mat&`.
- Add `void mulHs(const Ref<const Vec>& x, Ref<Vec> out) const` (per cone, structured).
- Add `void writeHsLowerTriangle(Scalar* kkt_values, const Index* slots) const`, or equivalent,
  so each cone writes its own (2,2) entries directly into the KKT value array.
- Zero cone: stores nothing, writes nothing (only regularization, see Phase 2).
- Nonnegative: store `w` (size dim); H = diag(w²) = diag(s/z).
- SOC: store `η` and `w` (see T3.1). For `dim ≤ 4` a dense lower triangle in K is fine.
  For large SOC (`dim > soc_sparse_threshold`, default 16) keep the dense triangle for now, but
  leave a TODO for Clarabel-style sparse expansion.

**T1.2 Remove `ConeSet::scaling_blocks_`** and the per-iteration copies. `KktSystem` must no
longer take `Mat` blocks.

*Accept:* all tests pass. Benchmark iteration counts are identical to baseline (this change must
be numerically neutral). `FrictionChainXL contacts=264` time per iteration drops by at least
50×. Peak RSS for 264 contacts stays below 200 MB (measure with `/usr/bin/time -v`).

**Done.** `ConeBase::scalingBlock()` replaced with `mulHs()`/`numHsEntries()`/
`writeHsLowerTriangle()`; `ZeroCone`/`NonnegativeCone` no longer store a dense `dim x dim` `Mat`
at all (`ZeroCone` stores nothing, `NonnegativeCone` keeps only its existing `w_` vector).
`SecondOrderCone` keeps its dense `W_`/`Hs_` internally (dim is always small there; rewriting its
NT-scaling math is Phase 3's job, not touched here). `ConeSet::scaling_blocks_` and
`scalingBlocks()` removed; `KktSystem` fills K's (2,2) block by asking each cone to write its
entries directly (via a reusable scratch buffer, no per-iteration heap allocation) instead of
copying/reading a dense matrix.

Verified: all 69 tests pass (3 new: per-cone `writeHsLowerTriangle`-vs-`mulHs` cross-checks,
satisfying the "structured == dense" test from Section 9), clean under
`-fsanitize=address,undefined` (tests + full benchmark binary). Benchmark suite: iteration
counts, statuses and residuals bit-for-bit identical to `bench_baseline.txt` (only timing
changed) -- confirms the change is numerically neutral, not just "still converges."
`FrictionChainXL contacts=264`: ~553ms/iter -> ~8.5ms/iter (~65x, exceeds the 50x target). Peak
RSS across the whole benchmark suite (including `contacts=264`): 26.4 MB (`/usr/bin/time -v`),
far below the 200 MB target and the ~3 GB the dense zero-cone block previously needed alone.

---

## Phase 2 — Per-cone regularization; never perturb equalities

**Problem (current code).**
- `KktSystem::updateScalingAndFactorize` subtracts `static_reg_A` on every (2,2) diagonal,
  zero-cone rows included.
- Iterative refinement runs against `K_`, which already contains the static regularization, so
  the regularization is never compensated.
- `factorizeWithRetry` bumps all diagonals uniformly, ×10 up to 18 times starting at
  `dynamic_reg_delta` (up to ~1e10).
- `escalateAndResolve` multiplies by 1000 per attempt.
- In hard cases the solver therefore reports success on a Newton system unrelated to the problem.

**T2.1 Settings.** Replace `static_reg_P`, `static_reg_A`, `dynamic_reg_*` with:
```cpp
struct RegularizationSettings {
  Scalar static_P      = 1e-8;   // (1,1) block
  Scalar static_nonneg = 1e-8;   // orthant rows of the (2,2) block
  Scalar static_soc    = 1e-8;   // SOC rows of the (2,2) block
  Scalar static_zero   = 0.0;    // equality rows: default exactly zero perturbation
  Scalar static_proportional = 2.2e-16 * 2.2e-16;  // times max|diag(K)|, added per block (0 on zero rows)
  Scalar dynamic_eps   = 1e-13;  // pivot threshold (sign-aware)
  Scalar dynamic_delta = 2e-7;   // pivot correction magnitude
  bool   dynamic_on_zero_rows = false;  // if false, a bad pivot on a zero row is a rank-deficiency signal (T2.5)
};
```
Keep the old field names as deprecated aliases for one release, if the maintainer wants
backward compatibility (ask).

**T2.2 Two matrices.** Maintain:
- `K_exact`: no static regularization anywhere. The zero-cone (2,2) diagonal is exactly 0 and
  the P diagonal is exactly P's.
- `K_fact`: `K_exact` plus per-block static regularization with the correct signs (+ on the x
  block, − on the z block). Only this matrix is factorized.
- Store the pattern once. Write values for both from the same slot map, with no `SparseMat` copies
  in the hot path (the current `K_try = K_` copy must go).

**T2.3 Refinement against the exact system.** Iterative refinement computes residuals with
`K_exact`. Settings: `refine_max_iter = 10`, `refine_reltol = 1e-13`, `refine_abstol = 1e-12`,
`refine_stop_ratio = 5` (stop if the residual does not shrink by this factor per step). Report the
final refinement residual in `Info`.

**T2.4 Dynamic regularization only inline and only per block.**

**Prove it or cut it (do this first, before any further `RegularizedLdlt` work).**
`RegularizedLdlt`'s per-pivot correction is 260 hand-written lines re-implementing QDLDL's
numeric factorization loop, has already produced two real bugs (a replace-vs-add pivot bug that
cost 15x+ IPM iterations on a real scene before being caught, and an AMD-permutation-direction
bug shared with `QdldlLdlt`), and still owes an attribution/NOTICE per `T0.2`. Its benefit is
unproven: the one real hard-instance test run against it (the frozen domino-scene KKT dump, see
`compareLinearSolverBackends()`) showed it matching `Eigen`/`Qdldl` bit-for-bit — no measured
win. Do **not** promote it to default on the strength of this task list alone. Instead:
- Keep `Settings::linear_solver` defaulting to `Qdldl` (matches Clarabel/QOCO, matters for the
  Phase 6 compat goal). `RegularizedLdlt` stays opt-in/experimental.
- Find or construct a real instance where `Eigen` and `Qdldl` both fail (`NumericalError` /
  `MaxIterations` / bad certificate) and `RegularizedLdlt` succeeds, using
  `compareLinearSolverBackends()` on a frozen dump the way the domino-scene report did it.
- If no such instance turns up after a reasonable search, delete `RegularizedLdlt` and
  `LinearSolverBackend::RegularizedLdlt` entirely rather than keep maintaining it — the static
  regularization + retry ladder + iterative refinement built in T2.1–T2.3 is then the whole
  story, and one fewer license-encumbered hand-rolled numeric kernel is a real win on its own.
- If it *is* kept: pass a per-pivot mask so correction applies only to pivots originating in P,
  orthant or SOC rows, with the expected sign (+ for x, − for z). Delete the uniform whole-matrix
  bump in `factorizeWithRetry` and the ×1000 `escalateAndResolve` path only for this backend —
  `Eigen`/`Qdldl` still need them, they have no per-pivot hook. If factorization is still
  unusable, return `Status::NumericalError` (or trigger T2.5). Count corrected pivots per block
  type and expose them in `Info` (`reg_pivots_P`, `reg_pivots_nonneg`, `reg_pivots_soc`,
  `bad_pivots_zero`).

Investigate the latest commit message ("Possibly found bug in custom ldlt decomposition"):
- Add a unit test comparing `RegularizedLdlt` with `QdldlLdlt` on random quasi-definite matrices
  where no correction is needed (must match to 1e-12).
- Add a test with forced corrections, checking that the factorization solves `K_fact + ΔD` exactly
  (reconstruct L D Lᵀ and compare).

**T2.5 Handling zero-cone pivots without regularization.**
- **Option A (required):** zero rows get 0 static regularization in `K_exact` but a tiny δ in
  `K_fact` if needed for factorization (setting `static_zero_factor_only = 1e-10`, applied only to
  `K_fact`). Refinement against `K_exact` removes its effect. This makes δ act as a
  preconditioner, not as a problem perturbation.
- **Option B (optional, behind `Settings::constrained_ordering`):** constrained ordering with the
  equality rows eliminated last, via SuiteSparse CAMD if available (CMake option), otherwise AMD on
  the non-equality indices followed by AMD on the equality rows. With A_E of full row rank and
  P + δ_P I + A_Iᵀ H⁻¹ A_I positive definite, the equality Schur complement is negative definite,
  so no zero-row regularization is needed at all. Benchmark the fill-in against Option A.
- **Rank deficiency (redundant equalities, hyperstatic contact):** if refinement against
  `K_exact` stalls or a zero-row pivot is bad (wrong sign or |d| < eps), flag
  `Info::equality_rank_deficient = true`. Then, only if `Settings::equality_proximal = true`, use a
  primal–dual proximal term on those rows: add −ρ to the zero-row diagonal consistently in both
  matrices and add −ρ(z − z_k) to the corresponding residual, with an outer proximal update
  z_k ← z at each IPM iteration (Friedlander–Orban style). This changes subproblems but not the
  fixed point. Default `ρ = 1e-8`. **Ask the maintainer before implementing this part**; it is
  a research design choice.

*Accept:*
- New test: a problem with 3 equalities, 1 SOC and 1 orthant block. Assert that the solution
  satisfies ‖A_E x − b_E‖∞ ≤ 1e-10 and that Newton directions satisfy the linearized equalities
  to 1e-12. Instrument through a test-only hook.
- New test: duplicated equality row (rank-deficient A_E). Must return `Solved`, or a clear
  `NumericalError` with `equality_rank_deficient = true` when the proximal option is off. Must
  never return a wrong "Solved".
- Benchmarks: no instance may stop converging.

---

## Phase 3 — Correct SOC scaling; replace clamps with safeguards

**Problem.** `SecondOrderCone::updateScaling` clamps β to [1e-8, 1e8], floors `v[0]`, and falls back
to `W = I` when |W| > 1e8. These silently break the NT identity W²z = s, so the direction is no
longer the central-path Newton step, and the method stagnates near the apex (separating contacts).
`applyWInv` uses a per-cone LU.

**T3.1 Closed-form NT scaling (verified numerically).** For s, z in the strict interior, with
J = diag(1, −1, …, −1):
```
a = sqrt(sᵀJs),  b = sqrt(zᵀJz),  η = sqrt(a/b)
s̄ = s/a,  z̄ = z/b,  γ = sqrt((1 + s̄ᵀz̄)/2),  w = (s̄ + J z̄)/(2γ)     // wᵀJw = 1
W   = η [ w0   w1ᵀ ; w1   I + w1 w1ᵀ/(1 + w0) ]                           // symmetric
W⁻¹ = J W J / η²          (flip the signs of the off-diagonal blocks, divide by η²)
H   = W² = η² (2 w wᵀ − J)
λ   = W z = W⁻¹ s
```
- Implement `applyW` and `applyWInv` as O(dim) rank-1 formulas, with no stored dense W and no LU.
- Compute `sᵀJs` as `(s0 − ‖s1‖)(s0 + ‖s1‖)` for accuracy near the boundary.
- Specialize dim 3 with fixed-size Eigen types.

**T3.2 Remove all clamps and the identity fallback.** Replace them with:
- **Interior guarantee:** the step length uses `max_step_fraction` (0.99) and a strict-interior
  check per cone after the update (`s0 − ‖s1‖ > 0` and `sᵀJs > 0` computed as in T3.1). If it
  fails, backtrack: α ← β_ls α with `linesearch_backtrack = 0.8`.
- **Centrality safeguard (new, beyond Clarabel for symmetric cones):** after a trial step, require
  for every cone block i: `min eigenvalue(λ_i ∘ λ_i) ≥ θ · μ` with `θ = 1e-4`. For the SOC, the
  eigenvalues of λ∘λ are (λ0 ± ‖λ1‖)². If this fails, backtrack.
- If α < `min_terminate_step_length = 1e-4` for two consecutive iterations, return
  `InsufficientProgress` with the best iterate (see T4.3).

**T3.3 Step length.** Keep the analytic quadratic-root SOC step. Add unit tests for:
- a direction tangent to the boundary,
- x exactly at the apex with dx pointing inward,
- dx = −x (α = 1),
- very large and very small scales (1e±8).

*Accept:*
- New test family "apex contacts": N contacts where half have zero normal force at the solution,
  and half are sticking with tangential force strictly inside the cone. Must converge to 1e-8 in
  ≤ 25 iterations, with no NaNs.
- Benchmarks converge. Report the iteration-count diff.

---

## Phase 4 — Termination, infeasibility, statuses

**T4.1 Relative termination in unscaled units.** Evaluate on unequilibrated quantities, with
x̂ = x/τ, ŝ = s/τ, ẑ = z/τ:
```
primal:  ‖A x̂ + ŝ − b‖∞      ≤ tol_feas · max(1, ‖b‖∞ + ‖x̂‖∞ + ‖ŝ‖∞)
dual:    ‖P x̂ + Aᵀ ẑ + q‖∞   ≤ tol_feas · max(1, ‖q‖∞ + ‖x̂‖∞ + ‖ẑ‖∞)
gap:     |pobj − dobj|        ≤ tol_gap_abs + tol_gap_rel · min(|pobj|, |dobj|)
```
Also add `tol_ktratio = 1e-6` (convergence requires κ/τ ≤ tol_ktratio). Compare the exact
formulas against Clarabel's source (`src/solver/core/.../info.rs`) and document any deviation in
`docs/design.md`.

**T4.2 Proper infeasibility certificates** (normalized, unscaled):
- **Primal infeasible:** bᵀz < 0, z ∈ K*, and ‖Aᵀz‖∞ ≤ tol_infeas_rel · (−bᵀz) (+ tol_infeas_abs).
- **Dual infeasible:** qᵀx < 0, ‖Px‖∞ ≤ tol_infeas_rel · (−qᵀx), and
  ‖Ax + s‖∞ ≤ tol_infeas_rel · (−qᵀx).

Return the normalized certificate in `Solution` (x, s, z scaled so that −bᵀz = 1, resp. −qᵀx = 1).

**T4.3 Statuses and best iterate.** Extend `Status` with `AlmostSolved`, `AlmostPrimalInfeasible`,
`AlmostDualInfeasible`, `MaxTime` and `InsufficientProgress`:
- Add `reduced_tol_*` settings (Clarabel-like defaults: feas 1e-4, gap 5e-5, infeas 5e-5,
  ktratio 1e-4).
- Track the best iterate by a merit function max(primal res, dual res, gap). Return it on
  `MaxIterations`, `MaxTime` and `InsufficientProgress`, and report `AlmostSolved` if it meets the
  reduced tolerances.
- Add `time_limit` (seconds, default +∞).

*Accept:* the existing `test_infeasible` cases pass. Add a near-infeasible friction case and a
dual-infeasible case (unbounded LP) with certificate checks. No false certificates on any benchmark.

---

## Phase 5 — Warm start (ConicXX feature Clarabel lacks)

**Problem.** `solve()` stores `warm_x_ = x_` etc. without dividing by τ, then resets τ = κ = 1 while
sᵀz ≈ 1e-9, which gives a badly uncentered start.

**T5.1** Store x/τ, s/τ, z/τ (in equilibrated units) after a successful solve.

**T5.2 Recentering.** Given the warm (x, s, z):
- Set s ← s + δ_s e and z ← z + δ_z e, with the smallest δ such that every cone block satisfies
  `min eig(s_i) ≥ ε_i` and `min eig(z_i) ≥ ε_i`.
- Then scale so that μ = (sᵀz + τκ)/(deg + 1) equals the target μ₀ = `warm_mu0` (default 1e-3,
  tunable), with τ = 1 and κ = μ₀.
- Zero-cone s stays exactly 0; zero-cone z is left free.
- Fall back to a cold start if the warm point's residuals exceed the cold point's.

**T5.3 Benchmark.** Add a benchmark: `FrictionChain` sequences where q and b are perturbed by
0.1–5% per "timestep" over 50 steps. Report cold and warm iterations per step.
*Accept:* warm ≤ cold on average for perturbations ≤ 1%, and never more than cold + 2.

---

## Phase 6 — Clarabel-compatible API

**T6.1 Arbitrary cone order.** Accept `std::vector<ConeT>` with
`ConeT = std::variant<ZeroConeT{dim}, NonnegativeConeT{dim}, SecondOrderConeT{dim}>` in any order.
Internally permute the rows of A and b into Zero → Nonneg → SOC order, and un-permute s and z on
output. Keep `ConeSpec` as a convenience.

**T6.2 `include/conicxx/clarabel_compat.h`.** A thin adapter mirroring the Clarabel C++ interface
so the downstream engine can switch solvers with a typedef:
- a `DefaultSolver`-like class constructed from (P, q, A, b, cones, settings),
- `solve()`, `solution()`, `info()`,
- `update_P / update_q / update_A / update_b`,
- settings with Clarabel names mapped onto ConicXX settings.

**Check the exact Clarabel C++ signatures and settings names against the Clarabel.cpp headers
(github.com/oxfordcontrol/Clarabel.cpp) before implementing; do not rely on memory.** Unsupported
Clarabel settings (e.g. chordal decomposition, other cone types) are accepted and ignored, with a
one-time warning.

**T6.3 Info fields.** Iterations, residuals, gap, μ, times, per-block regularization counts,
refinement residual, `equality_rank_deficient`.

*Accept:* a compat test that ports an existing Clarabel C++ usage example verbatim (modulo
include/namespace) and gets the same solution to 1e-7.

---

## Phase 7 — Allocation-free hot path

- **T7.1** Preallocate every temporary in `SolverImpl` (rhs, sol, xv, zv, Hz, λ, corrector, etc.).
  `Pmul` writes into a buffer. `KktSystem::solve` must not return `Vec` by value.
  *Accept:* a test with a counting allocator (or `-fsanitize` heap hooks) shows zero heap
  allocations inside `solve()` after the first call.
- **T7.2** `updateData`: when the pattern matches, copy values straight from `valuePtr()` into K
  through a precomputed index map (P upper → K slots, A → K slots). Remove the per-entry
  `SparseMat::coeff()` lookups.
- **T7.3** The backends currently rebuild the permuted matrix with `twistedBy(perm_)` on every
  factorize. Precompute the permuted value-index map once in `analyzePattern` and scatter values
  directly.
- **T7.4** Profile `FrictionChainXL` and `GroupLasso` after Phase 1 and report the top 5 hotspots.

---

## Phase 8 — Optional, needs maintainer sign-off

- **T8.1 Reduced system for contact.** Eliminate the block-diagonal SOC/orthant rows:
  `(P + δI + A_Cᵀ H_C⁻¹ A_C) Δx + A_Eᵀ Δz_E = …`. This gives a smaller KKT on (x, equality) and
  enables iterative/matrix-free solvers. Behind `Settings::kkt_mode = {Full, ReducedConic}`.
- **T8.2 Sparse SOC expansion** for large SOC blocks (Clarabel-style), only if the maintainer needs
  high-dimensional cones.
- **T8.3 Outer loop API** for the non-associated (De Saxcé) friction formulation: a sequence of
  warm-started convex SOCPs with an updated right-hand side. The outer update is supplied by the
  caller, so the library stays formulation-agnostic.

---

## 9. Tests to add (summary)

| Test | Phase |
|---|---|
| Structured cone scaling is numerically identical to dense (random s, z) | 1 |
| Memory/time regression for `FrictionChainXL 264` | 1 |
| Equality residual of solution and of Newton directions (unperturbed) | 2 |
| Rank-deficient / duplicated equalities | 2 |
| `RegularizedLdlt` equals `QdldlLdlt` without corrections; exact with forced corrections | 2 |
| NT identities: W²z = s, W W⁻¹ = I, λ = Wz = W⁻¹s, near-boundary accuracy | 3 |
| SOC max-step edge cases | 3 |
| Apex-contact family (separating plus sticking contacts) | 3 |
| Certificates for primal and dual infeasible problems; no false certificates | 4 |
| Warm-start sequence benchmark | 5 |
| Arbitrary cone order round trip; Clarabel compat example | 6 |
| Zero allocations in `solve()` | 7 |

## 10. Open questions for the maintainer

1. Which license for the project (T0.2)?
2. Keep the deprecated settings names for backward compatibility (T2.1)?
3. Implement the proximal treatment of redundant equalities (T2.5), or report and fail?
4. Is a SuiteSparse CAMD optional dependency acceptable (T2.5 Option B)?
5. Is P-block regularization acceptable, or should it also be refinement-compensated only?
   (Current plan: compensated via `K_exact`.)
6. Which Clarabel C++ version does the downstream engine use (for T6.2)?
7. **`RegularizedLdlt`: keep, demote, or cut (T2.4)?** Its only real-instance validation so far
   (the domino-scene dump) showed it matching `Eigen`/`Qdldl` exactly, i.e. no demonstrated
   benefit yet, against real cost (260 hand-written lines, two prior bugs, an unresolved license
   attribution). Default is being changed back to `Qdldl` pending a concrete counterexample where
   it alone succeeds; absent one, the plan is to delete it rather than extend it further.
