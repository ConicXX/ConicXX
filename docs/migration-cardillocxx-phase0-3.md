# conicxx integration notes: Phase 0–3 changes

Written for: the CardilloCxx coding agent that integrates conicxx as a contact-solver backend.

Scope: everything committed to conicxx between commit `784e5cd` ("Possibly found bug in custom
ldlt decomposition", the last commit CardilloCxx was validated against) and the current `HEAD`
(Phase 0 hygiene, Phase 1 structured cone scaling, Phase 2 per-cone regularization, Phase 3 SOC
scaling + step-length safeguard, Phase 4 termination/infeasibility/statuses, Phase 5 warm start --
see the addenda at the end of this file for Phases 4 and 5 specifically, added after the rest of
this note was first written). See `CONICXX_AGENT_TASKS.md` in this repo for the full task list
and per-phase rationale/verification if you want more detail than this note gives.

## TL;DR

- **`conicxx::Solver`'s public methods and `ConeSpec` did not change at all.** If you only call
  `setup()` / `updateData()` / `setWarmStart()` / `solve()` / `solution()` and read
  `Solution::x/s/z/objective`, **no code changes are required** for those call sites to keep
  compiling and behaving the same way.
- **`Settings` changed**: several regularization/refinement fields were renamed and regrouped.
  If you construct a `Settings` with any of the old field names below, **it will no longer
  compile** — see the rename table.
- **`Status` gained one new enumerator** (`InsufficientProgress`). If you have an exhaustive
  `switch` over `Status` with no `default:`, add a case (or add a `default:`).
- **`Info` gained two new fields** (`kkt_refinement_residual`, `equality_rank_deficient`),
  purely additive — nothing to change unless you want to start reading them.
- **License**: conicxx is now formally GPL-3.0 (`LICENSE`, `NOTICE` added; previously the repo
  had no license file at all). You told me CardilloCxx is already GPL-3.0, so this should be a
  non-issue, but it's worth a sanity check on your end.
- **Numerical behavior changed in ways worth re-validating on real scenes**, even though no code
  changes are required for them to compile — see "Behavioral changes worth re-testing" below.

## 1. `Settings` field renames (compile-breaking if you set these)

`Settings::static_reg_P` / `static_reg_A` / `dynamic_reg_eps` / `dynamic_reg_delta` were replaced
by a nested `Settings::regularization` (`RegularizationSettings`) struct, broken out **per cone
type** instead of per matrix side. `Settings::refine_tol` was split into two fields.

| Old (gone) | New | Notes |
|---|---|---|
| `settings.static_reg_P` | `settings.regularization.static_P` | same meaning, same default (1e-8) |
| `settings.static_reg_A` | `settings.regularization.static_nonneg` **and** `.static_soc` | previously one value applied to *all* (2,2)-block rows including equalities; now split by cone type, both default 1e-8 |
| *(none previously)* | `settings.regularization.static_zero` | **new**, default **0.0** — equality (zero-cone) rows get *no* static regularization by default now (see behavioral note below) |
| `settings.dynamic_reg_eps` | `settings.regularization.dynamic_eps` | default changed 1e-14 → **1e-13** |
| `settings.dynamic_reg_delta` | `settings.regularization.dynamic_delta` | default changed 1e-7 → **2e-7** |
| `settings.refine_tol` | `settings.refine_reltol` (1e-13) **and** `settings.refine_abstol` (1e-12) | both are used together now: `tol = refine_abstol + refine_reltol * ||rhs||` |
| *(none previously)* | `settings.refine_stop_ratio` (default 5) | stops refinement early if the residual doesn't shrink by this factor in one step |
| `settings.refine_max_iter` (default 3) | same name, default changed to **10** | |

**Action:** grep your codebase for `static_reg_P`, `static_reg_A`, `dynamic_reg_eps`,
`dynamic_reg_delta`, `refine_tol` and update to the new names. If you were relying on the old
defaults implicitly (didn't set them), you get the new defaults automatically — see the
behavioral notes below for whether that matters for your scenes.

## 2. New `Settings` fields (Phase 3, additive — nothing breaks, but you may want to tune them)

```cpp
Scalar centrality_theta = 1e-4;           // step-length centrality safeguard threshold
Scalar linesearch_backtrack = 0.8;        // backtracking factor when a trial step fails
Scalar min_terminate_step_length = 1e-4;  // 2 consecutive steps below this -> InsufficientProgress
```

These replace what used to be silent clamps/fallbacks inside the SOC scaling math (see Phase 3
below) with an explicit, tunable backtracking line search. Defaults were validated against
conicxx's own benchmark suite (iteration counts unchanged from before Phase 3) but were **not**
validated against your specific domino/slinky/log-cabin scenes — if you see behavior changes on
those, these three settings are the first thing to look at.

## 3. `Status` enum: new `InsufficientProgress`

```cpp
enum class Status {
  Unsolved, Solved, PrimalInfeasible, DualInfeasible, MaxIterations, NumericalError,
  InsufficientProgress,  // new
};
```

Returned when the step-length safeguard (above) backtracks to a step below
`min_terminate_step_length` for two consecutive iterations — i.e. the solver detected it wasn't
making real progress and stopped instead of grinding through `max_iter` iterations doing nothing
useful, or silently accepting a step too close to a cone's boundary to trust. The returned
`Solution` holds the current (not further-refined) iterate.

**Action:** if you have an exhaustive `switch(status)` (no `default:`) anywhere — e.g. mapping
`conicxx::Status` to your own solver-agnostic status enum, alongside QOCO/Clarabel — add a case
for it. Recommended treatment: same bucket as `MaxIterations` (iterate may be usable but isn't
converged to tolerance; a caller warm-starting the next timestep can still use it, same as your
existing `MaxIterations` handling presumably already does).

## 4. `Info`: two new fields (purely additive)

```cpp
Scalar kkt_refinement_residual = 0;  // last KKT solve's refinement residual against the *exact*
                                      // (unregularized) system -- how well it was actually solved
bool equality_rank_deficient = false; // set if an equality-row pivot stayed bad even after the
                                      // one-shot factor-only preconditioner -- signals redundant/
                                      // hyperstatic equality constraints (rank-deficient A_E)
```

`equality_rank_deficient` is likely directly useful for you: your domino/log-cabin scenes are
exactly the kind of hyperstatic-constraint case this flag is meant to catch. If a timestep comes
back with a non-`Solved` status and this flag set, that's a specific, actionable diagnosis
("redundant equality constraints in this configuration"), not just a generic numerical failure —
worth surfacing distinctly in your own logging if you don't already.

## 5. Backend choice: `RegularizedLdlt` is now explicitly experimental

Nothing changed in `LinearSolverBackend` itself (`Eigen` / `Qdldl` / `RegularizedLdlt`, same
enum, `Qdldl` still the default). But worth flagging: Phase 2 downgraded `RegularizedLdlt`'s
status from "candidate default" to "opt-in experimental, prove-it-or-cut-it" (see
`CONICXX_AGENT_TASKS.md` T2.4) — its only real-instance validation so far (your earlier domino-
scene dump) showed it matching `Eigen`/`Qdldl` bit-for-bit, i.e. no measured benefit, against real
maintenance cost (a 260-line hand-written numeric kernel, two bugs found so far). **If
CardilloCxx pins `Settings::linear_solver = LinearSolverBackend::RegularizedLdlt` anywhere,
consider switching to the default `Qdldl`** unless you specifically know of a case where
`RegularizedLdlt` solves something the other two can't — if you find one, that's valuable
evidence for the conicxx side of this decision too, worth reporting back.

## 6. New public tool: `conicxx/backend_comparison.h`

```cpp
#include <conicxx/backend_comparison.h>
std::vector<BackendComparisonResult> compareLinearSolverBackends(P, q, A, b, cone_spec, settings);
printBackendComparison(results);
```

Runs the same problem through all three `LinearSolverBackend`s from fresh `Solver` instances and
reports status/iterations/residuals per backend — this used to require throwaway scratch code
(per your earlier domino-scene investigation) and is now a permanent library function. Useful the
next time you need to compare backends on a frozen repro instance.

## 7. Behavioral changes worth re-testing on real scenes (no code changes needed, but re-validate)

None of these require CardilloCxx code changes to keep compiling — they're changes to what
`solve()` actually computes. Re-running your existing scene suite (domino, slinky, log-cabin) is
the way to catch anything relevant.

- **Equality rows are no longer regularized by default** (Phase 2). Previously every (2,2)-block
  row, including equality/zero-cone rows, got a uniform `static_reg_A` (1e-8) added to its
  diagonal. Now equality rows get `static_zero` (0.0 by default) — mathematically more correct
  (an equality constraint is never perturbed unless it's genuinely rank-deficient, in which case
  `Info::equality_rank_deficient` fires instead), but it's a real change in what number gets
  factored for scenes with large equality blocks (e.g. your kinematic chain constraints). If a
  scene that previously converged now hits `NumericalError` or `equality_rank_deficient = true`,
  that's this change surfacing a genuine rank-deficiency in your constraint set that was
  previously being papered over by the uniform regularization — not a conicxx regression to
  report, but worth knowing the cause.
- **SOC (friction cone) scaling was rewritten** (Phase 3): the old code silently clamped/
  identity-fell-back when a contact's Nesterov-Todd scaling got extreme (near-apex, i.e. near-
  zero contact force — exactly your domino-toppling scenario). That's gone, replaced by an
  explicit backtracking step-length safeguard (Settings above). On conicxx's own benchmark suite
  this is iteration-count-neutral or better everywhere, including a dedicated "half the contacts
  sit exactly at the apex" stress test (new `ApexContacts` benchmark family). It was **not**
  re-validated against your real domino/log-cabin frozen repro data — given that's exactly the
  scenario this phase targets, it's worth re-running if you still have those captures.
- **Structured cone scaling (Phase 1)**: large equality/nonnegative blocks no longer allocate a
  dense `dim x dim` matrix internally (this was the ~3GB-for-one-block bug your earlier report
  helped find and fix, but that fix landed *before* `784e5cd` — mentioned here only because if
  you had any workaround in CardilloCxx for slow/memory-heavy large-equality-block timesteps,
  Phase 1 went further in the same direction and it's worth checking whether that workaround is
  still needed).

## 8. What did *not* change (through Phase 3; see the Phase 4 addendum below for one exception)

- `conicxx::Solver`'s constructor, `setup()` (all four overloads), `updateData()`,
  `setWarmStart()`, `solve()`, `solution()`, `settings()`/`setSettings()`.
- `ConeSpec` (`zero_dim`, `nonneg_dim`, `soc_dims`, `totalDim()`, `isValid()`).
- `Solution::x/s/z/objective`.
- `conicxx/convert.h`, `conicxx/types.h`.

---

## Addendum: Phase 4 (termination, infeasibility, statuses)

Same TL;DR as before: `Solver`/`ConeSpec` still didn't change. What did:

### `Settings` changes (compile-breaking if you set the removed field)

| Old (gone) | New | Notes |
|---|---|---|
| `settings.tol_infeas` | `settings.tol_infeas_abs` **and** `.tol_infeas_rel` | both default 1e-7; used together now (`dot_bz < -tol_infeas_abs && res_primal_inf < -tol_infeas_rel*(-dot_bz)`, and symmetrically for dual) |

New, additive fields: `tol_ktratio` (1e-6 -- gates when infeasibility certificates are considered,
not `Solved` itself, see below), `time_limit` (default +infinity, seconds), and the `reduced_tol_*`
family (`reduced_tol_feas`, `reduced_tol_gap_abs`, `reduced_tol_gap_rel`, `reduced_tol_infeas_abs`,
`reduced_tol_infeas_rel`, `reduced_tol_ktratio`) used by the new `Almost*` statuses below.

### `Status`: four new enumerators

```cpp
enum class Status {
  ..., InsufficientProgress,  // from Phase 3
  AlmostSolved, AlmostPrimalInfeasible, AlmostDualInfeasible, MaxTime,  // new, Phase 4
};
```

Same action as before: add cases to any exhaustive `switch(status)`. `AlmostSolved` /
`AlmostPrimalInfeasible` / `AlmostDualInfeasible` mean the solver hit `MaxIterations`/`MaxTime`/
`InsufficientProgress` but the *best* iterate seen during the run (not necessarily the last one --
see below) met the looser `reduced_tol_*` tolerances; the returned `Solution` is that best
iterate. `MaxTime` means `Settings::time_limit` was reached (default is +infinity, i.e. this can
never fire unless you set it).

### `Info` changes: one field added, two fields' *meaning* changed (not their names/types)

```cpp
Scalar merit = 0;  // new: max(primal_residual, dual_residual, |duality_gap|) at the reported
                    // iterate -- what MaxIterations/MaxTime/InsufficientProgress used to pick it
```

**`Info::primal_residual` and `Info::dual_residual` now mean something different**, even though
the field names, types, and rough "smaller is better, should be near tol_feas at convergence"
intuition are unchanged:
- Before: `||Ax+s-tau*b||_2 / tau` and `||A'z+Px+tau*q||_2 / tau`, computed in conicxx's internal
  *equilibrated* units (2-norm).
- Now (T4.1): the same quantities but in real, unscaled problem units, using the infinity norm,
  and normalized by `max(1, ||b||_inf + ||x||_inf + ||s||_inf)` (resp. the q/x/z analogue) --
  matching Clarabel's actual convergence-check convention, not just a relabeling.

**If CardilloCxx logs, thresholds, or compares these two `Info` fields against your own numbers**
(e.g. cross-checking against QOCO's or Clarabel's own reported residuals for the same problem),
expect the *numbers* to change even though nothing needs to change in your code for it to keep
compiling -- they're now actually the same kind of quantity Clarabel itself reports for the same
problem, which they weren't reliably before (see `docs/design.md`'s "Termination and
infeasibility" for the exact formulas, including two places where this session's implementation
deliberately deviates from a literal reading of Clarabel's Rust source, confirmed against the
maintainer, and documented there in detail).

### Behavioral note: the pre-Phase-4 duality-gap check had a real bug

The relative-gap tolerance was checked against `max(|primal_obj|, |dual_obj|)` where it should
have used `min(...)` (both the task's own spec and Clarabel's actual source agree on `min`). This
made the relative-gap tolerance *harder* to satisfy than intended whenever primal and dual
objective values differ noticeably during a solve -- fixed as part of Phase 4. Net effect on
conicxx's own benchmark suite was iteration counts improving on most instances (fewer iterations
needed now that the check isn't needlessly strict), one isolated instance needing one more
iteration. Worth knowing if you've tuned anything (e.g. `max_iter`) around the old convergence
behavior on your own scenes.

---

## Addendum: Phase 5 (warm start)

**No API changes** -- `Solver::setWarmStart()`, `Settings::warm_start`, and the automatic
"reuse last solve's iterate" behavior your per-timestep loop presumably already relies on are all
still there with the same names and signatures. One new `Settings` field:

```cpp
Scalar warm_mu0 = 1e-3;  // target centrality for a recentered warm start -- see below
```

**This phase is a straight correctness/quality fix to warm-starting itself, worth knowing about
even though nothing needs to change in your code.** The previous implementation had a real bug:
after a solve, it captured the raw homogeneous-embedding iterate as the next warm start without
dividing by `tau` (only correct when `tau` happened to converge to exactly 1, which it generally
doesn't) and reset `kappa=1` even though a converged `s'z` is typically ~1e-9 -- both together gave
the *next* solve's first Newton step a badly uncentered starting point. This is now fixed:
the captured point is normalized correctly, then explicitly recentered (shifted to strictly
interior, rescaled so its centrality `mu` hits `warm_mu0` exactly) before being used, and compared
against a fresh cold start every time -- if the recentered warm point isn't actually better, the
solve falls back to cold automatically. **If CardilloMPI's own timestep loop already relies on
`Settings::warm_start` (the per-timestep contact-force continuity you're presumably using this
solver for in the first place), you should see equal-or-fewer iterations per timestep after this
phase, not more** -- if you see the opposite on a real scene, that's worth reporting back, since
`test/solver/test_warm_start.cpp`'s own accept-criteria numbers (mean iterations at ≤1%
per-step perturbation must not exceed cold-start's) only cover a synthetic friction-chain
sequence, not your actual scene dynamics.

---

## Addendum: Phase 7 (allocation-free hot path)

**No API changes, and nothing behavioral to test either.** This phase was a pure internal
performance refactor -- eliminating heap allocations from `Solver::solve()`'s per-IPM-iteration
hot path (preallocated scratch buffers, `KktSystem`/backend `solve()` taking an out-parameter
instead of returning `Vec` by value, a precomputed value-scatter map replacing a `twistedBy()`
rebuild on every factorization) plus a `KktSystem::updateData()` micro-optimization (direct
`valuePtr()` index lookups instead of `SparseMatrix::coeff()` searches). `Solver`, `Settings`,
`Solution`, and `ConeSpec` are all unchanged; the one signature that did change
(`ConeSet::margins()`) is on an internal class CardilloMPI has no reason to touch directly (you
integrate through `Solver`, not `ConeSet`).

**Worth knowing if you're timestep-rate-sensitive:** repeated `solve()` calls on the same `Solver`
instance (the pattern your per-timestep loop already uses via `updateData()`/warm-starting) should
now be measurably cheaper per call, especially for larger contact counts, since the KKT
factorization no longer rebuilds a permuted copy of the matrix from scratch every IPM iteration.
No iteration counts changed (verified against the full benchmark suite) -- this only affects wall-
clock time per iteration, not how many iterations a given scene needs.
