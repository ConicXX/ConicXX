# conicxx integration notes: Phase 0–3 changes

Written for: the CardilloCxx coding agent that integrates conicxx as a contact-solver backend.

Scope: everything committed to conicxx between commit `784e5cd` ("Possibly found bug in custom
ldlt decomposition", the last commit CardilloCxx was validated against) and the current `HEAD`
(Phase 0 hygiene, Phase 1 structured cone scaling, Phase 2 per-cone regularization, Phase 3 SOC
scaling + step-length safeguard). See `CONICXX_AGENT_TASKS.md` in this repo for the full task
list and per-phase rationale/verification if you want more detail than this note gives.

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

## 8. What did *not* change

- `conicxx::Solver`'s constructor, `setup()` (all four overloads), `updateData()`,
  `setWarmStart()`, `solve()`, `solution()`, `settings()`/`setSettings()`.
- `ConeSpec` (`zero_dim`, `nonneg_dim`, `soc_dims`, `totalDim()`, `isValid()`).
- `Solution::x/s/z/objective`, and all pre-existing `Info` fields.
- `conicxx/convert.h`, `conicxx/types.h`.
