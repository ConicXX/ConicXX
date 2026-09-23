# Design notes

This file collects rationale that's too long for a source comment, per `CONICXX_AGENT_TASKS.md`'s
T0.5 (moving long design commentary out of source files) and T4.1 (documenting deviations from
Clarabel's reference behavior). It will grow as later phases add more; for now it covers Phase 4.

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
