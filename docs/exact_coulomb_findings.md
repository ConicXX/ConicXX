# Exact Coulomb friction with an interior-point method: findings

Status: experimental, not pursued further (2026-10-30). This note records what was implemented on the
branch `exact_coulomb`, what was measured, why the approach does not reach the robustness and speed
needed for CardilloCxx, and how to reproduce every result. The design of the implementation itself
is in `design.md`, section "Exact Coulomb friction (experimental)".

## Summary

- The exact discrete Coulomb problem (implicit De Saxcé term, Acary, Armand, Nguyen, CMAME 461
  (2026) 119227) is solved by a dedicated infeasible-start Mehrotra IPM (`CoulombIpm`) that reuses
  the cones, NT scaling, KKT pattern, ordering, equilibration, refinement and warm start of the
  convex solver. The Newton matrix differs from the convex one only in row 0 of every Coulomb block
  and is factorized by a new unsymmetric LDU on the quasidefinite structure.
- Small problems (single contacts, hyperstatic box, spinning ball in CardilloCxx) are solved to 1e-9.
- The first fast domino results (5 iterations, faster than the convexified solve) were wrong: the
  complementarity test accepted velocity errors of millimetres to centimetres per second, and the
  dominos bounced. In the exact problem every interior iterate separates every active contact
  (`s_i in int K <=> u_N > 0`), so every unresolved error injects energy.
- With a physically meaningful, **local** termination criterion (velocity error per contact) the
  solver fails in the first domino step already at 1e-5.
- Cause: the exact central path is regular and monotone in mu and ends at the correct solution, but
  it makes a **late sharp turn** (at mu ~ 1e-9, scaled) where the motion of a group of dominos changes by
  O(1). The turn is triggered by **degenerate contacts** (gap velocity and impulse both ~ 1e-6) that
  the path resolves only at tiny mu. A mu-parametrized IPM cannot follow the turn (all variants
  tried fail); pseudo-arclength continuation follows it but costs 336 linear solves for 8 dominos and
  670 for 20, against about 25 for the convexified solve of the whole scene.
- Conclusion: no cheap fix within the IPM framework is in sight; the convexified law stays the
  production path.

## 1. What was implemented

### ConicXX (branch `exact_coulomb`)

| File | Content |
|---|---|
| `cone_spec.h` | `ConeSpec::soc_coulomb` (per SOC, 1 = Coulomb cone), `hasCoulomb()`, validation |
| `settings.h` | `CoulombSettings { smoothing = 0.1, mu_floor = 0.1 }`, `Settings::coulomb` |
| `solution.h` | `ContactState {Undecided, Stick, Slip, TakeOff}`, `Solution::contact_states`, `Info::coulomb_smoothing` |
| `cones/coulomb_map.{h,cpp}` | De Saxcé term `l_i = sqrt(||s_bar_i||^2 + eps^2)`, its row Jacobian `L`, `B = I + E L` and `B^-1` |
| `kkt/quasidefinite_ldlt.{h,cpp}` | `factorizeUnsymmetric()`: simplicial and supernodal LDU on the analyzed symmetric pattern; magnitude-only dynamic regularization |
| `kkt/kkt_system.{h,cpp}`, `kkt/sparsity_map.h` | `updateScalingAndFactorize(cones, coulomb_blocks, row0)` writing the modified rows |
| `ipm/coulomb_ipm.{h,cpp}` | The IPM: smoothing, centering floor, local termination measure, path monitor, continuation fallback, contact classification |
| `ipm/solver_impl.{h,cpp}` | `solve()` dispatches to `solveCoulomb()` if the spec has Coulomb cones (warm/cold start choice, retry cold, unscaling) |
| `test/solver/test_coulomb.cpp` | Unit tests, see below |
| `docs/design.md` | Design record |
| `tools/exact_coulomb/` | Replay and analysis tools of this note, CardilloCxx patch |
| `docs/exact_coulomb/` | Theory chapter (`exact_coulomb.tex`, a section of the CardilloCxx theory document `LaTeX/theory_doc.tex`, whose preamble and macros it needs) and its bibliography entries (`exact_coulomb.bib`); its termination criterion and results are those of the intermediate global measure |

Without Coulomb cones nothing changes (all convex tests unchanged; 103 tests pass).

Unit tests (`test_coulomb.cpp`): CoulombMap derivative vs. finite differences, `B` and `B^-1`;
LDU vs. dense LU, supernodal vs. simplicial LDU; reduced vs. unreduced KKT solve; single contact
stick, slip and take-off against closed forms; convexified vs. exact solution; hyperstatic box;
compliant and frictionless rows; warm start; sticking with smoothing 0, 0.1 and 1.

### CardilloCxx

The changes are in `tools/exact_coulomb/cardillocxx.patch` (apply in the CardilloCxx root with
`git apply`). They are:

| File | Change |
|---|---|
| `src/config/config.hpp`, `config.cpp` | Key `conicxx.exact_coulomb` (bool, default off) -> `Config::conicxx_exact_coulomb` |
| `docs/chapters/using_the_engine/setup.rst`, `docs/chapters/solvers/interior_point.rst` | Table rows for the key |
| `src/physics/assembly/conicxx_assembler.{hpp,cpp}` | Constructor flag `exact_coulomb`; with it `b` holds only `S * biasImpulse` (no De Saxcé shift from `v^n`) and `soc_coulomb` is all 1 |
| `src/physics/solver/conicxx_solver.cpp` | Passes the flag; warm-start slack adds `||s_bar||` on Coulomb cones; `sameConeSpec` compares `soc_coulomb`; debug print names the mode; optional problem dump (`CARDILLO_CONICXX_DUMP=<dir>` writes `<dir>/step<k>.txt`, read by the tools) |
| `tests/test_convexified_friction.cpp` | `Variant {solver, exactCoulomb}`; the exact ConicXX variant is checked like projected Jacobi (exact Signorini law, rebound `e V`, no penetration) with tolerance 1e-7 |

Requirements: ConicXX from this branch (`-DFETCHCONTENT_SOURCE_DIR_CONICXX=<ConicXX checkout>`), all
scene executables rebuilt (Config layout change).

## 2. The termination criterion

The exact problem in ConicXX form, unit cones `S = diag(1, mu, mu)` applied by the caller:

    Px + q + A'z = 0,   Ax + s - b - E l(s) = 0,   s o z = 0,   s, z in K,

`s = u + E ||u_bar||` the modified velocity, `u` the relative velocity.

Three criteria were used in turn:

1. `max_i ||s_i o z_i|| <= tol_abs` (velocity x impulse). With impulses 1e-4 to 1e-2 N s on domino this
   allows normal velocities `tol / z_i`: active contacts separated with 3 mm/s on average, up to
   8 cm/s. The dominos bounced (reported from the scene).
2. Per cone `||s_i o z_i|| / (V ||z_i|| + Z ||s_i||) <= tol` with global scales
   `V = max(||x||, ||s||, ||q|| / max P_ii)`, `Z = max(||q||, ||z||)`. This still lets every contact
   keep a velocity error of `tol V` with V the fastest velocity of the scene (including angular
   velocities). Domino at 1e-5, 0.1 s: the contacts inject 2.5e-4 to 6e-4 J per step (exact)
   against 1.4e-5 J (convexified); the worst single contact (~1e-3 m/s) is similar in both. Exact
   7.0 iterations / 263 ms against convexified 10 / 257 ms per step.
3. Local (current): `v_i = ||s_i o z_i|| / (||z_i|| + m_i ||s_i||) <= tol_abs + tol_rel ||s_i||` with
   the effective mass `m_i` of the contact (from the diagonal of P). In the spirit of SCS
   (`eps_abs + eps_rel * scale`), but per contact: a resting contact must reach 1e-5 m/s
   independent of other bodies.

Remarks:

- The convexified HSDE loop has the same weakness as criterion 1/2; it happens to overshoot mu by
  about 100x in its last iteration.
- The bouncing observed in the scene at 1e-6 came from a binary with criterion 1; with the current
  build and that config (regularization 1e-8, 30 refinement steps) step 1 fails for both modes.
- With criterion 3, domino step 1 at 1e-5 ends `InsufficientProgress` after 24 iterations (about 9 s),
  `comp = 9.25` at mu = 1.6e-10.

## 3. Why the IPM stalls: analysis on subproblems

### Method

1. Dump the failing step from CardilloCxx (`CARDILLO_CONICXX_DUMP`): 1696 bodies, 13432 contacts,
   no bilateral rows; the whole scene is one connected component.
2. Solve it convexified (`replay ... 0`) and keep `x` as the boundary state.
3. Cut a k-hop neighbourhood around a contact flagged by the path monitor (`neighborhood.py`); the
   other bodies are fixed at the convexified velocities (`b -= A_out x_out`, `q += P_in,out x_out`).
   Fixing them at the stalled exact iterate instead gives an infeasible subproblem.
4. Reference solution by the De Saxcé fixed point (`fixpoint`): convex solves with
   `b + E ||s_bar||` until the shift settles (to about 1e-7, limited by the convex solves).
5. Trace the exact central path `F(x, s, z; mu) = (r_d, r_p, s o z - mu e) = 0` densely by
   pseudo-arclength continuation in `(y, log mu)` (`trace.py`, `trace2.py`).

| Cut around contact | Bodies / contacts | exact IPM (1e-5) | convexified (1e-7) |
|---|---|---|---|
| 6741, 0 hops | 2 / 28 | Solved, 17 it | Solved |
| 6741, 1 hop | 8 / 84 | InsufficientProgress | Solved, 17 it |
| 6741, 2 hops | 20 / 220 | InsufficientProgress | Solved, 19 it |
| 6741, 3 / 5 / 8 hops | 47 / 135 / 296 bodies | InsufficientProgress | Solved, 23-26 it |
| 6973, 1-8 hops | as above | InsufficientProgress | Solved (8 hops: NumericalError) |
| 719, 1 hop | 8 / 92 | AlmostSolved | Solved |
| 719, 2-8 hops | 21-311 bodies | InsufficientProgress | Solved |

The 8-body cut (`6741, 1 hop`) reproduces the failure and is the laboratory below.

### The central path is regular and reaches the solution

- Fixed point: 43 take-off, 33 slip, 8 stick contacts; objective `1/2 x'Px + q'x = -1.586738`.
- The traced path ends at mu = 7e-12 with objective -1.58673801 (same solution).
- mu decreases monotonically along the path: **no fold**, and no sign change of `det J_y` in the
  critical range. (Pivots of unexpected sign in the LDU occur, but in even numbers.)
- The stalled IPM iterate is a different, nearly resting state: objective -1.586618, primal residual
  3.2e-5, contact 65 nearly sticking instead of sliding with 7e-3 m/s.

### The late turn

Along the path (scaled mu):

| mu | distance of x to the solution (max) | contact 65 tangential velocity | contact 42 |z_T|/z_N |
|---|---|---|---|
| 1e-5 | 1.17 | 5.2e-3 (slip) | 0.06 |
| 1e-7 | 1.64 | 4.4e-6 (almost stick) | 0.87 |
| 1e-8 | 1.60 | 5.4e-7 | 0.88 |
| 4e-9 | 1.29 | 1.4e-4 | 0.90 |
| 2e-9 | 0.73 | 3.2e-3 | 0.91 |
| 1e-9 | 8.9e-3 | 6.95e-3 (slip) | 0.91 |
| 1e-11 | 5.7e-5 | 6.95e-3 | 0.88 |

Between mu = 4e-9 and 1e-9 the solution moves by O(1) in a light rotational direction (objective
change only 1.2e-4 of 1.59) and contact 65 switches from almost-stick to slip. Its own
complementarity scale is `|u_T| z_N ~ 6e-5`, so the switch is not decided by contact 65.

The contacts that change most in the turn are degenerate (not strictly complementary) at the
solution:

| Contacts | gap velocity (s) | impulse (z) | s z |
|---|---|---|---|
| 40-43 | 1e-6 to 2.5e-6 | 2e-6 to 9e-6 | ~1e-11 |
| 52-55 | ~4e-5 | ~2.5e-7 | ~1e-11 |

The path resolves them only at mu ~ 1e-9 to 1e-11, and their resolution flips the motion of the
cluster (contacts 32-35, 42 change their impulses, 65 starts to slide). In a convex problem such
contacts only slow the convergence (order 1/2 in the path monitor); in the nonmonotone exact problem
they decide the solution late.

The criterion is right to reject the state before the turn: it is almost optimal (objective 1.2e-4
off) but contacts separate with up to 4e-4 m/s. A looser criterion stops there, which is exactly the
energy injection of section 2.

### Long-step IPM variants (all on the cuts above, 100 iterations, no continuation)

| Variant | Result |
|---|---|
| Neighbourhood `centrality_theta` 1e-4, 1e-3, 1e-2, 3e-2, 0.1 (0.3, 0.5 on the 8-body cut) | Fails on all cuts; solved only 8-body cut 6741 at 1e-2 and cut 719 at 1e-2 / 3e-2 |
| Pure centering step (sigma = 1, no Mehrotra term) after a blocked step | Fails on all cuts but one (719 at 3e-2, 64 it) |
| Smoothing 0, 0.01, 1; floor 0, 1; 30 refinement steps | No effect |
| Continuation fallback of `CoulombIpm` | Restarts from the last iterate with centrality >= 0.1 mu, often iterate 0; the corrector's Newton steps have relative size 5-100 and are blocked |

Even pure centering steps reach only alpha = 0.01 to 0.07. Since `s o z` is bilinear, a centering
step can only be blocked by its second-order term `ds o dz`, i.e. by huge Newton directions: the
Jacobian is nearly singular at the turn, although its determinant keeps its sign. Before the turn
the per-cone orders of the path monitor are 1-5; at the turn 30-600.

### Cost of following the turn

`trace2.py`: dense pseudo-arclength, per-cone relative metric, corrector to a neighbourhood
(weighted Newton step < 1e-2), step control towards 3 corrector iterations:

| Cut | Steps | Linear solves | Of these in the turn |
|---|---|---|---|
| 8 bodies / 84 contacts | 100 | 336 | 164 (62 steps for mu in (1e-9, 5e-9]) |
| 20 bodies / 220 contacts | 176 | 670 | (more turns) |

For comparison: the convexified IPM solves the full step (1696 bodies) in about 25 factorizations.
The number of turns grows with the problem, so a continuation through all of them is far too
expensive for time stepping.

## 4. Conclusions and open directions

- The termination criterion of any solver for the exact law must be local per contact (velocity
  units); global normalizations hide the errors that inject energy. This also holds for the
  convexified solvers and could be adopted there.
- The exact IPM fails because of the nonmonotone structure, not because of the linear algebra: the
  KKT solves stay accurate (refinement residual 1e-12), and the factorization is not the bottleneck.
- Not tried, in decreasing plausibility:
  - friction homotopy at moderate mu (blend `t E l(s)` from the convexified solution, t: 0 -> 1); the
    same late switch may appear in t;
  - weighted central paths `s_i o z_i = mu w_i e` that resolve degenerate contacts earlier;
  - a cheap end game from the stalled point (few De Saxcé fixed-point iterations with warm-started
    convex solves; the plain fixed point needs ~100 outer iterations on the 8-body cut);
  - semismooth Newton polishing (considered too expensive in complexity).

## 5. Reproducing

```bash
# CardilloCxx with this ConicXX branch and the patch
cd CardilloCxx && git apply ConicXX/tools/exact_coulomb/cardillocxx.patch
cmake -S . -B build-coulomb -DFETCHCONTENT_SOURCE_DIR_CONICXX=$PWD/ConicXX && cmake --build build-coulomb -j
# dump the steps of a short domino run (config with conicxx.exact_coulomb = on, sim.T reduced)
mkdir -p dump && CARDILLO_CONICXX_DUMP=$PWD/dump build-coulomb/examples/domino run.config

# tools
cmake -S ConicXX/tools/exact_coulomb -B build-tools && cmake --build build-tools -j
T=ConicXX/tools/exact_coulomb
build-tools/replay dump/step0.txt 0 1e-5 1 0.1 0.1 x_convex.txt     # convexified, writes x s z
build-tools/replay dump/step0.txt 1 1e-5 3                           # exact, path monitor
python3 $T/neighborhood.py dump/step0.txt x_convex.txt 6741 1 cut.txt  # 8-body cut
build-tools/fixpoint cut.txt 1e-10 cut.fp                            # De Saxce fixed point
python3 $T/cones.py cut.txt cut.fp 42 65                             # contact states
python3 $T/kkt.py cut.txt cut.fp                                     # exact KKT residuals
python3 $T/trace.py cut.txt 0 trace.npz                              # dense path, fold/det check
python3 $T/trace2.py cut.txt 1e-12                                   # efficient tracer, solve count
```

`replay <dump> <exact 0/1> <tol> <verbose> [smoothing] [mu_floor] [out]` uses the CardilloCxx-like
settings (regularization 1e-8, 30 refinement steps, 100 iterations); `XP_THETA=<theta>` sets
`centrality_theta`. Dump format (`dumpread.h`, `dumpio.py`): P, q, A, b, compliance as triplets /
vectors, then `zero_dim nonneg_dim n_soc coulomb_flag`.
