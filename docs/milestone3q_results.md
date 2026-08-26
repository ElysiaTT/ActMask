# Milestone 3Q Results

## Phase 0: order-invariant shortcut audit

**Conclusion: the prior 3P matched pilot does not require temporal direction.**
All eight evaluated representations reached aggregate matched Top-1 1.0000
when trained on the 3P original training worlds: scalar logistic, shallow tree,
small MLP, DeepSets, sorted-frame history, velocity-magnitude-only,
timestamp-free temporal MLP, and unordered-multiset MLP. This is why 3Q uses
exact closed-loop reversal pairs with matched unordered statistics rather than
claiming 3P as signed-dynamics evidence.

The raw three-seed audit is under
`outputs/actmask/milestone3q_signed_dynamics/order_invariant_audit*.json`.

## Pre-generation status (historical)

Before the accepted rotating redesign, signed-order generation, full benchmark,
OOD, anti-shortcut interventions, and the final decision were pending. Their
rules were pre-registered in `docs/milestone3q_plan.md` before data generation.

## Phase-1 early-stop evidence (historical)

The initial four-world GPU-PhysX probe validates exact closed-loop matching, but
does **not** yet validate all required execution label flips. Cube signed pairs
flip. Container loop-reversal pairs flip, while its opposite-velocity,
acceleration-sign, and phase-delay templates remain `(failure, failure)` even
after two bounded fixes: monotone execution-completion labels and increasing
the common horizon from 16 to 20. Per the 3Q brief, scale-up stops here rather
than weakening the pair acceptance rule.

## Option-1 probe_v2 result

The authorized separate `SignedMovingWindowPlacement` variant replaces the
container signed mechanism without changing frozen 3P code. Its four signed
window mechanisms become reachable after removing an unintended candidate
startup delay. The probe consequently passes the new container stage, but does
not pass as a four-task benchmark: `RotatingTargetInteraction` still has a
signed loop-reversal pair with two failed GPU executions after two bounded
candidate-time-alignment fixes (fixed reachable-angle aim, then signed target
tracking). No full generation was started.

## Fixed-phase rotating capture-window probe

The separately authorized `FixedPhaseRotatingCaptureWindow` task passes its
four-world GPU-PhysX probe.  The first 20-step trace exposed a late accidental
re-intersection of the negative path at steps 14--15; before model evaluation,
the common task horizon was frozen to 13 steps.  This is the first bounded
correction, with no branch-specific action, phase, geometry, or success
threshold change.

The accepted probe has 320 examples and 40 signed counterfactual groups.  All
40 have a GPU-PhysX `(success, failure)` label flip: loop/reversal 16 groups,
opposite velocity 8, acceleration sign 8, and phase/delay 8.  Current state,
phase, action, static/unordered features, velocity magnitude, sorted history,
and horizon all match with maximum error `0.0` under the `1e-6` rule.

On held-out grouped signed pairs, static-only, action-only,
candidate-template-only, angular-speed-magnitude-only, unordered-history, and
sorted-history controls each have pair-order accuracy `0.5`.  The observable
signed finite-difference rule and ordered temporal MLP each reach `1.0`; exact
history reversal reduces both to `0.0` (delta `1.0`).  The model NPZ contains
only the seven fair observable keys.  All four task families pass the 8-way
CUDA smoke, and the complete regression suite passes **138 tests** in 56.48 s.
Raw artifacts are in `outputs/actmask/milestone3q_signed_dynamics/rotating_probe_v2/`.

## Full signed-dynamics generation and final gate

The complete GPU-PhysX generator accepted all four task families: 1,000 base
worlds and 20,000 examples per family (80,000 candidate executions total).
Every family has 2,500 accepted signed groups, success rate 0.75, and maximum
strict-match error `1.49e-08`, below the frozen `1e-6` tolerance. The four-task
run took 66.64 s on the RTX 4090 D. The rotating family retains its separately
frozen 13-step horizon; the other families use 20 steps.

The three-seed diagnostic and five-seed confirmation agree exactly. On grouped
ID signed test pairs, static MLP, action MLP, unordered-history MLP, and
sorted-history MLP all obtain `0.5` pair-order accuracy. The signed observable
finite-difference rule and ordered temporal MLP both obtain `1.0`; feeding the
same models exact reversed histories gives `0.0`, a delta of `1.0`. A
phase/delay-mechanism OOD holdout retains `0.5` for static/unordered and `1.0`
for signed/ordered methods. The five-seed run took 40.63 s with a 34,206,208 B
PyTorch peak allocation. The final regression is **138 passed** in 57.55 s.

The final gate passes. `full_four_task/final_gate.json` records the decision,
counts, audit maxima, control results, OOD result, and verification evidence.
