# TimeArrow direction: CPU pilot

Date: 2026-09-02

## Scope

This is a simulator-free sensitivity check. It tests whether the tracked audit
evidence supports the shortcut finding and whether an order-only redesign can
retain learnable temporal signal after endpoint, final-frame, unordered-history,
and candidate-ID shortcuts are removed. It does not establish real-robot
performance or learned-method headroom.

## Environment and artifact boundary

- Python 3.11.7
- NumPy 1.26.4
- scikit-learn 1.2.2
- CPU-only models; PyTorch is not used
- The repository does not contain the original `outputs/` tree, so the full
  85,600-row audit cannot be regenerated from raw model inputs in this clone.
  The tracked per-example compressed evidence is independently rescored.

## Tracked evidence recomputation

The script streams `audit/results/canonical_predictions.csv.gz` and recomputes
all counts without reading `audit/results/summary.json`.

| Split | Rows | Arrow x action | Candidate parity | Remapped parity |
|---|---:|---:|---:|---:|
| ID | 60,000 | 1.000 | 1.000 | 0.000 |
| Physical OOD | 8,000 | 1.000 | 1.000 | 0.000 |
| Temporal OOD | 5,600 | 1.000 | 1.000 | 0.000 |
| Held mechanism OOD | 12,000 | 1.000 | 1.000 | 0.000 |
| Total | 85,600 | 1.000 | 1.000 | 0.000 |

Candidate-ID remapping breaks the metadata parity rule but leaves the observable
arrow-action rule untouched. Randomizing candidate IDs alone therefore does not
remove the substantive shortcut.

## Small grouped CPU comparison

The canonical run uses 1,200 synthetic worlds per construction and holds out 360
complete worlds (1,440 rows). Candidate-ID-to-action-direction mappings are
randomized independently by world.

| Construction / method | Pair-order |
|---|---:|
| Legacy endpoint arrow x action | 1.000 |
| Legacy ordered MLP | 1.000 |
| Order-only endpoint arrow x action | 0.500 |
| Order-only action-only logistic | 0.531 |
| Order-only current+action logistic | 0.531 |
| Order-only final-two+action logistic | 0.531 |
| Order-only unordered+action logistic | 0.531 |
| Order-only ordered MLP | 1.000 |
| Order-only post-hoc order x action rule | 1.000 |

Across five additional 600-world runs (seeds 7301--7305), the legacy arrow rule
was 1.000, the redesigned endpoint rule was 0.500, the ordered MLP was 1.000,
and the post-hoc order rule was 1.000 in every run. The unordered baseline ranged
from 0.453 to 0.572 and averaged 0.497.

## Decision

1. The audit direction is correct: the current TimeArrow construction is exactly
   shortcut-solvable in the tracked evidence.
2. Matching endpoints and unordered histories while randomizing candidate IDs can
   remove the old endpoint and metadata shortcuts without destroying temporal
   learnability.
3. The CPU redesign does not establish learned-method headroom. A compact legal
   post-hoc order rule also reaches 1.000, so the correct admission decision is
   "temporal evidence present; learned comparison not yet admitted."
4. The next useful experiment is not a larger neural model on this toy task. It is
   either a preregistered multi-mechanism construction that defeats a frozen suite
   of analytic controls, or an external/public verifier benchmark with execution-
   grounded candidate outcomes.

## Reproduction

```powershell
python scripts\cpu_validate_timearrow_direction.py `
  --output audit\results\cpu_direction_pilot.json
```
