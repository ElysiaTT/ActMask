# Counterfactual Action-Effect Binding Audit v1

Date: 2026-09-03
Status: benchmark implementation **READY**; learned-method claim **NOT READY**

## What changed

The project no longer treats a high history-conditioned accuracy as evidence
that a verifier learned dynamics. The benchmark now separates two questions:

1. **Causal-use integrity:** does the output depend on which action produced
   which effect?
2. **Decision usefulness:** does that dependence select better candidate
   actions than transparent system-identification baselines?

The first question is tested by matched twins and surgical interventions. The
second is tested with normalized candidate regret, exact-choice Top-1,
per-seed wins, a hierarchical bootstrap, OOD splits, and strong baselines. A
method must pass both tracks on every preregistered split.

## Historical result ledger

| Version | What happened | Valid interpretation | Forbidden interpretation |
|---|---|---|---|
| learned CPU v1 | Unbound TPA was algebraically fixed at 0.5; the bound model received the generator's exact sufficient Fourier basis and reached near-one TPA. | The metric, matched twins, and binding-erasure plumbing work. | A generic learned verifier discovered transferable dynamics. |
| CPU v2 | Raw paired model used binding but lost to generic Poly2/RBF system identification. | A learned pair encoder is not automatically more useful than fitting each history directly. | Learned verification improved candidate choice. |
| CPU v3 | Hybrid beat local-linear regret on ID by 19.7%, but failed parameter OOD, sparse-history uncertainty, noise OOD, and held-structure gates; it was 8.4% worse under noise OOD. | Training-only diagnostics can help ID, but the selection/calibration rule does not transport robustly. | The hybrid is a robust method improvement. |

Authoritative reports are
`audit/binding_cpu_learned_v1_saturation_diagnosis.md`,
`audit/binding_cpu_v2_antisaturation_report.md`, and
`audit/binding_cpu_v3_sysid_residual_report.md`. The frozen terminal decision
remains `CPU_V3_NO_GO`. No existing learned verifier result may be presented as
a method gain.

## Package boundary

`binding_bench/` is NumPy-only and intentionally does not import `actmask`,
PyTorch, a simulator, or a model implementation.

| Component | Responsibility |
|---|---|
| `schema.py` | Versioned dataset and prediction containers, shape/finite/identity/hash validation, and train-evaluation digest collision rejection. |
| `interventions.py` | Deterministic complete-pair permutation, effect-only binding break, twin-history swap, candidate-slot permutation, and semantic invariants. |
| `adapters.py` | Gives an external method a public view with candidate truth and candidate effects removed, then saves versioned predictions. |
| `metrics.py` | Candidate-ID alignment, tie-aware regret/Top-1/TPA, difficulty bins, shortcut diagnostics, per-seed metrics, and seed-then-twin bootstrap. |
| `evaluator.py` | Chooses the strongest declared baseline and applies frozen method and intervention gates for one split. |
| `benchmark.py` | Audits twin/episode/scene-group disjointness and requires all admission splits to pass. |

## Dataset schema

The version string is `actmask-binding-dataset-v1`. A split uses dimensions
`T` twins, two branches, padded history `H`, candidates `K`, action dimension
`A`, effect dimension `E`, and public-context dimension `C`.

| Field | Shape | Visibility | Meaning |
|---|---:|---|---|
| `history_actions` | `[T,2,H,A]` | method | Executed probe actions or action chunks. |
| `history_effects` | `[T,2,H,E]` | method | Aligned measured state/visual-effect deltas. |
| `history_times` | `[T,2,H]` | method | Public timestamps or normalized probe times. |
| `history_mask` | `[T,2,H]` | method | Valid padded action-effect pairs. |
| `context` | `[T,2,C]` | method | Current public observation/goal features; hidden mechanism variables are forbidden. |
| `candidates` | `[T,2,K,A]` | method | Candidate actions; matched branches expose the same candidate-ID set. |
| `candidate_ids` | `[T,2,K]` | identity only | Stable identifiers used only to align slots; an ID-only model is a shortcut control. |
| `true_utility` | `[T,2,K]` | evaluator only | Simulator-executed candidate utility, larger is better. |
| `candidate_effects` | `[T,2,K,E]` | evaluator only | Optional rollout effect truth for calibration. |
| `twin_ids` | `[T]` | identity only | Unique matched-twin identity. |
| `episode_ids` | `[T,2]` | audit only | Source episodes; never split across partitions. |
| `split_group_ids` | `[T]` | audit only | Scene/object/mechanism group; never split across partitions. |
| `seed_ids` | `[T]` | audit only | Independent generation or evaluation seed. |

Every persisted array, including dtype and shape, enters a canonical SHA-256.
Predictions must name that digest. If an evaluation digest appears in a
method's declared training digests, validation fails closed.

The prediction schema is `actmask-binding-prediction-v1`. It contains
larger-is-better `[T,2,K]` scores, the exact candidate/twin identities,
dataset/split/intervention labels, dataset digest, declared training digests,
and optional predicted effects/uncertainty. Metrics align by candidate ID, so
slot order cannot change a score.

The callable adapter receives `dataset.public_view()`: `true_utility` has zero
elements and `candidate_effects` is absent. This is an accidental-leakage
boundary, not a security sandbox. A public leaderboard should keep evaluator
truth server-side.

## Intervention semantics

| Case | Changes | Must remain invariant | Expected binding-aware response |
|---|---|---|---|
| `pair_preserving` | Reorders complete action-effect-time tokens. | Token multiset, candidates, truth. | Regret/TPA unchanged within 0.005. |
| `binding_breaking` | Deranges effects while actions/times stay fixed. | Action and effect marginals, candidates, truth. | Regret rises and TPA falls by at least 0.05. |
| `twin_history_swap` | Swaps complete histories between matched branches. | Context, candidates, truth. | Regret rises by at least 0.05. |
| `candidate_slot_permutation` | Permutes candidate data and truth together. | Candidate identity-to-value mapping. | Regret unchanged within 0.005. |

The evaluator also reports constant-score branch fraction, centered-identical
twin fraction, and mean predicted tie count. These expose candidate-constant,
branch-constant, and ID/slot strategies that can make aggregate metrics look
better without solving the task.

## Metrics and decision rule

Normalized regret is computed per branch as

`(best true utility - selected true utility) / (best - worst true utility)`.

Prediction ties receive the mean utility and fractional Top-1 credit rather
than arbitrary `argmax` credit. TPA centers candidate scores within each branch
before comparing the two twin branches, so a branch-wide offset cannot pass.

The default one-split gate requires:

- at least five seeds and 32 twins per seed;
- at least 10% regret reduction against the strongest declared non-oracle
  baseline, positive in at least four of five seeds, with bootstrap lower 95%
  bound above zero;
- at least +0.03 Top-1;
- binding break `regret +0.05` and `TPA -0.05`;
- twin swap `regret +0.05`;
- pair and candidate-slot changes no larger than 0.005;
- anti-saturation: original TPA below 0.95 and regret at least 0.01.

The full benchmark separately requires ID, parameter OOD, sparse history,
noise OOD, and held mechanism/structure to pass. `METHOD_GO` on one split is
never GPU authorization.

## CPU example

```powershell
python scripts/run_binding_bench_example.py `
  --output-dir audit/results/binding_bench_example
python -m pytest tests/test_binding_bench.py -q
```

The frozen example produced `TPA=0.8813`, `normalized regret=0.0351`, and
`Top-1=0.6438`. Binding breaking changed these to `TPA=0.4569` and
`regret=0.5184`; pair and slot changes were exactly zero. The example therefore
tests the harness without reproducing the suspicious 0.5-versus-near-1 regime.
Its `METHOD_GO` is only a self-test against a deliberately weak result-mean
baseline, not an ActMask research result.

For one external split use `scripts/evaluate_binding_bench.py`; for a frozen
multi-split evaluation plan use `scripts/evaluate_binding_benchmark.py`.
