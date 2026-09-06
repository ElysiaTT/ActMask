"""G2 bounded physical twins, private replay evidence, public allowlist export."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from binding_bench.schema import BindingDataset, dataset_arrays, save_dataset
from binding_bench.interventions import build_intervention_suite
from .rod import RodCOMEnv, save_snapshot
from .contracts import PUBLIC_KEYS, attempt_checks, sha256, utility


def generate(output: Path, *, twins_per_seed=8, seeds=(17001, 17002)):
    if not 2 <= twins_per_seed <= 16 or len(seeds) < 2 or len(seeds) > 4 or len(set(seeds)) != len(seeds):
        raise ValueError("CPU smoke requires 2-16 twins/seed and 2-4 distinct seeds")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; never overwrite replay evidence")
    private = output / "evaluator"
    private.mkdir(parents=True)
    rows, attempts = [], []
    env = RodCOMEnv()
    try:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for local_id in range(twins_per_seed):
                # IDs have no sign, COM value, seed, or branch semantics.
                uid = hashlib.sha256(f"cpu-smoke:{seed}:{local_id}".encode()).hexdigest()[:20]
                history = np.concatenate([-np.linspace(.15, .9, 4), np.linspace(.15, .9, 4)])
                candidates = np.concatenate([-np.linspace(.1, .95, 6), np.linspace(.1, .95, 6)])
                rng.shuffle(history)
                rng.shuffle(candidates)
                magnitude = rng.uniform(.035, .075)
                mechanisms = np.asarray([-magnitude, magnitude])
                rng.shuffle(mechanisms)
                contexts, effects, truths, trajectories = [], [], [], []
                for branch, com in enumerate(mechanisms):
                    env.set_anchor(float(com))
                    anchor = env.snapshot()
                    save_snapshot(private / "snapshots" / f"{uid}-{branch}.npz", anchor)
                    contexts.append(env.get_obs()["extra"]["body"].cpu().numpy()[0])
                    probes, probe_paths = zip(*(env.rollout(anchor, float(a)) for a in history))
                    outcomes, paths = zip(*(env.rollout(anchor, float(a)) for a in candidates))
                    effects.append(probes)
                    truths.append(outcomes)
                    trajectories.append(np.concatenate([probe_paths, paths], axis=0))
                effects, truths, contexts = map(np.asarray, (effects, truths, contexts))
                checks, marginal_error = attempt_checks(contexts, effects, truths)
                accepted = all(checks.values())
                np.savez_compressed(private / f"raw-{uid}.npz", mechanisms=mechanisms,
                                    history=history, candidates=candidates, history_effects=effects,
                                    candidate_effects=truths, context=contexts, trajectories=trajectories)
                attempts.append({"twin_id": uid, "seed": seed, "accepted": accepted,
                                 "checks": {k: bool(v) for k, v in checks.items()},
                                 "effect_marginal_error": marginal_error})
                if accepted:
                    rows.append((uid, seed, history, candidates, effects, truths, contexts))
    finally:
        env.close()
    (private / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n", encoding="utf-8")
    if not rows or set(row[1] for row in rows) != set(seeds):
        raise RuntimeError("G2 failed retention: inspect retained raw attempts; no retry or threshold tuning")
    count = len(rows)
    dataset = BindingDataset(
        dataset_id="maniskill-rod-com-cpu-smoke-v1", split="cpu_transport_smoke",
        history_actions=np.asarray([np.tile(row[2][None, :, None], (2, 1, 1)) for row in rows]),
        history_effects=np.asarray([row[4] for row in rows]),
        # Independent reset probes: equal time since anchor, not a temporal episode.
        history_times=np.zeros((count, 2, 8)), history_mask=np.ones((count, 2, 8), dtype=bool),
        context=np.asarray([row[6] for row in rows]),
        candidates=np.asarray([np.tile(row[3][None, :, None], (2, 1, 1)) for row in rows]),
        candidate_ids=np.asarray([[[f"{row[0]}-c{k}" for k in range(12)]] * 2 for row in rows]),
        true_utility=utility(np.asarray([row[5] for row in rows])),
        twin_ids=np.asarray([row[0] for row in rows]),
        episode_ids=np.asarray([[f"{row[0]}-{b}" for b in range(2)] for row in rows]),
        split_group_ids=np.asarray([row[0] for row in rows]), seed_ids=np.asarray([row[1] for row in rows]),
        candidate_effects=np.asarray([row[5] for row in rows]))
    schema_audit = save_dataset(private / "dataset.npz", dataset)
    build_intervention_suite(dataset, private / "suite", seed=44019,
                             source={"kind": "physical_cpu_smoke", "truth_visibility": "evaluator_only"})
    public = output / "public"
    public.mkdir()
    np.savez_compressed(public / "inputs.npz", **{k: v for k, v in dataset_arrays(dataset).items() if k in PUBLIC_KEYS})
    manifest = {"schema_version": "actmask-cpu-g2-v1", "scope": "transport_only_not_method_evidence",
                "attempts": len(attempts), "accepted": count, "seeds": list(seeds),
                "retention": count / len(attempts), "dataset_audit": schema_audit,
                "public_keys": sorted(PUBLIC_KEYS), "files": {str(p.relative_to(output).as_posix()): sha256(p)
                for p in sorted(output.rglob("*")) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--twins-per-seed", type=int, default=8)
    args = parser.parse_args()
    result = generate(args.output_dir, twins_per_seed=args.twins_per_seed)
    print(json.dumps({k: result[k] for k in ("attempts", "accepted", "retention")}, indent=2))
