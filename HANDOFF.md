# ActMask handoff

## Snapshot

- Working copy: `/data/project/tzh/papers/ActMask`
- This directory did not contain a Git repository before handoff.
- A new repository has been prepared locally; no GitHub remote or push has
  been configured yet.
- The project is a CPU-only synthetic milestone. It is not a validated robot,
  simulator, camera, CUDA, or real-world result.

## Reproduce the current milestone

From the repository root, use Python 3.11 and the pinned dependencies in
`requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m actmask.training.train --config configs/actmask/toy_cpu.yaml
python -m actmask.eval.evaluate --config configs/actmask/toy_cpu.yaml
python -m actmask.visualization.visualize_masks --config configs/actmask/toy_cpu.yaml
pytest tests/test_actmask_shapes.py
```

The training checkpoint, metrics, plots, simulator cache, and submission
bundles are generated material. They remain in the container for reference but
are excluded from the source repository by `.gitignore`.

## Repository layout

- `actmask/`: implementation and synthetic data generation.
- `configs/`: experiment configuration.
- `tests/`: shape, training, evaluation, and milestone checks.
- `docs/`: design, validation, and limitation notes.
- `paper/` and `paper_icra2027/`: draft paper/materials; do not treat drafts as
  accepted claims.
- `audit/`, `scripts/`, and `release_candidate/`: audit and packaging tools.

## Next owner checklist

1. Review the initial source-only commit and confirm licensing for every
   vendored or copied component.
2. Run the CPU smoke commands above and record Python/PyTorch versions.
3. Decide whether paper drafts and release packaging belong in the public
   repository or a separate private archive.
4. Create the GitHub repository, add its remote, and push the reviewed initial
   commit.
5. Keep the synthetic-only limitation visible until real/simulator evidence is
   independently validated.
