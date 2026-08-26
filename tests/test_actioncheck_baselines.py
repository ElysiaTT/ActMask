import torch

from actmask.experiments.actioncheck.pilot_baselines import BASELINES, _factory


def test_all_preregistered_baselines_construct_and_score() -> None:
    batch = 3
    context = torch.zeros(batch, 6, 18)
    action = torch.zeros(batch, 8, 7)
    task = torch.zeros(batch, dtype=torch.long)
    context_mask = torch.ones(batch, 6, dtype=torch.bool)
    action_mask = torch.ones(batch, 8, dtype=torch.bool)
    for name in BASELINES:
        model = _factory(name, 18, 7, 8, 5)
        score = model(context, action, task, context_mask, action_mask)
        assert score.shape == (batch,)
        assert torch.isfinite(score).all()
