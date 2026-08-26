import torch
from actmask.experiments.milestone3r_training_conditions import pairwise_group_loss


def test_pairwise_group_loss_prefers_positive_score():
    groups=torch.tensor([0,0,1,1]);labels=torch.tensor([1.,0.,1.,0.])
    assert pairwise_group_loss(torch.tensor([2.,0.,2.,0.]),labels,groups) < pairwise_group_loss(torch.tensor([0.,2.,0.,2.]),labels,groups)
