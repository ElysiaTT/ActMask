import numpy as np
from actmask.experiments.milestone3r_nl_probe import _score_velocity


def test_last_two_scoring_uses_only_estimated_velocity_and_shared_action():
    velocity=np.zeros((2,20),np.float32);actions=np.ones((2,3,3),np.float32)
    assert np.array_equal(_score_velocity(velocity,actions),np.zeros(2,np.float32))
