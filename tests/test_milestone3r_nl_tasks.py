from actmask.data.maniskill_3r_nl_tasks import MatchedDampedCaptureEnv, MatchedHystereticContainerEnv, MatchedDampedRotatingSlotEnv


def test_matched_final_task_families_expose_two_mechanisms_only():
    assert [cls.mechanism_count for cls in (MatchedDampedCaptureEnv,MatchedHystereticContainerEnv,MatchedDampedRotatingSlotEnv)]==[2,2,2]
    assert MatchedDampedCaptureEnv.success_radius==0.055
