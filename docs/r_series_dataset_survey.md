# R-Series public real-robot dataset survey

The survey is saved in machine-readable form at
`outputs/actmask/r_series_real_robot_verification/dataset_survey.json`.
It was performed before any large download.

| Candidate | Public status / license | Smallest useful path | Robot and modalities | Outcome/contact structure | R-Series decision |
| --- | --- | --- | --- | --- | --- |
| DROID | Public Google Cloud release; the official paper reports CC-BY 4.0; policy code is MIT | Official `droid_100`: 100 episodes / 2 GB (full RLDS: 1.7 TB) | Franka Panda, three RGB views, action, joint/cartesian/gripper state, language | RLDS debugging sample has reward/terminal markers but no balanced explicit failure or torque field; raw DROID has richer metadata/torque proxies | **Primary**: ideal-sized, official, multimodal logged-consistency prototype |
| RH20T | Public; license varies by scene (CC-BY-SA or CC-BY-NC) | UR5 RGB cfg package about 4.4 GB plus extraction; RGB-D much larger | UR5 cfg3/4; RGB-D, joint/TCP/gripper, F/T, audio; tactile in cfg7 | completion-quality codes, calibrated multiview, direct physical signals; no counterfactual replayer | Deferred: stronger physical signals but unsafe extraction margin under current disk budget |
| RoboMIND v1 | Public page but gated, Apache-2.0 release terms reported | >1 TB, large archive shards | Includes UR5e, multiview, state and language | about 5K failure trajectories/reasons | Deferred: promising failures, but gated and too large |
| RoboMIND 2.0 | Public ModelScope release, Apache-2.0 reported | Individual sim HDF5 task can be ~4.03 GiB | Six embodiments incl UR5e/UR, RGB/RGB-D, language, state and some force/tactile | digital twin and simulator, but public logged tree mainly success and Isaac Sim requires much more local infrastructure | Deferred: best strict re-execution direction once infrastructure expands |
| BridgeData V2 | Public CC-BY 4.0 data / MIT code | no compact official subset; TFDS ~132.5 GB | WidowX, RGB/action/state/language | contact-rich tasks but no explicit outcome/contact/F/T field in standard schema | Deferred for disk and label limits |
| Open X ASU TableTop | Public Open X release; obey original dataset terms | 110 UR5 episodes / 0.773 GB | UR5, RGB, proprioception, action, language | no explicit success/contact/F/T field | **Backup**: compact UR5 logged-consistency fallback |

The selected primary task is therefore **not** named counterfactual success
prediction.  Its valid first claim is action-conditioned consistency with a
logged real-robot continuation.  If future state, task-family or candidate
source is used as an input, it is logged as an oracle or leakage diagnostic,
not a fair method input.

Authoritative source links:

- [DROID documentation](https://droid-dataset.github.io/droid/the-droid-dataset)
- [DROID release/download notes](https://github.com/droid-dataset/droid_policy_learning)
- [RH20T homepage and API](https://rh20t.github.io/), [RH20T API](https://github.com/rh20t/rh20t_api)
- [RoboMIND v1 card](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND), [RoboMIND 2.0](https://log2r.github.io/RoboMIND2.0/), [RoboMIND-Sim](https://github.com/Open-X-Humanoid/RoboMIND-Sim)
- [BridgeData V2](https://rail-berkeley.github.io/bridgedata/)
- [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment)
