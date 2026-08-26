# History–Action Binding Audit v2 报告

终止状态：**SYMBOLIC_PREFLIGHT_FAIL**。按照预注册，没有运行 GPU smoke/pilot，没有生成 candidate rollout，也没有训练 learned verifier。

## 最终五个问题

- v2 数据是否可信：**symbolic evidence 完整可复算；真实 simulator dataset 未生成，因此不能声称真实数据已通过。**
- marginal shortcut 是否排除：**在 symbolic preflight 中是。** current/action/result/index corrected TPA 均为0.5；但这不是正式 GPU 结论。
- action–result binding 是否必要：**尚未建立。** Task A Fourier SysID 只有0.8789，低于冻结0.90，且 slot balance 未通过。
- novel candidate SysID 是否成立：**未同时成立。** Task A未达阈值；Task B为0.9082。
- 是否授权下一个 learned verifier Goal：**不授权。**

## Symbolic preflight

| Task | Marginal max | Arrow | Fourier SysID | Binding-break | Drop | Slot balance | Result multiset |
|---|---:|---:|---:|---:|---:|---:|---:|
| task_a_discrete_higher_order | 0.5000 | 0.5000 | 0.8789 | 0.5195 | 0.3594 | False | True |
| task_b_continuous_nonlinear | 0.5000 | 0.5000 | 0.9082 | 0.4889 | 0.4193 | False | True |

slot table 的冻结期望是每个 cell=4；实际两任务均为 min=3/max=5。没有改 sampler 后重跑。Task A Fourier corrected TPA=0.87890625；没有降低0.90阈值。

其余设计信号正确：两个任务 action/result/utility marginals 精确相同，Arrow=0.5，top-1 crossing=1.0，pair-preserving变化=0，binding-break drop分别0.3594/0.4193。

## v1 到 v2

| Version | Stage/Gate | Marginal | Arrow | SysID | GPU candidates | Training |
|---|---|---:|---:|---:|---:|---:|
| v1 | SHORTCUT_SATURATED | result/action shortcuts saturated | Task A=1.0 | explicit fits saturated | 2336 | no |
| v2 Task A | SYMBOLIC_PREFLIGHT_FAIL | 0.5000 | 0.5000 | 0.8789 | 0 | no |
| v2 Task B | SYMBOLIC_PREFLIGHT_FAIL | 0.5000 | 0.5000 | 0.9082 | 0 | no |

v1 的 raw TPA 混入 branch总体容易度，NDCG又奖励两个branch共享的候选幅值排序。v2 corrected TPA先在branch内center，再逐candidate比较 twin；branch常数分数严格为0.5。量化复盘见 `v1_failure_postmortem.json`。

## 为什么没有 trace

目标明确规定 symbolic preflight 失败后立即停止且不得运行 GPU pilot。`generate.py` 是 fail-closed 可执行入口；调用会写 refusal evidence 并以非零状态退出。缺少真实 trace 是遵守终止条件的结果，而不是把 symbolic 结果冒充 simulator evidence。

## 修复建议（只能新建 v2.1）

1. 用预先证明的 Latin-square slot permutation 取代当前 affine seed mapping；在新版本运行前符号证明每个slot/template cell严格相等。
2. 不降低SysID阈值；重新设计 Task A probe/candidate geometry或提高可识别性，并在新版本预注册后只运行一次 official preflight。
3. 保留本 v2 全部结果作为失败证据，v2.1 使用新路径，不能覆盖。

## 可复制命令

```bash
PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2.postmortem
PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2.symbolic
PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2.generate --phase smoke --task task_a_discrete_higher_order  # expected exit 3
PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2.run_v2 --stage finalize
PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m unittest tests.test_history_action_binding_v2 -v
```
