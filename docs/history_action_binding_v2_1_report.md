# History–Action Binding Audit v2.1 最终报告

终止状态：**GPU_SMOKE_EXECUTION_ERROR**。本 Goal 没有训练任何 learned model。

## 最终五个问题

1. slot balance 是否严格修复：**是**。
2. Task A/Task B symbolic 是否通过：**True / True**。
3. GPU 数据是否真实、完整、可由 trace 重算：**未生成；冻结 generator 在创建环境前发生执行错误，不能声称真实GPU数据成立**。
4. binding 与 novel-candidate SysID 是否成立：**未在两个正式GPU任务上同时成立**。
5. 是否授权 learned verifier Goal：**不授权**。

## Official symbolic preflight

| Task | Slot min/max | Marginal max | Arrow | Nearest | kNN | Fourier | Binding drop | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| task_a_discrete_higher_order | 4/4 | 0.5000 | 0.5000 | 0.6875 | 0.7500 | 1.0000 | 0.3750 | True |
| task_b_continuous_nonlinear | 4/4 | 0.5000 | 0.5000 | 0.7598 | 0.7812 | 0.9199 | 0.4297 | True |

## v2 失败诊断与 v2.1 修复

v2 的124个 Task A miss全部位于105°/285°的四个理论 utility-tie templates；float32残差被固定metric当成有方向样本。v2.1 保持metric与0.90阈值不变，增加第二个probe半径、移除裁剪tie candidates，并用cyclic Latin square把每格固定为4。

Task B 的科学设计没有依据开发结果修改，official seed由冻结哈希导出。

## GPU 与正式审计

Official symbolic 已通过，但 frozen generate.py 在 gym.make 前因 PROJECT_ROOT 未导入而 NameError；按 smoke failure 规则停止，未运行 Task B smoke 或 formal pilot。

## 证据边界

Protected project/environment unchanged: `True`；scientific core frozen: `True`。
Candidate executions: `0/5000`；output bytes: `56322605/1000000000`。

## 可复制命令

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2_1.diagnose_v2
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2_1.development
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m unittest tests.test_history_action_binding_v2_1 -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2_1.freeze
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2_1.symbolic --official
HOME=/data/project/tzh/papers/ActMask/outputs/actmask/history_action_binding_v2_1/runtime_home PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2_1.generate --phase smoke --task task_a_discrete_higher_order  # exited 1 before gym.make
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2_1.run_v2_1 --stage record-smoke-error
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2_1.run_v2_1 --stage finalize
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m unittest tests.test_history_action_binding_v2_1 -v
```
