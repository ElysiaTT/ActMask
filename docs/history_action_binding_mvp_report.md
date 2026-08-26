# History–Action Binding Audit MVP 报告

终止判定：**SHORTCUT_SATURATED**；learned verifier training：**不授权**。

## 四个最终问题

- 数据完整可信：**是**。两个 task 的 trace label/utility 100% 重算、twin candidates 逐字节相同、matched current observable 逐字节相同，且同 seed/same batch 复跑全部数组一致。这里的“可信”指本地执行证据和内部可重算性，不扩大为跨机器 bit-exact。
- action–result binding 必要：**not uniquely necessary; simple shortcuts saturate at least one formal task**
- novel-candidate generalization：**not established on both tasks**
- 下一阶段 learned model training：**不授权**。

## 正式 pilot

两个 64-twin pilot 在统一打开 outcome 之前已全部生成；没有 outcome-based resampling、过滤 00/01/10/11、改阈值或改候选分布。

| Task | Gate | 00/01/10/11 | flip ratio | best marginal | best explicit fit | fit TPA | binding-break drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| task_a_discrete_mode | SHORTCUT_SATURATED | 384/64/64/0 | 0.250 | class_prior | linear_least_squares_response | 1.000 | 1.000 |
| task_b_continuous_response | SHORTCUT_SATURATED | 448/34/30/0 | 0.125 | result_only | linear_least_squares_response | 0.500 | 0.000 |

## 全部无训练 baseline（正式 original binding）

### task_a_discrete_mode

| Baseline | Balanced acc. | Twin pref. acc. | NDCG | Regret | Utility MAE |
|---|---:|---:|---:|---:|---:|
| class_prior | 0.500 | 0.500 | 0.632 | 0.727 | 0.01521 |
| current_only | 0.500 | 0.500 | 0.632 | 0.727 | 0.01521 |
| candidate_index | 0.487 | 0.500 | 0.632 | 0.727 | 0.47018 |
| action_only | 0.714 | 0.500 | 0.711 | 0.500 | 0.03038 |
| result_only | 0.500 | 0.500 | 0.632 | 0.727 | 0.02530 |
| arrow_action_compatibility | 1.000 | 1.000 | 1.000 | 0.000 | 0.00005 |
| repeat_last_failed_action | 0.393 | 0.250 | 0.579 | 0.816 | 0.67893 |
| nearest_historical_action | 0.929 | 1.000 | 0.948 | 0.168 | 0.02169 |
| knn_action_result_lookup | 0.500 | 1.000 | 1.000 | 0.000 | 0.01519 |
| linear_least_squares_response | 1.000 | 1.000 | 1.000 | 0.000 | 0.01519 |
| quadratic_least_squares_response | 1.000 | 1.000 | 1.000 | 0.000 | 0.01519 |
| oracle_latent_rollout | 1.000 | 1.000 | 1.000 | 0.000 | 0.01519 |

### task_b_continuous_response

| Baseline | Balanced acc. | Twin pref. acc. | NDCG | Regret | Utility MAE |
|---|---:|---:|---:|---:|---:|
| class_prior | 0.500 | 0.500 | 0.661 | 0.678 | 0.01240 |
| current_only | 0.500 | 0.500 | 0.661 | 0.678 | 0.01240 |
| candidate_index | 0.500 | 0.500 | 0.661 | 0.678 | 0.46804 |
| action_only | 0.433 | 0.500 | 0.978 | 0.084 | 0.02065 |
| result_only | 0.500 | 1.000 | 0.661 | 0.678 | 0.01875 |
| arrow_action_compatibility | 0.433 | 0.500 | 0.978 | 0.084 | 0.00547 |
| repeat_last_failed_action | 0.900 | 0.875 | 0.977 | 0.000 | 0.70115 |
| nearest_historical_action | 0.467 | 0.500 | 0.973 | 0.084 | 0.01490 |
| knn_action_result_lookup | 0.500 | 0.500 | 0.978 | 0.081 | 0.01314 |
| linear_least_squares_response | 1.000 | 0.500 | 1.000 | 0.000 | 0.01493 |
| quadratic_least_squares_response | 1.000 | 0.500 | 1.000 | 0.000 | 0.01430 |
| oracle_latent_rollout | 1.000 | 0.500 | 1.000 | 0.000 | 0.01425 |

## Gate 解释

`SHORTCUT_SATURATED` 是预注册的终止状态：任一 simple no-training shortcut 的 balanced accuracy、Twin Preference Accuracy 或等价 NDCG ≥0.90。达到它以后没有训练 learned verifier。explicit fit 的 binding-drop 即使为正，也不能证明 binding 是唯一必要机制，因为更简单的边缘/方向/nearest 规则已经做满同一任务。

所有六类干预的逐样本变换保存在各 task 的 `audit/intervention_samples.npz`；original、pair-preserving、binding-breaking、twin-swap、slot permutation、coordinate reflection 的每个 baseline 逐样本预测保存在 `baseline_predictions.csv`。

## 数据与执行限制

这是小型一维 state-only PhysX MVP，而不是机械臂/RGB benchmark。matching tail 可访问 latent，但仅用于数据构造，完整记录且不进入 verifier probe pairs。公共 current observation 使用冻结的 1e-3 sensor resolution 达到 byte identity，同时原始 PhysX state 仍接受更严格的物理误差 gate。oracle rollout 是上限，不是可部署输入。

## 可复制命令

```bash
HOME=/data/project/tzh/papers/ActMask/outputs/actmask/history_action_binding_mvp/runtime_home PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding.run_mvp --stage smoke
HOME=/data/project/tzh/papers/ActMask/outputs/actmask/history_action_binding_mvp/runtime_home PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding.run_mvp --stage pilot
HOME=/data/project/tzh/papers/ActMask/outputs/actmask/history_action_binding_mvp/runtime_home PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding.run_mvp --stage audit
HOME=/data/project/tzh/papers/ActMask/outputs/actmask/history_action_binding_mvp/runtime_home PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding.run_mvp --stage finalize
PYTHONPATH=/data/project/tzh/papers/ActMask /data/envs/actmask-audit/bin/python -m unittest tests.test_history_action_binding_mvp -v
```
