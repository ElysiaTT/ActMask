# TimeArrow-v2 可复现性与数据捷径审计

审计 schema：`timearrow-shortcut-audit-v1`。本报告由 `audit/timearrow_shortcut_audit.py` 从冻结文件重新计算；不训练模型，不修改 `outputs/`、论文或已有表格。

## 结论

**TimeArrow-v2 应标记为 `shortcut-saturated`。** 一个不训练、只读取正式 observation/action 的 `ArrowActionCompatibility` 规则，在四个 canonical retained 集合的全部 85,600 行上达到 `1.000000`，并复现 ID test 的 `Top-1=1`、`Top-3=1`、`NDCG=1`、`regret=0`。因此这些整齐数字主要是数据构造、严格 flip 筛选和动作方向编码的代数后果。这个合法输入的 post-hoc audit stress baseline 不是预注册或 validation-selected comparator；它证明的是 benchmark 存在未排除的饱和捷径，因而现有结果不能支持 learned method 存在未解的 headroom 或完成机制识别。

本审计**没有发现足以断言人工伪造数据的直接证据**。四个 canonical 集合的 metadata/label 共 `85,600` 行，按 candidate outcome key 和 branch 直接连接后，success 布尔值 `85,600/85,600` 精确一致；outcome key 和 `(pair_id, branch)` 也都唯一。这里证明的是冻结文件内部自洽，不是其历史生成过程的独立真实性证明。发现的问题是确定性捷径、baseline 坐标遗漏、筛选偏差以及 provenance/重放证据不足。

## 代码事实

- [生成器](../actmask/experiments/milestone3r_nl_v2.py#L57-L68) 把 branch 1 构造成 branch 0 前四帧的严格反序；最后两帧相同。
- [候选动作生成](../actmask/experiments/milestone3r_nl_v2.py#L119-L149) 在 full 当前实现中以 `candidate_id % 2` 交错编码正/负方向。
- [筛选逻辑](../actmask/experiments/milestone3r_nl_v2.py#L198-L211) 先记录所有执行 outcome，再删除两 branch 结果相同的 non-flip pair。
- [observation schema](../actmask/data/maniskill_state_tasks.py#L205-L213) 的 `0:3` 是 payload position，`17:20` 是 goal position。
- 原 analytic [评分函数](../actmask/experiments/milestone3r_nl_v2_probe.py#L28) 只取 estimator 输出的 `[-3:]`，也就是 goal position；两个平移 family 的早期 cue 实际写在 payload position。
- 所谓 `hysteretic_mode_memory` 的 threshold 被存储但没有进入 [response 公式](../actmask/data/maniskill_3r_nl_tasks.py#L95-L111)；实际是带 delay 的 signed quadratic response。

## 数据事实：canonical retained 与 attempted outcomes

| 条件 | attempted branch executions | retained examples | non-flip pairs | retained parity rule | Arrow×Action | min abs(score) |
|---|---:|---:|---:|---:|---:|---:|
| id | 60,000 | 60,000 | 0 | 1.000000 | 1.000000 | 0.018133 |
| physical_parameter_ood | 8,000 | 8,000 | 0 | 1.000000 | 1.000000 | 0.744987 |
| temporal_delay_ood | 12,000 | 5,600 | 3,200 | 1.000000 | 1.000000 | 0.169639 |
| held_mechanism_ood | 12,000 | 12,000 | 0 | 1.000000 | 1.000000 | 0.018133 |

四个集合共尝试 `92,000` 次 branch execution；筛选前 parity 规则命中 `0.965217`，筛选后在 `85,600` retained examples 上恰为 `1.000000`。Temporal OOD 在 6,000 个 attempted pair 中保留 2,800 个（retention `46.667%`）；其余 3,200 个 non-flip pair 全是 `(success, success)`，且没有对应的 model input，因此不能对被删除样本运行公开输入规则。

扫描到的全部 30 个 v2 数据目录（包含 full/raw/prefix/probe/smoke 派生重复）共有 `247,344` 行：Arrow×Action 总准确率 `1.000000`，而 metadata parity 总准确率 `0.980659`。旧 probe/smoke 的 candidate mapping 并非奇偶交错，所以 parity 不是普遍规则；observable Arrow×Action 才是 30/30 目录共同的捷径。

所有 canonical retained pair 都满足：每个 `pair_id` 恰有两行、前四帧严格反序、最后两帧/action/timestamps/visibility/confidence/timing/TCP/static features 完全相同、标签恰好翻转。另将每行 metadata/label 按 `(condition, family, mechanism, world_id, candidate_id)` 连接到唯一 outcome，再取 `branch{branch}_success`，覆盖率和两类 success match rate 均为 `1.000000`。逐目录数据见 [dataset_inventory.json](results/dataset_inventory.json)。

## 零训练规则

```python
arrow = (history[:, 3, 0:3] - history[:, 0, 0:3]
         + history[:, 3, 17:20] - history[:, 0, 17:20])
action_direction = candidate_actions.sum(axis=1)
score = (arrow * action_direction).sum(axis=1)
prediction = score > 0
```

它不使用 `branch`、`candidate_id`、`family`、hidden parameters 或 success。在 retained 集合上，arrow 符号恢复 branch bit，动作净位移符号恢复 direction bit，标签就是二者 XNOR。全量逐样本 score、margin 和预测保存在 [canonical_predictions.csv.gz](results/canonical_predictions.csv.gz)。

### ID test ranking

| Candidate count | contexts | Top-1 | Top-3 | NDCG | regret |
|---|---:|---:|---:|---:|---:|
| C5 | 600 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| C10 | 600 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| C20 | 600 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

## 扰动实验

- **candidate parity remap**：只把 metadata 的 `candidate_id` 改成 `candidate_id ^ 1` 后，parity 公式准确率为 `0.000000`，而 Arrow×Action 仍为 `1.000000`。这说明根本捷径是观察箭头和动作方向，不是模型直接读取 candidate ID。
- **joint candidate record order shuffle**：在每个 history context 内联合重排 row/label/score/action，5 个种子的 C5/C10/C20 指标逐项不变；这是 evaluator 的顺序不变性检查。
- **adjacent action reassignment (`c -> c xor 1`)**：保持旧标签而交换动作后，规则准确率 `0.000000`、pair-order `0.000000`。随机 action-only reassignment 的结果接近 chance。它们只是 shortcut diagnostic；没有 simulator 重新打标签，不能称为物理 counterfactual。
- **exact early partner reverse**：在原标签下准确率 `0.000000`、pair-order `0.000000`，在 partner 标签下准确率 `1.000000`。
- **last-two-only**：arrow 全为零，准确率 `0.500000`、pair-order `0.500000`。
- **genuine random early-frame shuffle**：为每个 pair 采样共同的前四帧 permutation，5 个种子降至 chance 附近；这和现有代码中名字叫 `early_independent_permutation`、实际只是固定 `flip` 的操作不同。

逐样本扰动证据见 [id_test_candidate_perturbations.csv.gz](results/id_test_candidate_perturbations.csv.gz) 和 [id_test_temporal_perturbations.csv.gz](results/id_test_temporal_perturbations.csv.gz)。

## 原 analytic baseline 的坐标遗漏

| family | mechanism | goal-only pair-order | payload+goal pair-order |
|---|---|---:|---:|
| damped_moving_capture | history_identifiable_damping_drive | 0.500000 | 1.000000 |
| damped_moving_capture | hysteretic_mode_memory | 0.500000 | 1.000000 |
| damped_rotating_slot | hysteretic_mode_memory | 1.000000 | 1.000000 |
| hysteretic_moving_container | history_identifiable_damping_drive | 0.500000 | 1.000000 |
| hysteretic_moving_container | hysteretic_mode_memory | 0.500000 | 1.000000 |

goal-only 的 5-cell 均值为 `0.600000`，在同一 estimator 上 post-hoc 补齐 payload+goal 坐标后为 `1.000000`。保存报告基于有坐标遗漏的 goal-only baseline 所做的算术是可复现的：四个平移 cell `0.5`、一个 rotating cell `1.0`、总体 `0.6`，从而得到 learned-minus-analytic `+0.400`；坐标修正的 post-hoc stress baseline 为 `1.000000`。它未经预注册/验证选择，不应冒充原 comparator；但它足以证明原 `+0.400` 不能作为“仍有 learned headroom”的有效证据。

## 为什么数字会恰好是 0.500 / 1.000

- retained label balance 必为 `0.5`：每个 strict flip pair 按定义恰有一个正样本和一个负样本，并同时保存两行。
- static/action/final-two pair-order 必为 `0.5`：pair 两边这些输入严格相同，score tie，而 metric 明确给 tie `0.5`。
- candidate template marginal success 在 strict flip 上必为 `0.5`：当前 diversity audit 计算 `(branch0 + branch1)/2`，却没有检查 `branch × direction` 的 XOR interaction。
- static/action Top-1 在两 branch 平均后为 `0.5`：同一套 candidate scores 对应完全互补的 branch labels。
- 一旦规则把全部成功 candidate 排在失败 candidate 前，`Top-1=1`、`Top-3=1`、`NDCG=1`、`regret=0` 是同一次完美排序的代数派生，不是四项独立证据。

## 推断

1. GRU 很可能学到了低维 arrow-direction XNOR；这些结果不足以支持 learned method 在该 benchmark 上存在机制识别能力或未解的 headroom。这里的 payload+goal 规则是 post-hoc audit stress baseline，不是预注册 comparator。论文把任务定位为 controlled procedural test，这一限定不改变 shortcut 对方法比较效度的影响。
2. Physical/Temporal OOD 仍保留同一 arrow/action 编码，因此 OOD=1 并不证明对隐藏物理参数或 delay 的机制外推。
3. 在继续针对该 benchmark 优化复杂模型前，应先重设计任务：取消 strict-flip-only 选择、随机化 candidate-direction 映射、保存全部执行轨迹，并加入这个规则作为 admission gate。

## 可复现性与环境

独立环境状态：`passed`，import smoke=`True`，GPU smoke=`True`，pip check=`True`，core environment smoke=`True`，legacy exact reproduction=`False`。实际版本：`{"mani_skill": "3.0.1", "numpy": "2.2.6", "opencv_python": "4.10.0.84", "python": "3.11.15", "sapien": "3.0.3", "torch": "2.6.0+cu124", "torch_cuda": "12.4"}`。已知差异：`[{"current": "NVIDIA A16", "kind": "historical_hardware_records_differ_by_stage", "milestone3p": "NVIDIA GeForce RTX 4090 D", "milestone3s": "NVIDIA A16"}, {"current": "580.65.06", "kind": "driver_mismatch_vs_milestone3p", "milestone3p": "580.76.05"}, {"current": "glibc2.35", "historical": "glibc2.39", "kind": "libc_mismatch_vs_historical_platform_records"}, {"current": "isolated conda prefix plus pip packages; only the pre-existing PhysX GPU asset is symlinked under audit HOME", "historical": "milestone3s records repaired cross-environment package symlinks", "kind": "environment_construction_differs"}]`。详细证据见 [environment](environment/)。
最小再执行状态：`completed_with_mismatch`，verification=`False`，schema=`True`，success=`True`，filter=`True`，retained arrays exact=`False`，max abs error=`2.47955322265625e-05`。这是 reduced-batch 下对冻结 seed/world/candidate 子集的 bounded re-execution，不是相同 parent batch 或 simulator snapshot replay。详见 [replay](replay/)。
回放阻碍/错误：Replay mismatch/limitation: the reduced-batch current-runtime subset preserves exact schemas, metadata, outcomes, labels, and filtering, but retained float arrays are not bit-exact; inspect mismatch_details and per_case_array_comparisons.

## 尚未验证 / 不能声称

- `candidate_outcomes.jsonl` 只保存最终布尔值，没有逐步 distance/success、最终 simulator state 或 discarded-pair model inputs；冻结文件内部一致性不能独立证明这些布尔值确由历史 GPU PhysX 运行产生。
- 项目没有 Git 元数据和完整旧环境 lock；只能重建核心版本，不能声称 bit-exact legacy environment。
- 当前 generator 的交错 candidate mapping 与同一 config hash 下旧 probe 的前五正/后五负 mapping 不一致，而 generation report 没有 source hash。这是 provenance gap，不是造假证明。
- action-only shuffle 未经 simulator 重新打标签，只能诊断 shortcut，不能用来估计真实 counterfactual performance。

## 复现命令与产物

```bash
/root/miniconda3/bin/python audit/timearrow_shortcut_audit.py
/root/miniconda3/bin/python -m unittest discover -s audit/tests -v
```

- [summary.json](results/summary.json)：全部指标与结论
- [dataset_inventory.json](results/dataset_inventory.json)：所有发现的数据目录
- [source_manifest.json](results/source_manifest.json)：冻结输入与关键源码 SHA-256
- [canonical_predictions.csv.gz](results/canonical_predictions.csv.gz)：85,600 行逐样本规则证据
- [id_test_candidate_perturbations.csv.gz](results/id_test_candidate_perturbations.csv.gz)：candidate 扰动逐样本证据
- [id_test_temporal_perturbations.csv.gz](results/id_test_temporal_perturbations.csv.gz)：temporal 扰动逐样本证据
