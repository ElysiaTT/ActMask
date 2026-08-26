# ActMask 第二阶段 CPU-only 评估

## 结论

**NO-GO：当前证据不支持进入模拟器集成。**

最终五种子结果表明，Full ActMask 使用了动作输入，但没有显示出可靠的速度或时间依赖：移除/打乱动作分别使 AP 降低 `0.0745`/`0.0684`，而移除/打乱速度反而使 AP 提高 `0.0343`/`0.0125`。同时，Full 的 ID AP/IoU 为 `0.5190 ± 0.0462` / `0.4029 ± 0.0405`，显著低于最强公平几何基线 `TimeAlignedTrajectoryProximity` 的 `0.9475` / `0.7973`。

候选动作排序也同向落后：Full 的 Top-1 成功动作准确率为 `0.5000 ± 0.0791`，而公平基线为 `0.8000`；Full 的 oracle regret 为 `0.4979 ± 0.0782`，公平基线为 `0.2000`。因此，较好的掩码与较好的排序在这些方法之间共同出现，但这不是“ActMask 改善动作选择”的证据，反而说明当前 learned 模型尚未达到合理几何基线。

本阶段严格保持 CPU-only：未接入模拟器、VLA、RGB-D、预训练检查点、GPU 或 CUDA 扩展。

## 运行范围与可追溯性

- 命令：`python -m actmask.experiments.milestone2 --config configs/actmask/milestone2_cpu.yaml`
- 输出目录：`outputs/actmask/milestone2_evaluation/`
- 最终状态：`run_manifest.json` 为 `complete`；CPU 运行时间 `78.114` 秒。
- 种子：`[1301, 2303, 3307, 4327, 5347]`，没有最佳种子筛选。
- 环境断言：`effective_device=cpu`，`cuda_available=false`，`cuda_used=false`。
- 最终配置 digest：`e354b60c7ef06e72d4e66d9ecfbb815b277e19d5650fca02a181b041dedeebf0`；split digest：`7dc0a2b230052d6dd8d622dcc063cac35edaf1bb3e36bb862e60e1f5fe7ecbd5`。

ID 分组数/样本数为 train `40/80`、validation `16/32`、test `20/40`；每个 OOD 轴为 `10/20`。`split_manifest.json` 的自动泄漏检查覆盖 `group_id`、`base_scene_id`、`pair_id`、`geometry_seed`、`scenario_seed` 和 `trajectory_family_id`，全部交集为空。每个 OOD 清单也明确记录了对应的 `ood_*` 域，而非误记为 ID。

所有阈值只在 ID validation 上以 F1 选择，再冻结用于 ID test、情景、反事实和 OOD。Full 的五个冻结阈值依次为 `0.642951`、`0.792954`、`0.572794`、`0.872675`、`0.824054`；完整记录见 `validation_thresholds.json`。

## 数据与标签审计

标签由每个点的真实未来轨迹与候选夹爪轨迹的同步接触计算，不从模型输入复制标签。数据包含恒定加速度、曲线轨迹、延迟运动开始、多移动物体、快速移动干扰物、静态近接非接触、near-miss 动作、不同动作时长和夹爪半径、位置噪声、速度噪声和 5% 的固定形状安全背景点 dropout。

`temporal_velocity_delay=0.04` 是独立的观测时延：模型接收的速度为一阶后向传播后的陈旧速度再加观测噪声；物理未来轨迹、动作和接触标签不变。该实现与物理上的 delayed-onset 场景不同。ID/OOD 支持严格分离了速度幅值、加速度、动作时长、夹爪半径、轨迹曲率、点噪声和干扰物数量七个轴。

## 方法与信息预算

| 方法 | 可用信息 | 定位 |
| --- | --- | --- |
| `CurrentPositionProximity` | 当前点、动作线段、夹爪半径；无速度、无未来轨迹 | 公平当前位置几何基线 |
| `FuturePositionProximity` | 当前点、观测速度、动作时长/终点/半径；恒速终点近似 | 公平预测终点几何基线 |
| `TimeAlignedTrajectoryProximity` | 当前点、观测速度、完整候选动作；恒速同步轨迹近似 | 最强公平几何基线 |
| `GeometricOracle` | 合成数据保存的精确未来轨迹/最小接触距离 | 仅上界，排除在公平比较之外 |
| `Full`、`NoVelocity`、`NoAction`、`PositionOnly` | 从头训练的受控 learned feature ablation | learned 比较 |
| `ShuffledVelocity`、`ShuffledAction`、`TimeShuffled` | Full checkpoint 的全局 derangement 输入干预 | 评估时特征使用诊断 |

## ID 掩码结果

下表为五种子均值 ± 样本标准差；Full 括号内为最小--最大值。完整的 mean/std/min/max、情景行和 PR 曲线见 `multi_seed_summary.*`、`scenario_metrics.*` 和 `pr_curves/`。

| 方法 | AP / PR-AUC | IoU | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full ActMask | `0.5190 ± 0.0462` (`0.4889–0.5988`) / `0.5178 ± 0.0465` | `0.4029 ± 0.0405` | `0.4710 ± 0.0299` | `0.7546 ± 0.1382` | `0.5734 ± 0.0419` |
| `CurrentPositionProximity` | `0.7320 / 0.7319` | `0.4965` | `0.7378` | `0.6029` | `0.6636` |
| `FuturePositionProximity` | `0.8569 / 0.8568` | `0.6517` | `0.9552` | `0.6723` | `0.7891` |
| `TimeAlignedTrajectoryProximity`（公平） | `0.9475 / 0.9475` | `0.7973` | `0.9170` | `0.8592` | `0.8872` |
| `GeometricOracle`（上界，排除） | `1.0000 / 1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |

每个种子的最强公平方法都在 **ID-validation AP** 上预先选为 `TimeAlignedTrajectoryProximity`，从未使用测试结果。Full 平均 AP 比它低 `0.4285`，平均 IoU 低 `0.3943`；Full 的最佳 AP `0.5988` 仍低于该基线的 `0.9475`。

## 特征使用消融

`ΔAP = Full − 条件`；正数表示移除或打乱该输入会损失 AP。

| 条件 | ID AP（均值 ± 标准差） | ID IoU（均值 ± 标准差） | ΔAP |
| --- | ---: | ---: | ---: |
| Full | `0.5190 ± 0.0462` | `0.4029 ± 0.0405` | `0.0000` |
| NoVelocity | `0.5534 ± 0.0456` | `0.3941 ± 0.0255` | `-0.0343` |
| NoAction | `0.4445 ± 0.0146` | `0.4001 ± 0.0406` | `+0.0745` |
| PositionOnly | `0.4518 ± 0.0201` | `0.4112 ± 0.0082` | `+0.0673` |
| ShuffledVelocity | `0.5315 ± 0.0497` | `0.3786 ± 0.0696` | `-0.0125` |
| ShuffledAction | `0.4507 ± 0.0517` | `0.3325 ± 0.0601` | `+0.0684` |
| TimeShuffled | `0.5166 ± 0.0455` | `0.4034 ± 0.0411` | `+0.0024` |

动作证据通过：NoAction 和 ShuffledAction 都损失约 `0.07` AP。速度证据失败：NoVelocity 和 ShuffledVelocity 的 AP 均高于 Full；TimeShuffled 的变化仅 `0.0024` AP。因此不能声称 Full 已可靠地同时利用速度和时间。

## 直接反事实与交换配对

对每个严格二元对，评估预测与自身 GT 的 Bernoulli 对数似然，并与交换后的 GT 指派比较；`matching_accuracy` 的精确平局计 `0.5`，正 `matching_margin` 支持正确配对。下表只取直接要求的 ID 场景，而不是把同一 counterfactual type 的其他场景混入：速度场景保持位置/动作，动作场景保持位置/速度，时间场景保持场景和空间动作。数值为 Full 的五种子统计。

| 直接干预 | 配对数/种子 | Matching accuracy | Matching margin | 不变点概率稳定性 |
| --- | ---: | ---: | ---: | ---: |
| `velocity_counterfactual` | 2 | `0.4000 ± 0.2236` [`0.0000–0.5000`] | `+0.0046 ± 0.0058` [`-0.0026–0.0102`] | `0.999647 ± 0.000401` |
| `action_counterfactual` | 2 | `1.0000 ± 0.0000` [`1.0000–1.0000`] | `+0.0724 ± 0.0391` [`0.0115–0.1108`] | `0.973895 ± 0.009590` |
| `timing_counterfactual` | 2 | `0.7000 ± 0.4472` [`0.0000–1.0000`] | `-0.0002 ± 0.0056` [`-0.0096–0.0046`] | `0.995832 ± 0.002853` |
| `irrelevant_background_change` | 20 | `0.5000 ± 0.0000` | `0.0000 ± 0.0000` | `0.999988 ± 0.000004` |

无关背景的 GT 不变，交换匹配为不可辨识平局 `0.5` 是预期行为，稳定性才是关键。动作干预匹配很强；但直接速度配对低于随机平局方向、时间配对方差很大且 margin 接近零。这与消融结论一致：当前 Full 的速度/时序条件化证据不足。

## ID/OOD 分离结果

`ID−OOD AP gap` 为同种子 ID AP 减该轴 OOD AP；正数表示该 OOD 切片更低。负数不等于泛化成功，只说明该小切片的绝对 AP 恰高于 ID 均值。所有阈值均冻结自 ID validation。

| OOD 轴 | Full OOD AP（均值 ± 标准差 [min–max]） | ID−OOD AP gap（均值 ± 标准差 [min–max]） |
| --- | ---: | ---: |
| `unseen_velocity_magnitude` | `0.4686 ± 0.0606` [`0.4003–0.5362`] | `+0.0504 ± 0.0473` [`-0.0281–0.0962`] |
| `unseen_acceleration` | `0.5029 ± 0.0775` [`0.3930–0.5965`] | `+0.0161 ± 0.0778` [`-0.0771–0.0959`] |
| `unseen_action_duration` | `0.5132 ± 0.0391` [`0.4564–0.5636`] | `+0.0058 ± 0.0285` [`-0.0332–0.0352`] |
| `unseen_gripper_radius` | `0.5657 ± 0.0482` [`0.5150–0.6457`] | `-0.0467 ± 0.0196` [`-0.0725–-0.0234`] |
| `unseen_trajectory_curvature` | `0.5323 ± 0.0331` [`0.4994–0.5727`] | `-0.0132 ± 0.0561` [`-0.0652–0.0778`] |
| `increased_point_noise` | `0.5592 ± 0.0865` [`0.4576–0.6264`] | `-0.0401 ± 0.0696` [`-0.1185–0.0340`] |
| `increased_distractor_count` | `0.4977 ± 0.0510` [`0.4226–0.5569`] | `+0.0213 ± 0.0512` [`-0.0375–0.0796`] |

最弱绝对 OOD AP 为未见速度幅值的 `0.4686`。OOD 覆盖完整，但这些数值不能支持可部署的泛化结论。

## 多候选动作排序

ID 测试的 20 个场景各有五个候选（successful、failed、wrong-timing、wrong-direction、near-miss），共 100 个候选。非 oracle 方法的动作分数是候选动作下的平均预测点掩码概率；`GeometricOracle` 使用精确候选 utility，仅作排除在公平比较之外的上界。

为避免固定候选 ID 泄漏，绝对分差不超过 `1e-6` 的分数视为同一并列块：Top-k 和 regret 为块内均匀期望，pairwise/AUC 并列计 `0.5`，`candidate_id` 只用于显示溯源。政策和容差记录在 `action_ranking_metrics.json` 的 `tie_policy=uniform_expected_within_score_tolerance` 与 `score_tie_tolerance=1e-6`。

| 指标 | Full ActMask | `TimeAlignedTrajectoryProximity`（公平） | `GeometricOracle`（上界） |
| --- | ---: | ---: | ---: |
| Top-1 successful-action accuracy | `0.5000 ± 0.0791` [`0.4000–0.6000`] | `0.8000` | `1.0000` |
| Top-3 success recall | `0.8500 ± 0.0729` [`0.7750–0.9500`] | `1.0000` | `1.0000` |
| Top-3 success hit | `0.9200 ± 0.0908` | `1.0000` | `1.0000` |
| Pairwise ranking accuracy | `0.7142 ± 0.0280` | `0.9375` | `1.0000` |
| Ranking AUC | `0.7142 ± 0.0280` | `0.9375` | `1.0000` |
| Global binary AUC | `0.7038 ± 0.0453` | `0.9448` | `1.0000` |
| Oracle regret | `0.4979 ± 0.0782` [`0.4000–0.5974`] | `0.2000` | `0.0000` |

Full 比公平基线低 `0.3000` Top-1、并高 `0.2979` regret。掩码和排序的方向一致，但当前证据并不支持 learned ActMask 带来选择改进。

## 失败案例

每个种子在 `runs/seed_<seed>/failure_cases/` 生成六张仅基于真实预测的三维 PNG，共 30 张；每份 `selection_manifest.json` 都记录选择理由和图路径。最终五个种子中，以下六类均为 `qualifying: true`：

- `actmask_beats_geometric`
- `geometric_beats_actmask`
- `velocity_counterfactual_pair`
- `action_counterfactual_pair`
- `ood_failure`
- `action_ranking_failure`

反事实面板只接受来源数据的完整 variant `0/1` 二元对，候选排序记录不会因保留场景名或 pair ID 而伪装成物理反事实。排序失败图只在最高分并列块的所有候选都失败且存在更高 utility oracle 时才标记为严格失败；没有案例时会显式标为 `qualifying: false`，不会伪造图。

## 对研究问题的直接回答

1. **ActMask 是否同时使用动作和速度？** 动作使用有证据；速度使用没有，时间条件化也没有可靠证据。
2. **它是否击败最强合理公平几何基线？** 否。Full 的 ID AP/IoU 为 `0.5190/0.4029`，公平基线为 `0.9475/0.7973`。
3. **更好的掩码是否改善候选动作选择？** 方法间的两类指标同向，但 Full 两者都更差；不能声称当前 ActMask 改善了选择。
4. **现有证据是否足以进入模拟器？** 否。下一阶段前至少应修复速度/时间依赖、提高严格反事实一致性、稳定地超过公平几何基线，并降低 Top-1 排序 regret。

## 主要产物

- `run_manifest.json`：CPU 环境、种子、完整性断言、运行时间。
- `split_manifest.json`：分组分割、OOD 域、数据统计和无泄漏检查。
- `scenario_metrics.*` 与 `multi_seed_summary.*`：ID/OOD/情景掩码指标。
- `validation_thresholds.json`：validation-only 阈值选择记录。
- `counterfactual_metrics.*` 与 `counterfactual_multi_seed_summary.*`：正确配对 vs. 交换配对的原始和五种子统计。
- `action_ranking_metrics.*` 与 `action_ranking_multi_seed_summary.*`：候选动作排序与无偏并列政策。
- `id_ood_generalization_gap*`：逐轴 ID/OOD 差距。
- `runs/seed_*/failure_cases/`：六类可追溯失败图及选择清单。

这些结论只针对当前的合成、CPU-only 基准，不能外推到真实机器人、真实视觉或仿真环境。
