# ActMask Milestone 2B 计划：Make Dynamics Necessary

## 为什么需要重新设计

Milestone 2A 的负结果必须保留：Full ActMask 的 ID AP 为 `0.5190 ± 0.0462`，而 `TimeAlignedTrajectoryProximity` 为 `0.9475`；动作证据通过，但速度证据失败，Full 的候选动作 Top-1 成功率仅 `0.5000`，因此模拟器集成结论为 NO-GO。

速度门失败有两个直接原因。第一，2A 模型只接收一个时刻的点、单帧观测速度和动作，无法从观测历史区分测量噪声、加速度、遮挡、异步采样和延迟。第二，数据中的不少接触仍可由当前空间几何或恒速外推解释，导致移除或打乱速度没有形成稳定损失。

2A 的 `TimeAlignedTrajectoryProximity` 又非常接近标签生成器：标签由同步采样的点/夹爪未来距离产生，而该基线用公开的单帧速度做恒速同步采样；在大量近恒速、直线动作样本上，两者只有“真实非线性未来”与“恒速外推”的差别，所以它天然接近标签规则。2B 将把精确未来轨迹完全留给 oracle，并让公平方法只能使用带噪、缺失、遮挡和延迟的历史观测。

## 数据契约

每个样本严格拆成三部分：

- `observable`：`points_history [K,N,3]`、`visibility_history [K,N]`、`timestamps [K]`、从可见历史估计的速度与置信度、候选 `action_command`、名义动作/观测时延；learned 模型和公平基线只能接收这些字段。
- `targets`：未来接触掩码、逐时刻接触标签、候选成功/utility、hard-positive/hard-negative 标记；只用于训练损失和事后评估。
- `hidden_state`：精确未来点轨迹、精确速度/加速度、精确夹爪执行轨迹、精确执行时延和接触状态；仅标签生成与 `ExactHiddenTrajectoryOracle` 可用。

公平方法的入口会验证字段白名单；传入任何 hidden-state 字段都必须报错。分组 manifest 在展开运动反事实、动作候选和无关扰动之前建立，`base_scene_id`、`pair_id`、`trajectory_family_id`、`geometry_seed` 不得跨 split。

## 让动态成为必要条件

全部非重复候选的数据按点构造并审计两类难例：当前远离动作扫掠体、未来按时进入的 hard positive，以及当前靠近但移开、错时或已被夹爪错过的 hard negative。两类在各自正/负标签中的比例都必须至少为 40%；候选集合采用 10/4/18/16 个 hard-positive/easy-positive/hard-negative/easy-negative 角色点作为平衡基础。

场景覆盖相反速度、相同当前状态但不同加速度、延迟启动、曲线/圆周、分段运动、多物体交叉、快速无关干扰物、执行延迟、时长不匹配、空间/时间 near-miss 和间歇遮挡。OOD 单独保留速度、加速度、曲率、动作时延、观测时延、遮挡率、噪声、对象数和 hard-case 组合。

## 方法

公平基线全部只读 `observable`：

1. `CurrentPositionProximity`；
2. `EstimatedVelocityFutureProximity`；
3. `EstimatedTimeAlignedTrajectoryProximity`；
4. `ConstantVelocityKalmanProximity`；
5. `ExactHiddenTrajectoryOracle` 读取 hidden state，仅作排除上界。

learned 方法保留单帧拼接 MLP 为 `ActMaskMLP`，并新增 `TemporalActMask`。Temporal 模型编码点历史和可见性，估计当前运动状态，构造未来点假设与名义夹爪轨迹，形成同步相对位置/距离/速度/最近接近时刻特征，再输出点掩码、可选逐时刻接触 logits 和候选 utility logit。

## 训练目标与消融

总损失由掩码 BCE（`1.00`）、逐时刻接触 BCE（`0.20`）、候选成功 BCE（`0.45`）、成功/失败候选的 pairwise ranking loss（`0.30`）、速度与动作反事实正确配对 margin（各 `0.25`），以及 GT 不变时的无关背景稳定损失（`0.10`）组成。只有 GT 确实改变的反事实才要求正确配对，不对不变目标强迫差异。

评估 `ActMaskMLP`、完整 Temporal、无历史/速度、打乱历史、无动作、打乱动作、无反事实损失、无可见性/置信度和无 utility head。除单独训练的 loss/head 消融外，输入消融在 Full checkpoint 上进行，避免把重新优化掩盖成特征证据。

## 不可事后修改的验收门槛

- 动作 removal 与 shuffle 均使 AP 至少下降 `0.05`；
- 历史/速度 removal 与 shuffle 均使 AP 至少下降 `0.05`；
- velocity/action counterfactual matching accuracy 均至少 `0.80`；
- irrelevant perturbation stability 至少 `0.85`；
- Temporal Top-1 动作成功率相对 2A 的 `0.5000` 至少提高 `0.15`；
- Temporal 至少在 noisy、occluded、delayed 或 nonlinear OOD 子集之一超过最强公平 observable 基线；
- oracle 明确排除、五个确定性 CPU seeds、validation-only 选择、全部测试通过。

这些门槛不会在看到结果后降低；最终 `docs/milestone2b_results.md` 和 `outputs/actmask/milestone2b/summary.md` 将按实际产物给出 GO 或 NO-GO。
