# History–Action Binding Audit MVP：预注册协议

状态：**工程预注册，正式 pilot 尚未运行**。本文件与 `common.py` 的协议常量在
8-twin-per-task smoke 只验证运行稳定性和完整性后冻结；smoke 不用于修改结果分布、
候选集合或 gate 阈值。

## 科学问题

给定历史 action–result 对、匹配的当前可观测状态，以及 twin 间逐字节相同的候选
action 集，verifier 是否依赖“哪个 action 导致哪个 result”的绑定关系，而不是
current-only、action-only、candidate slot、历史边缘统计或一比特方向捷径？

本 MVP 只做数据准入和无训练审计。它不训练 GRU、TCN、Transformer 或其他 learned
verifier，也不修改 TimeArrow-v2、`audit/` 或论文目录。

## 两个 state-only GPU-PhysX 任务

动态 response 刚体被约束在一维轨道上。每个 control step 的命令通过外力作用于刚体，
位置和速度由 PhysX 连续积分。response pose 只在 episode 初始化或候选分支状态恢复时
写入；历史帧不使用逐帧 `set_pose`、early reversal 或 observation array editing。

- Task A（离散隐藏模式）：`F=4 z u`, `z∈{-1,+1}`。四个 signed probe 暴露
  push/pull 响应方向。
- Task B（连续隐藏响应）：`F=4(αu-βu|u|)`；每个 twin 的 `(α,β)` 不同且只保存为
  oracle metadata。probe 绝对幅值 `{0.2,0.5,0.8}`，候选绝对幅值来自
  `{0.15,0.35,0.65,0.95}`，两集合严格不相交。

probe 后使用实际执行的闭环 matching tail 把 response 带回公共 decision state。控制器
可访问 latent，只用于数据构造并完整记录，不作为 verifier 历史输入。禁止直接复制一个
twin 的 decision state 给另一个 twin。候选执行只在两个状态都完成匹配后各自 clone。

## 数据规模、独立性和保存内容

smoke 每任务 8 twins；通过完整性检查后，正式 pilot 每任务 64 twins。每 twin 8 个候选，
两个分支全部执行，累计 smoke+pilot 候选执行数为 2304，小于 5000；结果目录硬上限 1GB。
不按 outcome 重采样、不过滤 00/01/10/11，不改变任务或阈值来追求漂亮分布。

每个 twin 保存：seed/split、环境/协议/源码 hash、oracle latent、完整 probe transition、
matching tail、matched current state、candidate action/hash、每步 response/actuator/goal 状态、
distance、contact、first-success、final error、min distance、continuous utility、binary success、
initial/decision/final state hash。binary label 必须能从 trace 100% 重算。

统计独立单位是 twin/world，不把一个 twin 的多 candidate/step 当成独立样本。

## 固定干预

1. Pair-preserving shuffle：整体重排 `(action,result)` 对，只破坏时间顺序。
2. Binding-breaking shuffle：固定 actions 和 result multiset，只置换 action–result 对应。
3. Twin-history swap：当前状态和候选不变，只交换两个 twin history，并保存 simulator
   期望的 preference 方向。
4. Novel-candidate：历史 probe 与候选幅值严格不重合。
5. Candidate-slot permutation：candidate/label/utility/trace 联合重排，排名指标必须不变。
6. Coordinate reflection：history/current/candidate/goal 一致反射，物理距离和排名应不变。

## 必报无训练 baseline

class prior、current-only、candidate-index、action-only、result-only、
ArrowActionCompatibility、repeat-last-failed-action、nearest historical action、kNN lookup、
linear least-squares response、quadratic least-squares response、oracle latent rollout。所有
baseline 保存逐样本分数，不能选择性省略。

指标包括 AUROC、AUPRC、balanced accuracy、Brier、ECE、utility MAE、Spearman、top-1
utility、NDCG、normalized regret。twin 级报告 00/01/10/11、flip ratio、平均
`|U_A-U_B|`、Twin Preference Accuracy、matching error 和 current-only branch predictability。

## 冻结阈值和终止 gate

- `DATA_INTEGRITY_FAIL`：任一 trace label 不能重算、twin candidate 非逐字节相同、candidate
  sampler 接触 latent/twin/future/label、pair current error > `1.5e-3`、单状态 match 超出
  position `5e-4` 或 velocity `1.5e-3`、split/seed 泄漏、预算超限。
- `SHORTCUT_SATURATED`：任一预注册 simple shortcut 在 twin preference 或 NDCG 达到
  `0.90`。这是停止训练的结论，不是失败后调数据的理由。
- `NO_IDENTIFIABLE_SIGNAL`：oracle 和 explicit linear/quadratic fit 均不能以 twin-cluster
  bootstrap 显著优于最强 current/action marginal baseline。
- `BINDING_AUDIT_PASS`：完整性通过；无 marginal/shortcut saturation；正确 binding 相对
  binding-break 的 twin preference 至少下降 `0.10`；pair-preserving shuffle 变化不超过
  `0.02`；explicit fit 在 novel candidates 的 twin accuracy 至少 `0.70`；slot permutation
  与 coordinate reflection 保持排名不变。
- 只有 `BINDING_AUDIT_PASS` 才输出 `LEARNED_TRAINING_AUTHORIZED=true`。其他任何结论均不
  训练 learned verifier。

固定脉冲 profile、完整数值阈值、seed 和 split 见
`actmask/experiments/history_action_binding/common.py::PROTOCOL`；冻结结果目录会保存其
canonical JSON、SHA-256、源码清单与生成证据清单。
twin 间还要求以固定 `1e-3` state-sensor resolution 得到逐字节相同的公共 decision
observation；原始 PhysX state 另存并继续接受更严格的 position/velocity match gate，不能用
量化掩盖物理 matching 失败。
