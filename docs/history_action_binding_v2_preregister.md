# History–Action Binding Audit v2：预注册

状态：**正式 pilot 未运行**。v2 是独立版本；v1、TimeArrow、`audit/`、论文和已有环境均为
只读保护对象。本 Goal 不训练 learned verifier。

## v1 驱动的修复目标

v1 的 Task A 把 latent 写进一阶 action–result moment，Task B 又保留了 branch 总体容易度
和共同 candidate 排序。v2 因此要求 twin 的 action marginal 相同、result multiset 相同、
current observation 相同、candidate set 相同；只有 action–result correspondence 不同。

旧 Twin Preference Accuracy 会被 branch 级常数预测利用。v2 的 terminal metric 先对每个
branch 的 candidate utility 去均值，再对每个 candidate 比较 twin 差异。常数 branch score
严格只能得到 0.5。

## 数据生成方程

action 为二维向量，response 刚体仍在一维 GPU-PhysX rail 上运动。每个 world 有公开随机
坐标旋转 `rho`，branch assignment 精确平衡且随机交换。

Task A：

`F = 4 ||u|| cos(2(angle(u)-phi))`

Task B：

`F = 4 [alpha ||u|| - beta ||u||^2] cos(2(angle(u)-phi))`

Task B 的 alpha∈[1.05,1.25]、beta∈[0.18,0.32]、hidden phase jitter∈[0°,45°]
连续采样；twin 共享 alpha/beta，phase 相差45°。probe directions 为公开 frame 下每45°一条，
所以 phase shift 只循环置换 response，result multiset 不变。Task B probe magnitudes 为
0.20/0.50/0.80；candidate magnitudes 为0.35/0.65，严格不重合；candidate angles 为 probe
grid 加15°。每个 twin 16 candidates，slot 使用 seed 控制的平衡置换，sampler 不接触 latent。

历史和 matching tail 都由连续 PhysX rollout 产生。只有 episode 初始化和 candidate state
branching 可恢复 state；不编辑 observation、不反转帧、不逐帧 set_pose。matching controller
可访问 latent，但完整保存且不进入 verifier history。

## Symbolic preflight

GPU smoke 前必须在同一固定方程上验证：action/result marginals、slot balance、top-1 crossing；
current/action/result/index corrected TPA≤0.60；Arrow≤0.70；Fourier SysID≥0.90；binding break
至少下降0.20；pair-preserving变化≤0.02。失败即停止，不运行 GPU pilot。

## 冻结执行

symbolic 通过后每任务8-twin smoke，只检查 integrity；随后冻结 config、阈值和全部 source
hash。正式 pilot 每任务64 twins，不按 outcome 重采样或过滤，保存全部00/01/10/11。
smoke、pilot和1-twin双复跑合计候选执行4672≤5000，输出≤1GB。

公共 probe/current 使用冻结的1e-3 sensor resolution以定义 verifier 输入；同时保存原始
PhysX state，并分别执行 position≤5e-4、velocity≤1.5e-3、twin physical current≤1.5e-3
和 probe-result physical multiset≤1.5e-3 gate。量化不能掩盖物理 matching 失败。

## Baseline 和指标

必报 class prior、current-only、candidate-index、action-only、result-only、Arrow、
repeat-last-failed、nearest history、kNN、linear SysID、quadratic/Fourier SysID、oracle。
保存全部逐样本预测。

除 AUROC/AUPRC/Brier/ECE/Spearman/NDCG/regret 外，必报 within-twin pairwise ranking、
centered utility、candidate-conditioned twin preference、centered binding score、逐 candidate
twin preference。统计独立单位是 twin/world。

## Terminal gate

- `DATA_INTEGRITY_FAIL`：trace、candidate equality、matching、marginal equality、sampler、split
  或 hash 任一失败。
- `MARGINAL_SHORTCUT_FAIL`：current/action/result/index 任一 corrected binding TPA≥0.70。
- `INTERACTION_SHORTCUT_FAIL`：Arrow/nearest/kNN 任一 corrected binding TPA≥0.90。
- `NO_IDENTIFIABLE_SIGNAL`：oracle 或 Fourier SysID 不能以 twin bootstrap 显著超过最强
  marginal baseline。
- `BINDING_AUDIT_PASS`：完整性通过、上述 shortcut 均不过线、Fourier novel TPA≥0.85、
  binding-break下降≥0.20、pair shuffle变化≤0.02、slot/coordinate invariance误差≤1e-12。

只有两个任务都为 `BINDING_AUDIT_PASS` 才输出 `LEARNED_TRAINING_AUTHORIZED=true`。
完整常量和 seed 以 `common.py::PROTOCOL` 为准。
