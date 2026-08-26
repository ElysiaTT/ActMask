# History–Action Binding Audit v2.1：预注册

状态：**仅完成开发设计；official symbolic preflight 尚未运行。** 本版本独立于 v1、
TimeArrow 和失败的 v2，且不训练 learned verifier。所有 v2.1 scientific core、阈值和本文件
在 official seed 产生前冻结；official 结果只允许生成一次。

## 1. v2 失败驱动的唯一修复

v2 Task A 的 0.87890625 不是广泛的错误：124 个 miss 全部来自 templates 2/6/10/14，
其理论 twin preference 为零，而 float32 约 `1e-10` 的残差超过了冻结 metric 的 `1e-12`
informativeness epsilon。原 Task A 又只有一个 probe radius，使包含 `r` 和 `r²` 的 Fourier
矩阵秩亏。v2 slot sampler 的 multiplier 在不对齐的 `seed//16` 边界改变，所以虽然 shift
总体出现四次，multiplier×shift 没有形成平衡乘积，最终出现 cell=3/5。

v2.1 只做以下修复，不改 metric 或阈值：

- Task A probe magnitudes 从 `[0.80]` 改为 `[0.40, 0.80]`；
- Task A candidate directions 改为公开坐标旋转后的
  `[10°,20°,30°,40°,50°,60°,70°,80°]`；
- 两个任务的 slot schedule 改为严格 cyclic Latin square；
- Task B 的方程、probe、candidate directions、连续参数分布和阈值保持 v2 不变。

Task A candidates 的 magnitudes 仍为 `[0.35,0.65]`，与 probe magnitudes 不重合；candidate
angles 也不与 45° probe grid 重合。candidate sampler 只能读取 task、公开 rotation 和 seed。

## 2. 数据生成方程

二维 command 驱动一维 GPU-PhysX rail。每个 world 有公开坐标旋转 `rho`，twin 的隐藏
phase 相差45°，branch assignment 精确平衡并随机交换。

Task A：

`F = 4 ||u|| cos(2(angle(u)-phi))`

Task B（保持 v2）：

`F = 4 [alpha ||u|| - beta ||u||²] cos(2(angle(u)-phi))`

其中 `alpha∈[1.05,1.25]`、`beta∈[0.18,0.32]`、phase jitter `∈[0°,45°]`；
Task B probe magnitudes 为 `[0.20,0.50,0.80]`，candidate directions 是 probe grid 加15°。

历史必须由连续 env steps 产生。只允许 episode reset、history 结束后的 decision-state snapshot
以及每条 candidate 前从该 snapshot 恢复；禁止逐帧 `set_pose` 构造 history。matching controller
可以访问 latent，但其命令与完整 trace 必须保存且不能进入 fair verifier history。

## 3. 严格 slot 证明

每个 twin 的排列定义为：

`template(slot, seed) = (slot + seed mod 16) mod 16`。

对任意连续64个 candidate seeds，`seed mod 16` 每个 residue 恰好出现4次。固定任意 slot，
加法 mod16 是 template 上的双射，因此16×16 slot/template table 每格恰好为4。这个性质由
公式保证并由单元测试穷举验证，不依赖 official outcome。

## 4. 开发与 official 隔离

开发只使用 `common.py::PROTOCOL[development_seeds]` 明列的20个 seeds。开发输出不得作为
official 结果。冻结步骤记录：

- preregistration SHA-256；
- protocol SHA-256；
- scientific core 及 protected v2 metric dependency 的逐文件 SHA-256；
- v2 diagnosis 和 development validation SHA-256；
- protected project/environment manifests。

official seed 从这些冻结字段的 canonical SHA-256 前16个十六进制字符确定：

`int(first_16_hex,16) % 2000000000 + 1`。

`official_run_started.json` 使用 exclusive-create；一旦存在，official symbolic 不得重跑。
official 后不得修改 task equation、sampler、metric、threshold 或 scientific core。

## 5. Official symbolic preflight

每任务64 twins、16 candidates，两个任务必须同时满足：

构造完整性：

- twin action marginal exact；
- result multiset exact；
- probe utility multiset exact；
- current public state exact；
- candidate trajectories branch-exact；
- 16×16 slot/template table 每格严格等于4；
- top-1 crossing rate ≥0.90；
- sampler signature 仅为 `(task, public_rotation, candidate_seed)` 且不读取 oracle字段。

Shortcut gates：

- current/action/result/index corrected TPA 均≤0.60；
- Arrow≤0.70；
- nearest 和 kNN 均严格<0.90。

Signal gates：

- oracle 和 Fourier 均以 twin bootstrap 显著超过最强 marginal baseline；
- Fourier corrected candidate-conditioned TPA≥0.90；
- binding-breaking shuffle 至少下降0.20；
- pair-preserving shuffle 变化≤0.02；
- reverse slot permutation 与37° coordinate rotation 不改变相对0.90阈值的通过/失败结论。

保存每个 task 的 `symbolic_samples.npz`、12个 baseline 的逐样本 prediction CSV、
全部 intervention samples 和 preflight summary。

任一检查失败，立即输出 `SYMBOLIC_PREFLIGHT_FAIL`：不得创建 GPU env，不得创建 smoke/pilot
目录，candidate executions=0，不得改阈值、seed、task 或 sampler，不训练模型。

## 6. 条件式 GPU smoke 与 formal pilot

仅当两个 symbolic task 全通过：

1. 每任务8 twins smoke，只检查 execution/data integrity；
2. 两个 smoke 均通过后，每任务64 twins formal pilot；
3. 不按 outcome 重采样、过滤或删除，保存全部00/01/10/11。

必须保存 public/physical history、probe commands、matching tail、candidate trajectories、
response/TCP proxy/target、每步 distance/contact、首次成功时间、initial/decision/final state hash、
seed、split、oracle latent、环境版本和可复制命令。trace 必须可以独立重算 utility 和 success。
正式 GPU stages 共 `4608` candidate executions，低于5000；全部输出必须≤1GB。

## 7. Baselines、干预和正式 gate

必报12项：class prior、current-only、candidate-index、action-only、result-only、Arrow、
repeat-last-failed、nearest、kNN、linear SysID、quadratic/Fourier SysID、oracle rollout。

必报干预：pair-preserving shuffle、binding-breaking shuffle、twin-history swap、slot
permutation、coordinate rotation、novel-candidate evaluation。保存所有逐样本预测。

只有两个 formal GPU tasks 都通过 data integrity、marginal/interaction shortcut、binding drop、
novel-candidate SysID 和 invariance gates，才允许输出
`LEARNED_TRAINING_AUTHORIZED=true`。即使授权，本 Goal 也不训练任何 learned model。

## 8. 不可变限制

- 不降低 symbolic Fourier 0.90 阈值；
- 不修改 protected v2 corrected metric；
- 不筛除困难 twins、candidates 或 concordant outcomes；
- 不根据 official 结果重选 seed；
- 不覆盖 v1、v2、TimeArrow、audit、paper 或现有环境；
- 失败是允许的终止结果，但必须完整、可复算并 fail-closed。
