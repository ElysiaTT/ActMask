# 可直接交给 Codex 的 CPU-first Goal

## Goal

在 ActMask 中把论文主方向从 generic arrow-of-time 调整为
**Counterfactual Action-Effect Binding Audit for History-Aware Robot
Verifiers**。全程禁止使用 GPU。先在 CPU 上用预注册、matched twins 和因果干预
验证：当 current state、candidate set、candidate slot、past-action multiset、
past-result multiset 与通用顺序统计均匹配时，模型的候选动作排序是否真正依赖
“哪个历史动作产生了哪个观测结果”。

## 已有证据（不得覆盖或挑选性删除）

- TimeArrow-v3 为 NO-GO：unordered history 几乎保留全部性能，不能作为时序推理证据。
- BindingCheck-CPU-v1 已通过 symbolic direction admission。
- BindingCheck learned CPU v1 已在 5 个种子的 ID、parameter OOD、noise OOD、
  held-mechanism 上通过，bound TPA 为 0.9875--0.9966；unbound 与边缘控制均为
  0.5000。
- learned spectral Ridge 与 analytic witness 基本持平，所以当前贡献定位必须是
  benchmark/audit，而不是宣称 learned method SOTA。

## 硬约束

1. 不运行 CUDA、GPU simulator 或任何隐式 GPU 后端；先检查依赖和设备路径。
2. 每轮在看结果前冻结 generator、split、seed、baseline、intervention、metric 和
   GO/NO-GO gate；失败结果必须原样保留。
3. 禁止 test-set 调参、换 seed、删除困难样本、只报均值或把开发结果称为 confirmatory。
4. 强制报告 current/action/result/candidate-ID/slot、arrow/endpoint、nearest/kNN、
   analytic sysid 与同容量 unbound-history 控制。
5. 同时做 pair-preserving shuffle 和 binding-breaking shuffle；前者应不变，后者应
   显著下降。只做普通 shuffle 不足以证明 binding。
6. 主指标使用 candidate-conditioned twin preference accuracy 和 normalized regret；
   同时完整报告 Top-1，不能用 TPA 隐藏 exact argmax 较低的问题。
7. CPU pass 只允许建议下一个独立实验，不自动授权 GPU。

## 下一阶段 CPU 工作

在不使用已知正确 Fourier basis 的前提下，独立预注册并比较 raw paired-token
Ridge/MLP、DeepSets、轻量 attention 和 unbound 等容量控制。加入 history-length、
probe dropout、非均匀动作覆盖、异方差噪声、未知频率/非加性 held mechanisms。
优先用现有或公开的预计算 state/flow/action features 做外部 verifier 审计，避免为了
第一轮验证训练视觉 backbone。

## 完成条件

输出可复现代码、原始逐 seed/split JSON、审计报告和失败记录。只有当 raw-token
模型在至少 5 个种子的 ID、parameter OOD、noise OOD、真正结构外 held mechanism
上稳定超过全部边缘/unbound 控制，binding break 显著下降、pair-preserving 基本
不变且 regret gate 通过，才把方法方向提升为候选；否则保持 benchmark/audit 定位
或据证据转向。整个 Goal 结束时仍不得启动 GPU。
