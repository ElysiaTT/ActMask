# BindingCheck learned CPU v1 最终报告

> **Post-result correction:** the 0.5-versus-near-1 bimodality was later shown
> to be structurally forced by identical unbound twin features and a
> generator-matched spectral basis. This experiment is retained only as a
> plumbing/identifiability pilot. See
> `audit/binding_cpu_learned_v1_saturation_diagnosis.md` and the independent
> anti-saturation v2 result.

日期：2026-09-03
冻结决策：**LEARNED_CPU_AUDIT_TRANSPORTS**
设备：CPU-only；没有导入或调用 PyTorch/CUDA、ManiSkill、PhysX。

## 一句话结论

动作—结果绑定方向在五个独立种子、ID、参数 OOD、带噪 OOD 和 held-mechanism
上均稳定通过预注册门槛。成功来自 action-effect correspondence：同容量 unbound
模型和全部边缘控制均为 0.5000，擦除绑定使模型回到 0.5000，而完整搬移 pairs
不改变结果。

## 冻结结果

训练集在每个种子包含 512 个 harmonic-2 twins 和 512 个 harmonic-4 twins；每个
评测 split 为 256 twins。预注册 SHA-256 为
`12e94be22289f00b3926d949b9d224fa89d8c1cf054bb92b6f882cd43a16ce32`，
与结果文件内记录一致。全量实验耗时 31.63 秒。

| Split | Bound Ridge TPA | 最差种子 | Unbound TPA | Analytic TPA | Binding drop | Normalized regret |
|---|---:|---:|---:|---:|---:|---:|
| ID | 0.9962 | 0.9934 | 0.5000 | 0.9961 | 0.4962 | 0.000063 |
| Parameter OOD | 0.9939 | 0.9932 | 0.5000 | 0.9942 | 0.4939 | 0.000086 |
| Noise OOD | 0.9875 | 0.9822 | 0.5000 | 0.9875 | 0.4875 | 0.000764 |
| Held mechanism | 0.9966 | 0.9949 | 0.5000 | 0.9966 | 0.4966 | 0.000052 |

各 split 的五种子 bound TPA：

- ID：0.9985 / 0.9937 / 0.9934 / 0.9976 / 0.9978；
- parameter OOD：0.9944 / 0.9932 / 0.9937 / 0.9951 / 0.9932；
- noise OOD：0.9851 / 0.9878 / 0.9907 / 0.9822 / 0.9919；
- held mechanism：0.9968 / 0.9949 / 0.9968 / 0.9968 / 0.9976。

所有 split 的 strongest marginal、class/current、candidate slot、action-only、
result-only 和 arrow-action TPA 都是 0.5000；Top-1 crossing 都是 1.0000。

## 干预审计

| Split | Pair-preserving | Binding erased | Twin-history swap | Slot reversal | Coordinate rotation |
|---|---:|---:|---:|---:|---:|
| ID | 0.9962 | 0.5000 | 0.0038 | 0.9962 | 0.9961 |
| Parameter OOD | 0.9939 | 0.5000 | 0.0061 | 0.9939 | 0.9936 |
| Noise OOD | 0.9875 | 0.5000 | 0.0125 | 0.9875 | 0.9875 |
| Held mechanism | 0.9966 | 0.5000 | 0.0034 | 0.9966 | 0.9967 |

这组结果比“shuffle 后下降”更有辨识力：pair-preserving 也改变序列位置但完整保留
对应关系，得分完全不变；binding erasure 保留动作和结果的各自 multiset，仅破坏
对应关系，得分回到机会水平。因而当前证据支持的是 binding necessity，而不是泛泛的
顺序敏感或时间箭头。

## 必须保留的限制

1. Bound Ridge 几乎等于 analytic spectral witness。它证明一个 CPU-trainable
   verifier 能传输该审计信号，但**没有证明新 learned architecture 超过解析方法**。
2. Top-1 accuracy 分别只有 0.6762、0.6492、0.5270、0.8043，尽管 normalized
   regret 只有 `5e-5` 到 `8e-4`。原因是密集候选网格包含很多近等价最优项；论文中
   应把 regret 和 candidate-conditioned TPA 作为主指标，不能只展示接近 1 的 TPA，
   也不能声称 exact argmax 已解决。
3. Held mechanism 是训练基函数的未见组合，而不是未知频率或结构完全不同的动力学。
   它支持 compositional transport，不支持开放世界动力学泛化。
4. 数据仍是符号/解析生成。没有视觉观测、接触物理、执行误差或真实机器人证据。

## 方向调整

立即停止把 generic arrow-of-time 当成主论文方向。当前优先级最高的题目应改为：

> **Counterfactual Action-Effect Binding Audit for History-Aware Robot
> Verifiers**：在 current state、candidate set、action/result marginals 和顺序统计
> 完全匹配时，验证模型是否真正保留“哪个动作导致哪个后果”。

论文贡献应优先定位为 evaluation/benchmark protocol：matched twins、
pair-preserving 对照、binding-breaking 对照、candidate-conditioned TPA、regret 和
parameter/noise/mechanism OOD。新模型只能作为次级贡献，除非后续 raw-token 模型在
不注入正确 Fourier basis 的情况下也稳定通过，并在更真实日志上显示出超越强解析
baseline 的必要性。

下一阶段仍应保持 CPU-only：冻结 raw-token Ridge/MLP/DeepSets 小模型和历史长度、
probe 缺失、非均匀采样、未知响应频率等 stress tests；然后用公开或已有预计算特征
评估 HAVE/TeV/AEM 风格 verifier。只有新的独立 gate 通过后才建议规划 GPU，当前
结果不授权 GPU。

## 复现与证据

```powershell
python scripts\cpu_binding_learned_v1.py `
  --output audit\results\binding_cpu_learned_v1.json
```

- 预注册：`docs/binding_cpu_learned_v1_preregister.md`
- 实现：`scripts/cpu_binding_learned_v1.py`
- 原始结果：`audit/results/binding_cpu_learned_v1.json`
- 前置准入：`audit/binding_cpu_v1_pre_admission_report.md`
- 文献矩阵：`docs/cpu_first_direction_literature_2026-09.md`
