# BindingCheck-CPU-v1 预筛报告

日期：2026-09-03
状态：**CPU_DIRECTION_ADMITTED**
设备边界：仅 CPU；未导入 PyTorch、ManiSkill 或 PhysX，未调用 GPU。

## 结论

“动作—结果绑定审计”通过了预注册的方向准入门槛。这个结论只说明当前
matched-twin 构造确实需要知道“哪个动作产生了哪个结果”，而不是依赖当前状态、
候选槽位、动作/结果边缘分布或简单时间箭头。它不证明新的 learned verifier 优于
解析系统辨识，也不支持视觉、仿真或真实机器人效果声明。

## 冻结实验结果

每个任务使用 5 个预注册种子，每个 task/seed 为 256 个 matched twins、每个
twin 两个分支和 16 个候选。全部结构完整性检查通过，两个任务的 Top-1 crossing
均为 1.000。

| 任务 | 最强边缘控制 | Arrow | Nearest | kNN | Fourier binding witness | 破坏绑定后 | 完整 pair 搬移后 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| discrete higher-order | 0.5000 | 0.5000 | 0.6875 | 0.7500 | 1.0000 | 0.6250 | 1.0000 | PASS |
| continuous nonlinear | 0.5000 | 0.5000 | 0.8306 | 0.8315 | 0.9982 | 0.4997 | 0.9982 | PASS |

连续非线性任务的五个种子上，Fourier witness 分别为 0.9984、0.9984、
0.9984、0.9993、0.9964；不存在只靠挑选最好种子得到的通过。离散任务五个种子
均为 1.0000。

关键因果对照符合预注册预期：

- 完整移动 action-result pairs，两个任务的分数变化均为 0；
- 只移动 results、破坏对应关系，分数分别下降 0.3750 和 0.4985；
- 交换 twin histories 后分别降至 0.0000 和 0.0018；
- 候选槽位反转和整体旋转没有改变准入结论；
- 所有 current/action/result/slot 边缘控制均为 0.5000。

运行耗时 24.43 秒。原始 JSON 保留每个 task/seed 的完整指标、per-template
结果和所有 gate，而不是只保存聚合数值。

## 科学解释

这次通过与 TimeArrow-v3 的失败形成了清楚的分界：TimeArrow-v3 的 unordered
history 几乎保留全部性能，因此不能支撑“时序推理”结论；本实验完整打乱 pairs
不影响结果，而只破坏 action-effect binding 会造成大幅下降。当前更可靠的论文问题
不是“模型是否知道时间正向”，而是“history-aware verifier 是否保留了干预动作与
后果的对应关系”。

同时，解析 Fourier 系统辨识几乎饱和，说明当前结果支持的是**评测协议/诊断基准**
方向，而不是已经证明一个新神经网络方法有增益。下一阶段必须独立预注册 learned
CPU baseline、parameter/noise OOD 和 held-mechanism；如果 learned model 没有稳定
利用绑定，或者只复现解析规则，就应把论文主贡献限定为 benchmark/audit。

## 可复现命令

```powershell
python scripts\cpu_binding_pre_admission.py `
  --output audit\results\binding_cpu_v1_pre_admission.json
```

对应预注册：`docs/binding_cpu_v1_preregister.md`
原始结果：`audit/results/binding_cpu_v1_pre_admission.json`
