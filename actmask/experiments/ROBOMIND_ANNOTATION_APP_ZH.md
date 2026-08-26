# RoboMIND 中文机器人前景标注网站

此网站服务于 `rm_geometry_mask_audit/manual_reference_plan` 的 200 张人工
参考标注帧。它只产生 `robot_only` 二值掩码，不训练模型，也不生成伪掩码。

## 启动

```bash
cd /data/projects/tzh/papers/ActMask
/home/tzh/conda_envs/actmask/bin/python actmask/experiments/robomind_annotation_app.py
```

打开 `http://127.0.0.1:8765`。若通过云平台映射端口后在远程浏览器访问，改为：

```bash
/home/tzh/conda_envs/actmask/bin/python actmask/experiments/robomind_annotation_app.py --host 0.0.0.0 --port 8765
```

默认仅监听 `127.0.0.1`，不会暴露到网络。只有在已经确认平台端口访问策略时才使用
`--host 0.0.0.0`。

## 使用方法

- 用画笔涂抹可见的双臂 Franka 链接、腕部与夹爪；橡皮可修正边界。
- 不涂被操作物、桌面、工具、阴影、反光与背景。
- `Ctrl+S` 保存当前帧；`A`/`D` 切换前后帧；`N` 跳转至下一未标注帧；`Ctrl+Z` 撤销。
- 服务端会将画布掩码阈值化为单通道 `0/255` PNG，并写到该帧在清单中声明的
  `robot_only_masks/` 路径；进度和备注写入 `annotation_progress.json`。

完成全部标注后，运行已有验证器：

```bash
/home/tzh/conda_envs/actmask/bin/python outputs/actmask/rm_geometry_mask_audit/manual_reference_plan/validate_annotations.py --require-complete
```
