# 抽样复现说明

## 环境

推荐 Ubuntu 20.04、NVIDIA GPU 24 GB、CUDA 12.x。当前 3090 实测服务器为 RTX 3090 24 GB、驱动 550.127.05；Viewer 与 3DGS 使用独立 Python 环境。

## 启动

```bash
export SE3_STAGE=/data/temporal_sequence_2026-05-22-10-25-21
export SE3_OUT=/data/temporal_sequence_output_2026-05-22-10-25-21
export SE3_GS_RENDER_URL=http://127.0.0.1:8110/render
export SE3_CURRENT_EWA=1
export SE3_RIG_DENSE_MULTIVIEW=1
export SE3_RIG_DENSE_MULTIVIEW_STEP=2
export SE3_DIRECTIONAL_BACKGROUND=1
export SE3_INFORMATION_DOMAIN=1
bash scripts/run_submission_demo.sh
```

浏览器打开：

```text
http://127.0.0.1:8142/?v=20260818-workbench-inline-v2
```

抽样复现固定 frame 40 和 `config/target_l4_rig_v1.yaml`。系统只移动共享的 `T_L_R_view`，并由固定 `T_R_Ci_mount` 派生七路相机目标位姿。先关闭 3DGS/视觉补偿验证几何 RGB-D，再开启视觉补偿查看最终输出。页面应同时出现畸变源图、未经美化的畸变几何重建、最终畸变图，以及 ABS DIFF/像素来源图；详细输出契约见 `../../docs/current_architecture_v2.md`。

当前候选报告：`04_generated_results/current_rig/current_rig_report.json`，record 为 `20260818_173153_rig_e2643d`。

## 资源要求

完整重建需要 DA3 深度、80 帧全局位姿、时序融合、全局 Surfel/TSDF 缓存及七个 3DGS 模型。抽样复现无需重新训练 3DGS，但必须提供模型权重、源码 revision 和对应 SHA-256。

LaMa 是独立的视觉服务：

```bash
export LAMA_MODEL=/home/heqing/models/lama/big-lama.pt
bash scripts/run_ai_inpaint_service.sh
export SE3_AI_INPAINT=1
export SE3_AI_INPAINT_URL=http://127.0.0.1:8112/inpaint
```

若模型或服务不可用，Viewer 自动保留几何结果或 OpenCV/3DGS 视觉结果，不影响几何重建。

## 当前候选版验收

```text
materials commit: 8d8c00a
algorithm baseline: 9ab7995
viewer:     http://127.0.0.1:8142/?v=20260818-workbench-inline-v2  (3090 服务器本机)
report:     20260818_173153_rig_e2643d
frame:      40
cameras:    7
failures:   []
```

关键原则：`final_distorted_polished` 只能作为观看输出；`reconstruction_distorted`、深度图和 SE(3) 优化只使用受信几何来源。任何视觉补偿像素都标为 `visual-only`，不得回写深度。
