# GCR-NVS

跨车型、多相机、真实稀疏 LiDAR 约束的新视角重建项目。

当前发布分支：`release/gcr-nvs-inference-only-20260826`

对应训练与实验开发分支：`final/gcr-nvs-t0-uniview-a-20260826`

当前冻结版本：`GCR-NVS-T0-UniWorld-A-2026.08.25`

## 项目是什么

项目名 `GCR-NVS` 是 **Geometry-Constrained Reconstruction for Novel View Synthesis** 的缩写，即“几何约束的新视角重建”。这里的重点是 Reconstruction：真实标定和稀疏 LiDAR 约束三维结构，RGB 负责保真纹理，系统不是把输入图像交给 AIGC 重新绘画。

输入同一时刻的七路去畸变 RGB、当前/邻近时序 LiDAR、相机标定和目标相机的 SE(3) 位姿，输出目标相机原生分辨率的 RGB、深度、可见性、来源和空洞掩码。系统优先保持源图真实颜色；生成模型只处理确实没有源观测的 residual hole。

```text
188 条序列
  ├─ 七路去畸变 RGB ──> DA3Metric-Large 稠密表面
  │                  └─> DINOv2-L + AnyUP 语义/边界特征
  ├─ 当前 LiDAR ──────> 真实公制结构与遮挡锚点
  └─ 邻近 LiDAR ──────> 经过配准和动态门控的结构补偿
                         |
                         v
                 T0StructureConstraintNet
                         |
                 目标 SE(3) + z-buffer
                         |
          真实源 RGB 采样 + provenance/validity
                         |
            UniWorld/DA3 三重回投影 gate
                         |
             A-only RGB-D-S residual completion
```

职责不能混用：LiDAR 不直接绘制 RGB；DA3 不冒充真实深度；DINO/AnyUP 不替代几何；生成器不重绘有效像素。

## 快速开始

```bash
git clone <your-gitlab-url>/gcr-nvs.git
cd gcr-nvs
git switch release/gcr-nvs-inference-only-20260826
python -m pip install -e '.[inference]'
```

大体积依赖通过环境变量指定：

```bash
export GCR_NVS_DATA_ROOT=/media/2T_HD/GCR-NVS_DATA/sequences
export GCR_NVS_DA3_ROOT=/home/heqing/Depth-Anything-3
export GCR_NVS_DA3_MODEL=/home/heqing/models/depth-anything-3/DA3Metric-Large
export GCR_NVS_ARTIFACT_ROOT=/media/2T_HD/GCR-NVS_DATA
```

## 项目目录

```text
src/gcr_nvs/       数据、几何、模型、推理和评估 Python 包
scripts/           数据转换、cache、推理、渲染、审计和可视化入口
configs/           实验、数据 split、相机与目标位姿配置
datasets/          数据格式、示例、manifest 和转换说明
models/            模型卡、checkpoint 注册表和外部权重说明
references/        论文、第三方项目、基础模型、许可和 bib 引用
docs/              技术报告、研发历程、失败实验和论文对比
tests/             单元测试和数据契约测试
```

## 数据集

正式数据由 188 条序列组成，每条序列通常含 80 个时间帧、7 路相机和 1 路 LiDAR。数据不复制进 Git；项目使用 manifest 引用 2T 盘上的原始文件。

先阅读 [datasets/README.md](datasets/README.md)，再执行：

```bash
# 1. 发现序列、去重并生成 train/val/test（只写 manifest）
PYTHONPATH=src python scripts/build_dense_route_manifest.py \
  --formal-root /media/2T_HD/GCR-NVS_DATA/sequences \
  --incoming-root /media/2T_HD/GCR-NVS_DATA/incoming \
  --output-root /media/2T_HD/GCR-NVS_DATA/dense_route_manifest_188_v1

# 2. 审计图像尺寸、时间戳、PCD 字段和缺失文件
PYTHONPATH=src python scripts/audit_dense_route_dataset.py \
  --manifest /media/2T_HD/GCR-NVS_DATA/dense_route_manifest_188_v1/all_sequences.jsonl \
  --output /media/2T_HD/GCR-NVS_DATA/dense_route_manifest_188_v1/dataset_audit.json

# 3. 按 split 生成去畸变 RGB + DA3 深度 cache
PYTHONPATH=src python scripts/build_da3_depth_cache_split.py \
  /media/2T_HD/GCR-NVS_DATA/sequences \
  --split /media/2T_HD/GCR-NVS_DATA/dense_route_manifest_188_v1/dense_route_split_v1.yaml \
  --split-name train \
  --output-root /media/2T_HD/GCR-NVS_DATA/outputs/da3_depth_cache_dense_train_v1 \
  --model-dir "$GCR_NVS_DA3_MODEL" --source-root "$GCR_NVS_DA3_ROOT"
```

示例 manifest、单帧目录结构、相机原生分辨率和去畸变契约见 [datasets/examples/README.md](datasets/examples/README.md)。禁止把 raw `K,D` 与 rectified RGB 混用。

## 模型与 checkpoint

模型职责、输入输出、训练状态和 SHA256 见 [models/MODEL_REGISTRY.md](models/MODEL_REGISTRY.md)。最终版本使用：

| 模块 | 作用 | 定档权重 |
|---|---|---|
| DA3Metric-Large | RGB 对齐的稠密深度候选 | 外部模型路径 |
| T0StructureConstraintNet | LiDAR 锚定 DA3 表面和目标结构 | `t0_structure_rectified_v2_dinob14_320x180_c48_best.pt` |
| DINOv2-L + AnyUP | 语义和边界描述子 | 外部预训练权重 |
| RGB-D-S A-only adapter | residual hole 局部补全 | `adapter_step_001000.pt` |
| Wan/VACE | 未来视频扩散/4D 路线 | 当前关闭，仅预留 |

定档权重和结果位于：

`/media/2T_HD/GCR-NVS_DATA/releases/GCR-NVS-T0-UniWorld-A-20260825/`

## 推理与审计

本分支是 inference-only 发布版，不包含训练代码。训练和 checkpoint 生成请切换到 `final/gcr-nvs-t0-uniview-a-20260826`；本分支只加载已经冻结的 checkpoint 和派生 cache。

### 最小可复现 Demo

`scripts/demo_t0_reconstruction.py` 将去畸变、T0 表面、背景 gate、A-only residual completion、裂缝修补和目标畸变映射拆成独立阶段；默认只运行真实 RGB 采样的 T0 结果，额外阶段必须显式打开：

```bash
PYTHONPATH=src python scripts/demo_t0_reconstruction.py \
  --source-root "$GCR_NVS_DATA_ROOT" \
  --sequence 2026-05-26-13-46-53 --frame 73 \
  --target-rig configs/target_rigs/l4_simulated_3700x1400x2000.yaml \
  --output /media/2T_HD/GCR-NVS_DATA/outputs/gcr_nvs_demo_frame73
```

运行说明、每阶段产物和输入输出契约见 [最小 Demo 说明](docs/DEMO_MINIMAL_INFERENCE_2026-08-26.md)。

### 架构图与过程图

论文式总架构图：`outputs/paper_figures/gcr_nvs_final_architecture_cn_20260826.png`

单帧推理过程追踪图：`outputs/paper_figures/gcr_nvs_aonly_process_trace_cn_20260826.png`

图的可重复生成脚本：`scripts/draw_gcr_nvs_paper_architecture_cn.py`。

4K T0 审计入口：

```bash
PYTHONPATH=src python scripts/render_t0_highres_source_audit.py \
  --target-camera all --depth-mode edge \
  --output runs/t0_highres_source_audit_edge_all
```

UniWorld/DA3 gate 和 A-only 应用入口分别是 `scripts/apply_t0_uniview_background_gate.py` 与 `scripts/apply_t0_rgbds_hybrid_to_2px.py`（正式版本使用 `--route a`）。

统一端到端推理入口：

```bash
PYTHONPATH=src python scripts/infer_t0_end_to_end.py \
  --source-root /media/2T_HD/GCR-NVS_DATA/sequences \
  --sequence 2026-05-22-10-25-21 --frame 73 \
  --target-rig configs/target_rigs/l4_all_up_20cm.yaml \
  --output /media/2T_HD/GCR-NVS_DATA/outputs/inference/l4_all_up_20cm_frame73 \
  --cache-root /media/2T_HD/GCR-NVS_DATA/outputs/da3_depth_cache_dense_train_v1 \
  --teacher-root /media/2T_HD/GCR-NVS_DATA/outputs/geometry_teacher_full_v1 \
  --dynamic-root /media/2T_HD/GCR-NVS_DATA/outputs/dynamic_masks_full_v1 \
  --structure-checkpoint /media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823/checkpoints/t0_structure_rectified_v2_dinob14_320x180_c48_best.pt \
  --distortion camera_intric.yaml \
  --anyup-checkpoint /media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823/outputs/checkpoints/anyup_multi_backbone.pth \
  --anyup-root third_party/anyup
```

目标 rig 支持显式 `K`、Brown `D`、`T_camera_from_vehicle`、`width/height`，也支持从已有相机继承内参并叠加新的车辆位姿。完整输入/输出契约见 [端到端推理说明](docs/END_TO_END_INFERENCE_CONTRACT_2026-08-26.md)。

## 最终结果

昨晚最终 21 个 case 的报告和七视角图：

`/media/2T_HD/GCR-NVS_DATA/outputs/evaluations/t0_uniview_rgbds_full_a_20260825/`

结果契约：5--10 cm、10--20 cm、20--50 cm 混合位姿；21/21 成功；observed/gate 锁定像素修改数为 0。20--50 cm 平均生成区域约占全图 2.10%，这部分是受控补全而非真实观测恢复。

## 文档导航

按《中安智联实测手册》整理的提交材料索引见 [submission/README.md](submission/README.md)，逐项要求核对见 [docs/SUBMISSION_MATERIALS_AUDIT_2026-08-26.md](docs/SUBMISSION_MATERIALS_AUDIT_2026-08-26.md)。

- [最终技术路线与定档说明](docs/FINAL_RELEASE_ROUTE_2026-08-26.md)
- [UniWorld-View 论文/源码对比](docs/UNIWORLD_VIEW_REFERENCE_ANALYSIS_2026-08-25.md)
- [Wan2.1/VACE 预留路线](docs/WAN_VACE_RESERVED_ROUTE_2026-08-26.md)
- [完整研发历程和失败复盘](docs/GCR_NVS_RND_HISTORY_AND_FAILURES_2026-08-24.md)
- [正式 RGB-D-S 实验记录](docs/T0_RGBDS_DIFFUSION_FORMAL_RESULT_2026-08-25.md)
- [188 数据训练计划](docs/T0_FULL188_TRAINING_PLAN_2026-08-24.md)
- [数据清单与审计说明](docs/DATA_INVENTORY_2026-08-22.md)
- [参考论文、第三方项目和许可](references/README.md)
- [端到端推理契约](docs/END_TO_END_INFERENCE_CONTRACT_2026-08-26.md)
- [发布分支内容清单](docs/RELEASE_CONTENTS_2026-08-27.md)
- [外部模型下载说明](docs/EXTERNAL_MODELS_DOWNLOAD_2026-08-27.md)
- [隐私与可移植性审计](docs/PRIVACY_AND_PORTABILITY_AUDIT_2026-08-27.md)
- [最终模型、图片、视频与下载说明](docs/FINAL_ARTIFACTS_AND_DOWNLOADS_2026-08-27.md)

`docs/` 保留全部历史技术说明、周报、消融和事故记录；没有删除失败路线，失败 checkpoint 只是不允许进入默认入口。

推理发布分支：`release/gcr-nvs-inference-only-20260826`。该分支删除训练包和训练入口，只保留 frozen model loading、数据转换/cache、端到端推理、渲染、审计和测试，适合部署或对外复现。

发布机器、Python/CUDA/PyTorch、依赖分层和完整运行命令见 [Inference 运行环境说明](docs/INFERENCE_RUNTIME_ENVIRONMENT_2026-08-26.md)。

## 测试与发布检查

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/verify_final_project.py \
  --manifest /media/2T_HD/GCR-NVS_DATA/dense_route_manifest_188_v1/all_sequences.jsonl
```

Wan/VACE 默认配置必须保持 `backend=disabled`。任何生成后端都必须满足 `hole_only=true`、`observed_lock=true`，否则入口拒绝执行。

发布版还包含 `submission/` 提交材料索引、数据/视频血缘表、最小 Demo 说明和评测结果；大体积原始数据、cache、外部基础模型权重及 MP4 不复制进 Git，而是由 2T 数据盘路径和 manifest 明确引用。
