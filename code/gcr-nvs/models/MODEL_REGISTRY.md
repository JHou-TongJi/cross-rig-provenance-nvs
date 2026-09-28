# 模型注册表

本文件是模型、权重、输入输出和验收状态的唯一索引。文件名相似但输入契约不同的 checkpoint 不得互换。

## 正式默认链路

### 1. DA3Metric-Large

- 类型：Depth Anything 3 metric monocular depth
- 作用：提供与去畸变 RGB 对齐的连续稠密表面候选
- 输入：单路 rectified RGB
- 输出：camera-z depth、confidence
- 权重：`/home/heqing/models/depth-anything-3/DA3Metric-Large`
- 状态：外部冻结模型
- 边界：不是 LiDAR 真值，不能直接决定公制结构或替换 RGB

### 2. T0StructureConstraintNet

- 文件：`src/gcr_nvs/models/t0_structure_constraint.py`
- 训练入口：`scripts/train_t0_structure_constraint.py`
- 正式权重：`/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823/checkpoints/t0_structure_rectified_v2_dinob14_320x180_c48_best.pt`
- 定档副本：`models/t0_structure_rectified_v2_dinob14_320x180_c48_best.pt`
- 输入：DA3 depth/confidence、当前/时序 LiDAR anchor、RGB edge、ray、SE(3)、semantic feature、dynamic/visibility
- 输出：depth residual、confidence/visibility 和结构校准表面
- 结构：48 channels、320×180 train grid、DINOv2-B semantic condition、恒等初始化
- 状态：正式使用

### 3. DINOv2 + AnyUP

- 文件：`src/gcr_nvs/models/dino_anyup_descriptor.py`、`source_encoder.py`、`feature_upsampler.py`
- 作用：语义对应、物体边界和源像素检索，不直接画最终 RGB
- 当前正式推理：DINOv2-L/14，输入约 364×644；部分结构训练 checkpoint 使用 B/14，配置必须随权重读取
- AnyUP：`/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823/outputs/checkpoints/anyup_multi_backbone.pth`
- 状态：外部预训练冻结；correspondence head 属于消融

### 4. T0 dense surface renderer

- 文件：`scripts/render_t0_highres_source_audit.py`、`src/gcr_nvs/evaluation/render_dense_lidar_reprojection.py`
- 输入：校准表面、源 RGB、source topology、目标 SE(3)
- 输出：RGB、depth、validity、provenance、hole mask
- 关键约束：真实源 RGB 采样；`splat_radius=0` 的正式 4K 审计；无效像素必须单独标记
- 状态：正式使用

### 5. UniWorld/DA3 background gate

- 文件：`scripts/apply_t0_uniview_background_gate.py`
- 作用：source→target→source 三重回投影，筛选有后景证据的 T0 hole
- 输出：gate RGB、visibility mask、gate report
- 状态：正式使用；借鉴 UniWorld-View 的遮挡思想，不加载 Wan

### 6. RGB-D-S A-only adapter

- 文件：`scripts/train_t0_rgbds_diffusion_adapter.py`、`scripts/apply_t0_rgbds_hybrid_to_2px.py`
- 定档权重：`/media/2T_HD/GCR-NVS_DATA/outputs/checkpoints/t0_rgbds_adapter_a_256x64_v2/adapter_step_001000.pt`
- 定档副本：`models/t0_rgbds_adapter_a_256x64_v2_step_001000.pt`
- 输入：masked RGB、depth、dynamic/semantic control、residual hole mask
- 输出：只写 residual hole 的局部 RGB
- 状态：正式使用 A-only；D2 vehicle expert 已否决

### 7. Wan/VACE reserved backend

- 文件：`src/gcr_nvs/models/video_diffusion_backend.py`
- 默认：`backend=disabled`, `enabled=false`
- 作用：未来 60GB 级 GPU 上接入视频扩散，只允许 hole-only + observed-lock
- 状态：接口和契约测试已完成，模型加载和 4D 结果未完成

## Checkpoint 选择规则

必须同时检查 route name、输入 cache contract、坐标系、DINO 配置、分辨率、训练 split 和验证报告。不能只按文件名、step 或单一 PSNR 选择权重。错误 raw `K,D`、旧 sparse raymarch 或泄漏条件生成的 checkpoint 只能放在历史消融目录。

## 端到端输出

部署入口是 `scripts/infer_t0_end_to_end.py`。它只加载冻结 checkpoint 和派生 cache；目标 rig 的 `K/D/T/width/height` 经 rectification 后驱动显式投影，最终输出 RGB、depth、validity、provenance 和报告。Inference-only 分支保留这一入口以及模型定义，但删除训练包和训练脚本。
