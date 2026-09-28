# 基础模型与可替代性

## Depth Anything 3

- 当前：`DA3Metric-Large`，输入去畸变 RGB，输出 camera-z 稠密深度和 confidence。
- 作用：提供连续、与 RGB 对齐的表面候选；不替代真实 LiDAR。
- 可替代：可以改用 Depth Anything V2（DA2）或其他可许可的单目深度模型，只需实现 `DepthEstimator` 接口并输出相同的 `depth/confidence/camera-z` 契约。
- 影响：替换 DA3 会改变表面 cache，必须重新做 LiDAR holdout、边界、coverage 和 RGB source reprojection 验收；不能直接混用旧 cache。

## DINOv2

- 当前：结构 checkpoint 使用 B/14 条件；最终高分辨率语义审计使用 L/14 约 644×364 输入。
- 作用：语义对应、物体边界和 source reference 检索，不直接生成 RGB。
- 可替代：DINOv2-B/S/L、SigLIP 或其他冻结视觉 encoder，只要提供稠密 descriptor 和稳定的归一化 cosine 接口。

## AnyUP

- 当前：`anyup_multi_backbone.pth`。
- 作用：将 DINO 特征上采样到更高空间网格，保持边界和局部语义；不能创造未观测颜色。
- 可替代：双线性/内容感知上采样或其他 feature upsampler，但必须重新检查 correspondence precision 和边界指标。

## Stable Diffusion Inpainting / A-only adapter

- 当前：冻结 Stable Diffusion inpainting 主干，只训练 T2I Adapter；正式路由为 A-only。
- 作用：残余 true hole 的 RGB-D-S 局部补全。
- 边界：不允许修改 observed T0 或 UniWorld gate 像素；生成结果不能当真实观测。

## Wan2.1/VACE

- 当前：未加载，仅 `video_diffusion_backend.py` lazy interface。
- 未来：geometry/control + reference token 注入视频扩散，用于 residual temporal hole completion。
