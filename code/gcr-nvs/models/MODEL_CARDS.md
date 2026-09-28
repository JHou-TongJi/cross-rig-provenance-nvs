# 模型卡与模块清单

## 最终主链路

| 模块 | 代码 | 输入 | 输出 | 参数状态 | 发布状态 |
|---|---|---|---|---|---|
| DA3Metric-Large | 外部 `/home/heqing/Depth-Anything-3` | 去畸变 RGB | camera-z depth/confidence | 冻结 | 正式 |
| T0StructureConstraintNet | `src/gcr_nvs/models/t0_structure_constraint.py` | DA3、LiDAR、ray、semantic、visibility | calibrated dense surface | 训练 checkpoint | 正式 |
| DINOv2/AnyUP | `dino_anyup_descriptor.py`, `feature_upsampler.py` | RGB | semantic descriptors | 冻结/可选 head | 正式语义条件 |
| Dense reprojection | `geometry/dense_depth_alignment.py`, `render_t0_highres_source_audit.py` | surface + RGB + pose | RGB/depth/valid/provenance | 确定性 | 正式 |
| UniWorld gate | `apply_t0_uniview_background_gate.py` | surface layers + visibility | gated background RGB/mask | 确定性 | 正式 |
| RGB-D-S A-only | `train_t0_rgbds_diffusion_adapter.py` | masked RGB-D-S + hole | residual RGB | adapter only | 正式 |

## 可复现实验模型

| 模块 | 代码/权重 | 用途 | 当前状态 |
|---|---|---|---|
| TemporalLiDARAdapter | `models/temporal_lidar_adapter.py`, `train_temporal_lidar_adapter_stream.py` | 当前帧稀疏点的时序结构补偿 | 可选结构 teacher |
| SemanticCorrespondenceHead | `models/semantic_correspondence_head.py` | DINO-L/AnyUP source-target correspondence | 消融/待进一步训练 |
| SemanticSurfaceCompletion | `models/semantic_surface_completion.py` | 分层深度与语义表面补全 | pilot，不进入默认结果 |
| DenseWarpCompletion | `models/dense_warp_completion.py` | DA3 surface warp 的局部修补 | pilot |
| DeepFillV2Reference | `models/deepfill_v2_reference.py` | 旧 RGB hole baseline | 已隔离 |
| GeometryAppearanceNVS | `models/geometry_appearance_nvs.py` | 早期统一几何外观网络 | 历史消融 |
| Unified3DField | `models/unified_3d_field.py` | 早期 sparse raymarch/field 路线 | 已隔离，不能替代 T0 |
| FoundationDINO | `models/foundation_dino.py` | 早期 plane-sweep + DINO 代价体 | 历史消融 |
| PlaneSweep | `training/train_plane_sweep.py` | 光度多视图深度估计 | 历史消融，不作为结构主导 |
| Restormer/Refiner | `models/restormer.py`, `models/refiner.py` | RGB 残差细化 | 历史消融 |
| WanVACEBackend | `models/video_diffusion_backend.py` | 未来视频扩散/4D hole completion | lazy reserved, disabled |

## 权重位置

最终 checkpoint 在 2T 盘定档目录和原隔离包中各有一份。第三方模型不进 Git：

```text
/home/heqing/models/depth-anything-3/DA3Metric-Large
/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823/outputs/checkpoints/anyup_multi_backbone.pth
/media/2T_HD/GCR-NVS_DATA/models/runwayml-stable-diffusion-inpainting
```

每个正式权重都必须在结果 manifest 中记录 route、输入 cache、坐标契约、训练 split 和 SHA256。
