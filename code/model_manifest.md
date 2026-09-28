# 联合方案模型与依赖清单（待最终核验）

本表服务于 `l4-se` + `gcr-nvs-inference-only-20260826` 的比赛复现。`GCR-NVS inference-only` 只提交冻结推理定义、脚本、配置和测试，不包含训练入口；训练分支仅作为研发来源登记，不作为本候选包的运行依赖。

| 组件 | 用途 | 状态 | 交付方式 |
|---|---|---|---|
| DA3Metric-Large | RGB 对齐稠密深度候选 | 正式使用 | 提交来源、版本、许可证和 SHA256，不提交权重本体 |
| T0StructureConstraintNet | LiDAR 锚定结构校准 | 正式推理使用 | 记录 checkpoint 路径、版本和 SHA256；训练入口不属于 inference-only 提交 |
| DINOv2-L/14 | 语义描述子 | 正式使用 | 记录上游来源、权重版本和许可证 |
| AnyUP | 特征上采样/边界对应 | 正式使用 | 记录 checkpoint、代码 revision 和许可证 |
| Stable Diffusion Inpainting | A-only residual hole completion 基础模型 | 可选正式补全轨 | 记录模型卡和许可证 |
| RGB-D-S A-only adapter | 只写 residual hole | GCR-NVS-A 路线，需以最终实测开关确认 | 记录 checkpoint、训练数据契约和 generated fraction；不允许修改 observed/gate 像素 |
| OpenCV | 去畸变、重投影辅助、视觉清理 | 正式依赖 | 记录版本 |
| Open3D/TSDF/Surfel | 旧几何轨结构处理 | 旧方案依赖 | 记录版本和使用模块 |
| 3DGS | 旧方案 visual-only 视觉补偿 | 非几何主轨 | 标记不参与深度/SE(3) |
| Wan/VACE | 视频扩散/4D 预留 | 当前关闭 | 不作为当前正式结果，不在 Demo 中宣称已经完成 |

提交前必须补齐：来源 URL、许可证、代码 revision、权重文件名、SHA256、实际启用状态和是否影响几何输出。对每个生成模块还要记录 `observed_lock=true`、`hole_only=true` 和 `exports_generated_depth=false`；否则不能进入正式联合结果。
