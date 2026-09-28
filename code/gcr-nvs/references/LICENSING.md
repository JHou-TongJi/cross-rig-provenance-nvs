# 依赖与许可记录

仓库只提交自有代码、配置、文档和小型样例，不提交第三方大模型权重。部署前必须根据实际下载版本核对许可。

| 依赖 | 本地路径/来源 | 用途 | 许可动作 |
|---|---|---|---|
| Depth Anything 3 | `/home/heqing/Depth-Anything-3` | 稠密深度候选 | 按上游仓库/权重许可 |
| Depth Anything 2（替代） | 外部可选 | DA3 替代 | 按上游许可，需重建 cache |
| DINOv2 | torch hub/本地缓存 | 语义描述子 | 按 Meta 上游许可 |
| AnyUP | `third_party/anyup` | 特征上采样 | 按上游仓库许可 |
| Stable Diffusion inpainting | `/media/2T_HD/GCR-NVS_DATA/models/runwayml-stable-diffusion-inpainting` | A-only hole completion | 按模型卡和上游许可 |
| UniWorld-View/Wan/VACE | `/tmp/UniWorld-View`（参考） | 未来视频扩散 | 当前未加载；接入前单独核对 |

禁止把外部模型权重复制进 Git 或在没有许可的情况下打包发布。
