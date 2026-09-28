# 参考文献与第三方项目

本目录记录本项目实际借鉴或依赖的论文、开源实现和预训练模型。它不是模型权重目录；权重许可和本地路径见 [LICENSING.md](LICENSING.md)。

## 直接参考

- [UniWorld-View](UniWorld-View.md)：遮挡感知点云渲染、三重回投影、几何条件与参考外观分离。Wan/VACE 仅作为未来 residual-hole 视频扩散后端。
- [Depth Anything 3](BASE_MODELS.md#depth-anything-3)：DA3Metric-Large 作为 RGB 对齐稠密表面候选。
- [DINOv2](BASE_MODELS.md#dinov2)：语义和边界描述子。
- [AnyUp](BASE_MODELS.md#anyup)：DINO 特征空间上采样。

## 研究路线边界

本项目是 calibrated data reconstruction，不是 AIGC 新视角绘图。真实源 RGB 的 observed 区域逐像素保留；生成模型只允许处理真实 disocclusion。任何只凭文本/随机噪声生成整图的结果都不属于本项目正式指标。

完整 bib 条目见 [CITATIONS.bib](CITATIONS.bib)。
