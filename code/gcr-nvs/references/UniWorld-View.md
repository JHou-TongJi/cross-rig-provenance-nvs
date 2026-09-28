# UniWorld-View 在 GCR-NVS 中的参考边界

项目源码审计副本：`/tmp/UniWorld-View`；论文：`/tmp/uniworld-view.pdf`。

## 借鉴的机制

1. 源表面到目标表面再回到源表面的 triple reprojection，用于判断 visibility/occlusion/hole。
2. soft z-buffer 和法线/深度竞争，防止前景纹理拖到后景。
3. 几何 control branch 与 reference appearance branch 分离。
4. 未来视频扩散只处理 residual true hole，而不是重绘整张源 RGB。

## 没有直接照搬的部分

- UniWorld-View 的 Wan/VACE 14B 主干当前没有加载；24GB GPU 不适合完整 60GB 级 clip。
- UniWorld 的单目深度不能替代本项目的真实稀疏 LiDAR 公制结构。
- 其源码中的 hard-lock、法线阈值、投影中心等问题已经在 `docs/UNIWORLD_VIEW_CODE_AUDIT_2026-08-25.md` 中记录，不能把论文图等同于当前实现。

## 对本项目的创新转化

```text
真实 current LiDAR > gated temporal LiDAR > DA3 dense surface
              |
              v
      T0 structure constraint
              |
      source RGB / semantic reference
              |
      observed immutable + hole-only completion
```

区别在于 UniWorld 提供的是几何条件生成框架；GCR-NVS 的正式结果首先是可审计的真实数据重建，生成分支不是 RGB 主渲染器。
