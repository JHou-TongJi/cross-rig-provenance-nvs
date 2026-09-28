# GCR-NVS 0.6.0-t0-uniview-a

## 定稿内容

- 建立完整最终项目分支：`final/gcr-nvs-t0-uniview-a-20260826`。
- 固定 T0 Structure HD、DA3-first dense surface、真实源 RGB 采样、UniWorld/DA3 gate 和 A-only RGB-D-S residual completion。
- 增加 `datasets/` 数据格式、manifest 样例、188 序列统计和转换流程文档。
- 增加 `models/` 模型注册表与模型卡，完整区分正式模型、可选模块、历史消融和 Wan/VACE 预留后端。
- 保留所有 `docs/` 历史技术说明、失败复盘、周报、消融和论文源码分析。
- 增加 `configs/releases/t0_uniview_a_final.yaml` 和 `scripts/verify_final_project.py`。
- 记录昨晚 21/21 结果、模型 SHA256 和外部数据/权重路径。

## 明确未完成

Wan2.1/VACE 14B 没有在当前机器加载，也没有宣称当前版本已经完成 4D 视频扩散。当前代码只提供带 observed-lock/hole-only 契约的 lazy backend。
