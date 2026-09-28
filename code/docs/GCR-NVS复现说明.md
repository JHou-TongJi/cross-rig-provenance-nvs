# GCR-NVS inference-only 复现说明

## 1. 本地分支

本地副本：`../gcr-nvs-inference-only-20260826/`  
远程分支：`release/gcr-nvs-inference-only-20260826`  
当前本地基线：`a34df762972d0d31b4c2a835c19b5d3ef3f1bb34`。

这是推理发布分支，不包含训练入口。训练和 checkpoint 生成仍在 `final/gcr-nvs-t0-uniview-a-20260826`；提交时只保留冻结模型加载、数据转换、推理、渲染、审计和测试代码。

当前示例命令使用的 `configs/target_rigs/l4_all_up_20cm.yaml` 是 GCR-NVS 的推理审计 rig：它继承源相机的相对安装关系，并在车体 z 方向施加 `+0.20 m` 测试位移，统一输出 1920×1080。它不是主办方提供的目标车型 GT，也不等同于 `l4-se/config/target_l4_open_7v_3700x1400x2000.yaml` 的开放式车体参考 preset。最终联合实测必须在 `report.json`、`frame_mapping.csv` 和 `quality_metrics.csv` 中记录实际 `rig_id`、分辨率和参数来源。

## 2. 已验证环境

| 项目 | 版本 |
|---|---|
| OS | Ubuntu 20.04.6 LTS |
| Python | 3.10.4 |
| GPU | NVIDIA RTX 3090 24GB |
| PyTorch | 2.4.0 + cu124 |
| torchvision | 0.19.0 + cu124 |
| CUDA runtime | 12.4 |
| NumPy | 2.1.2 |
| OpenCV | 4.11.0.86 |
| Transformers | 4.51.3 |

核心依赖见新分支 `requirements-inference-core.txt`；RGB-D-S A-only 补全另外需要 `requirements-inference-optional.txt`。确定性 T0/DA3 主链不依赖 diffusers，扩散补全分支需要额外环境和模型。

## 3. 安装与环境变量

```bash
git clone "${GCR_NVS_REPO_URL:-http://<3090-server>:9082/zhouguohao/gcr-nvs.git}"
cd gcr-nvs
git switch release/gcr-nvs-inference-only-20260826
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference-core.txt
python -m pip install -e .
```

大体积数据、模型和缓存应通过环境变量提供，不写死本机路径：

```bash
export GCR_NVS_DATA_ROOT=/path/to/sequences
export GCR_NVS_DA3_ROOT=/path/to/Depth-Anything-3
export GCR_NVS_DA3_MODEL=/path/to/DA3Metric-Large
export GCR_NVS_ANYUP_CHECKPOINT=/path/to/anyup_multi_backbone.pth
export GCR_NVS_ANYUP_ROOT=/path/to/third_party/anyup
export GCR_NVS_CACHE_ROOT=/path/to/da3_depth_cache
export GCR_NVS_TEACHER_ROOT=/path/to/geometry_teacher
export GCR_NVS_DYNAMIC_ROOT=/path/to/dynamic_masks
```

## 4. 端到端入口

```bash
PYTHONPATH=src python scripts/infer_t0_end_to_end.py \
  --source-root "$GCR_NVS_DATA_ROOT" \
  --sequence 2026-05-22-10-25-21 --frame 73 \
  --target-rig configs/target_rigs/l4_all_up_20cm.yaml \
  --output /path/to/output/release_frame73 \
  --cache-root "$GCR_NVS_CACHE_ROOT" \
  --teacher-root "$GCR_NVS_TEACHER_ROOT" \
  --dynamic-root "$GCR_NVS_DYNAMIC_ROOT" \
  --structure-checkpoint /path/to/t0_structure_rectified_v2_dinob14_320x180_c48_best.pt \
  --distortion camera_intric.yaml \
  --anyup-checkpoint "$GCR_NVS_ANYUP_CHECKPOINT" \
  --anyup-root "$GCR_NVS_ANYUP_ROOT"
```

## 5. 输出检查

成功后至少检查：

```text
target_rig/<camera>/t0_highres_blue.png
depth.npy
validity.npy
provenance.npy
report.json
summary.json
```

数组形状应一致；`validity/provenance/depth` 不能只保存 RGB 图而缺失；报告需要记录目标 rig、源帧、模型版本、设备、耗时和生成比例。

## 6. 运行自检

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/verify_final_project.py \
  --manifest /path/to/all_sequences.jsonl
```

远程发布记录为 `105 passed, 12 warnings`。正式提交前要在交付环境重新运行，并把日志、commit、配置 hash 和权重 hash 写入报告。

## 7. 与 l4-se 的关系

`l4-se` 先完成源数据解析、相机去畸变、LiDAR/RGB-D 几何锚定、七相机刚性 rig 和 SE(3) 审计；GCR-NVS 不替换这些输入契约，而是接收结构/深度/语义条件，提升稠密表面和 residual hole 的视觉结果。

两套代码暂不强行复制成一个仓库。提交包中应通过 README 和 manifest 明确两个目录、两个 commit、两个入口以及最终联合输出的来源。
