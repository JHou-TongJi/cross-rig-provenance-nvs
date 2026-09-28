"""Build the lightweight, auditable submission-material index.

Large RGB/PCD/cache/video files remain on the 2T data disk.  This script
creates the required CSVs and markdown index files with absolute source paths,
parameters, and frame lineage so an evaluator can locate or reproduce them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from gcr_nvs.datasets.manifest import build_sequence_manifest, sensor_timestamp


DATA = Path("/media/2T_HD/GCR-NVS_DATA")
POSE = "mixed_10_20cm"
SEQUENCE = "2026-05-22-10-25-21"
CAMERAS = ("CAM_BACK", "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT")
VIDEO_ROOT = DATA / "outputs/videos_240fps_comparison_20260826_20cm" / POSE
AONLY_ROOT = DATA / "outputs/evaluations/t0_uniview_rgbds_full_a_20260825"


def sha256(path: Path, limit: int | None = None) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb") as stream:
        while True:
            chunk_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if chunk_size <= 0:
                break
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_quality(out: Path) -> None:
    report_path = AONLY_ROOT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for pose, cameras in report.items():
        for camera, item in cameras.items():
            result = AONLY_ROOT / pose / camera / "rgbds_a_4k.png"
            diff = DATA / "outputs/source_vs_aonly_diff_20260826" / pose / camera / "a_only_abs_diff.png"
            rows.append({
                "sequence": SEQUENCE,
                "frame": 73,
                "pose": pose,
                "camera": camera,
                "resolution": "1920x1080 or source-native audit",
                "observed_coverage": item.get("original_2px_coverage"),
                "gate_fill_of_hole": item.get("gate_filled_fraction_of_hole"),
                "generated_fraction_all": item.get("generated_fraction"),
                "vehicle_risk_fraction_of_hole": item.get("vehicle_risk_fraction_of_hole_low"),
                "locked_pixels_modified": item.get("locked_pixels_modified"),
                "result_path": str(result),
                "diff_path": str(diff),
                "metric_status": "measured proxy; no target RGB GT for shifted pose",
            })
    write_csv(out / "05_评价结果/quality_metrics.csv", rows, list(rows[0]))


def build_frame_mapping(out: Path, sequence_root: Path, fps: int) -> int:
    records = build_sequence_manifest(sequence_root, compute_quality=False)
    rows: list[dict[str, object]] = []
    output_index = 0
    for interval, (left, right) in enumerate(zip(records[:-1], records[1:])):
        slices = max(1, round((sensor_timestamp(right) - sensor_timestamp(left)) * fps))
        steps = [0] if interval == 0 else []
        steps.extend(range(1, slices))
        steps.append(slices)
        for step in steps:
            alpha = step / slices
            for camera in CAMERAS:
                rows.append({
                    "output_index": output_index,
                    "output_time_s": f"{output_index / fps:.6f}",
                    "fps": fps,
                    "pose": POSE,
                    "camera": camera,
                    "left_anchor_frame": left.frame_id,
                    "right_anchor_frame": right.frame_id,
                    "left_timestamp_s": f"{sensor_timestamp(left):.6f}",
                    "right_timestamp_s": f"{sensor_timestamp(right):.6f}",
                    "alpha": f"{alpha:.8f}",
                    "interpolation": "timestamp-aware linear RGB",
                    "source_video": str(VIDEO_ROOT / f"{camera}_source_{fps}fps.mp4"),
                    "t0_blue_video": str(VIDEO_ROOT / f"{camera}_t0_blue_{fps}fps.mp4"),
                    "a_only_video": str(VIDEO_ROOT / f"{camera}_a_only_{fps}fps.mp4"),
                })
            output_index += 1
    write_csv(out / "04_生成结果/frame_mapping.csv", rows, list(rows[0]))
    return output_index


def build_lineage(out: Path) -> None:
    rows = [
        {"artifact": "raw_dataset", "role": "188 sequences / RGB / PCD / camera_config", "path": str(DATA / "incoming_sequences"), "status": "external; source data not copied to Git", "license_or_note": "follow challenge data-use and privacy rules"},
        {"artifact": "manifest", "role": "sequence/frame/camera lineage", "path": str(DATA / "dense_route_manifest_188_v1/all_sequences.jsonl"), "status": "external generated manifest", "license_or_note": "sequence-level split; no frame leakage"},
        {"artifact": "rectified_rgb_da3_cache", "role": "Brown rectification + DA3 dense depth", "path": str(DATA / "outputs/da3_depth_cache_dense_train_v1"), "status": "external cache", "license_or_note": "newK, D=0 internal raster"},
        {"artifact": "temporal_lidar_teacher", "role": "current/nearest temporal registered structure", "path": str(DATA / "outputs/geometry_teacher_full_v1"), "status": "external cache", "license_or_note": "LiDAR is geometry-only"},
        {"artifact": "structure_checkpoint", "role": "T0StructureConstraintNet", "path": str(DATA / "t0_baseline_isolated_20260823/checkpoints/t0_structure_rectified_v2_dinob14_320x180_c48_best.pt"), "status": "frozen validated checkpoint", "license_or_note": "SHA256 in model registry"},
        {"artifact": "a_only_checkpoint", "role": "RGB-D-S residual hole adapter", "path": str(DATA / "outputs/checkpoints/t0_rgbds_adapter_a_256x64_v2/adapter_step_001000.pt"), "status": "frozen validated checkpoint", "license_or_note": "hole-only; observed immutable"},
        {"artifact": "single_frame_results", "role": "3 poses × 7 cameras A-ONLY", "path": str(AONLY_ROOT), "status": "validated 21/21", "license_or_note": "source vs A-only DIFF available"},
        {"artifact": "video_demo", "role": "80 real anchors → 240 FPS comparison", "path": str(VIDEO_ROOT), "status": "validated 1920x1080 / 39.5 s", "license_or_note": "timestamp-aware linear interpolation, not new geometry"},
    ]
    write_csv(out / "05_评价结果/data_lineage.csv", rows, list(rows[0]))


def build_video_manifest(out: Path) -> None:
    rows: list[dict[str, object]] = []
    all_video = VIDEO_ROOT / "all_cameras_comparison_source_blue_a_only_240fps.mp4"
    candidates = [all_video]
    candidates.extend(VIDEO_ROOT / f"{camera}_{variant}_240fps.mp4" for camera in CAMERAS for variant in ("source", "t0_blue", "a_only"))
    candidates.extend(VIDEO_ROOT / f"{camera}_comparison_source_blue_a_only_240fps.mp4" for camera in CAMERAS)
    for path in candidates:
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("all_cameras"):
            kind, camera = "three-view contact video", "CAM_BACK + CAM_FRONT_NARROW + CAM_FRONT_RIGHT"
        elif "comparison" in name:
            kind, camera = "source / T0-blue / A-only comparison", name.split("_comparison")[0]
        elif "_source_" in name:
            kind, camera = "rectified source RGB", name.split("_source_")[0]
        elif "_t0_blue_" in name:
            kind, camera = "T0 blue-hole audit", name.split("_t0_blue_")[0]
        else:
            kind, camera = "A-only final RGB", name.split("_a_only_")[0]
        rows.append({"file": name, "path": str(path), "kind": kind, "camera": camera,
                     "pose": POSE, "sequence": SEQUENCE, "fps": 240,
                     "resolution": "1920x1080", "duration_s": 39.504,
                     "frame_count": 9481, "bytes": path.stat().st_size,
                     "contract": "80 real anchors; timestamp-aware linear RGB interpolation"})
    write_csv(out / "04_生成结果/video_manifest.csv", rows, list(rows[0]) if rows else ["file", "path"])


def write_markdown(out: Path, frame_count: int) -> None:
    (out / "01_技术方案").mkdir(parents=True, exist_ok=True)
    (out / "02_数据说明").mkdir(parents=True, exist_ok=True)
    (out / "03_工程代码").mkdir(parents=True, exist_ok=True)
    (out / "04_生成结果").mkdir(parents=True, exist_ok=True)
    (out / "05_评价结果").mkdir(parents=True, exist_ok=True)
    (out / "06_Demo展示").mkdir(parents=True, exist_ok=True)
    (out / "01_技术方案/技术方案报告.md").write_text(
        "# 技术方案报告索引\n\n"
        "完整方案见仓库 `docs/FINAL_RELEASE_ROUTE_2026-08-26.md`、`docs/END_TO_END_INFERENCE_CONTRACT_2026-08-26.md` 和 `docs/FINAL_ARCHITECTURE_CN_2026-08-26.md`。\n\n"
        "核心路线：去畸变 RGB + 真实稀疏/时序 LiDAR → DA3 稠密表面 → T0 结构约束 → SE(3)/z-buffer → 真实 RGB 采样 → UniWorld/DA3 gate → A-ONLY residual hole。\n",
        encoding="utf-8")
    (out / "02_数据说明/数据与传感器说明.md").write_text(
        "# 数据与传感器说明\n\n"
        "正式数据为 188 条序列，通常每条 80 帧、7 路 RGB、1 路 LIDAR_CONCAT PCD 和逐帧 camera_config。完整数据盘点见 `docs/DATA_INVENTORY_2026-08-22.md`。\n\n"
        "源车型内外参与 Brown 畸变参数来自数据集；目标 L4 采用 `configs/target_rigs/l4_simulated_3700x1400x2000.yaml` 的 simulation-only 假设，不能替代实车联合标定。\n",
        encoding="utf-8")
    (out / "03_工程代码/运行说明.md").write_text(
        "# 工程运行说明\n\n"
        "代码入口见根目录 `README.md`；发布分支为 `release/gcr-nvs-inference-only-20260826`，开发分支为 `final/gcr-nvs-t0-uniview-a-20260826`。\n\n"
        "最小检查：`PYTHONPATH=src python scripts/verify_final_project.py --manifest /media/2T_HD/GCR-NVS_DATA/dense_route_manifest_188_v1/all_sequences.jsonl`。完整 Hybrid 命令见 `docs/INFERENCE_RUNTIME_ENVIRONMENT_2026-08-26.md`。\n",
        encoding="utf-8")
    (out / "04_生成结果/生成结果索引.md").write_text(
        "# 生成结果索引\n\n"
        f"单帧 A-ONLY 结果：`{AONLY_ROOT}`，三档位姿、7 路相机共 21 个 case。\n\n"
        f"240 FPS Demo：`{VIDEO_ROOT}`。总览文件：`{VIDEO_ROOT}/all_cameras_comparison_source_blue_a_only_240fps.mp4`。\n\n"
        f"视频由 80 个真实锚帧生成，共 {frame_count} 帧、1920×1080、约 39.5 秒；逐帧来源见 `frame_mapping.csv`。\n",
        encoding="utf-8")
    (out / "05_评价结果/实测结果说明.md").write_text(
        "# 实测结果说明\n\n"
        "`quality_metrics.csv` 来自 A-ONLY `report.json`；`data_lineage.csv` 记录外部数据、cache、checkpoint、结果和视频路径。\n\n"
        "已测：observed coverage、gate 填洞代理、生成占比、车辆风险 hole、锁定像素。几何误差、语义保留率和时序异常率没有独立真值的部分明确标为待测/代理，不虚构指标。\n",
        encoding="utf-8")
    (out / "06_Demo展示/Demo说明.md").write_text(
        "# Demo 展示说明\n\n"
        "推荐播放 `04_生成结果` 中的 240 FPS 三联对比视频，包含 source RGB、T0 蓝色空洞审计和 A-ONLY 最终结果。该视频满足手册“连续场景、不少于 100 帧”的展示要求（80 个真实锚帧，240 FPS 插帧后超过 9,000 帧）。\n\n"
        "注意：中间帧是按传感器时间戳的 RGB 线性插值，不是重新执行 240 FPS 三维重建；残影应作为边界说明，不得宣称为视频扩散生成。\n",
        encoding="utf-8")
    (out / "05_评价结果/failure_cases.md").write_text(
        "# 失败案例与已知边界\n\n"
        "- 20–50 cm 位姿会产生真实 disocclusion；A-ONLY 生成占比上升，不代表观测覆盖率提升。\n"
        "- CAM_FRONT_NARROW 与 CAM_BACK_LEFT 是弱视角，应单独检查，不用全局平均掩盖退化。\n"
        "- 240 FPS 视频的中间帧为线性 RGB 插值，快速转弯和遮挡变化仍可能出现残影；根治需要更密集真实帧或时空模型。\n"
        "- L4 目标相机为公开车体尺寸基础上的仿真 rig；实车部署必须重新测量 K/D/T、安装高度、时间同步和 LiDAR-相机外参。\n"
        "- 评估蓝色 validity 仅是审计标记，不参与 RGB 或 DIFF。\n",
        encoding="utf-8")
    (out / "README.md").write_text(
        "# GCR-NVS 提交材料索引\n\n"
        "本目录按《中安智联赛题实测手册》组织轻量提交材料。大体积原始数据、cache、checkpoint 和 MP4 保留在 2T 数据盘，本文档通过绝对路径、参数和 CSV 血缘指向它们。\n\n"
        "必看：`04_生成结果/frame_mapping.csv`、`05_评价结果/quality_metrics.csv`、`05_评价结果/data_lineage.csv`、`06_Demo展示/Demo说明.md`。\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("submission"))
    parser.add_argument("--sequence-root", type=Path, default=DATA / "incoming_sequences" / SEQUENCE)
    parser.add_argument("--fps", type=int, default=240)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame_count = build_frame_mapping(args.output, args.sequence_root, args.fps)
    build_quality(args.output)
    build_lineage(args.output)
    build_video_manifest(args.output)
    write_markdown(args.output, frame_count)
    print(json.dumps({"output": str(args.output), "video_fps": args.fps, "mapping_rows": frame_count * len(CAMERAS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
