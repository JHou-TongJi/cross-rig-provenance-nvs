"""Run a reproducible DA3 + LiDAR metric-alignment audit on a rectified probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from gcr_nvs.geometry.dense_depth_alignment import align_monocular_depth
from gcr_nvs.models.monocular_depth import DepthAnything3Estimator


def _min_pool(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    source_height, source_width = depth.shape
    sy, sx = source_height // height, source_width // width
    if sy < 1 or sx < 1:
        return cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    cropped = depth[:height * sy, :width * sx]
    pooled = np.where(
        np.isfinite(cropped) & (cropped > 0), cropped, np.inf,
    ).reshape(height, sy, width, sx).min(axis=(1, 3))
    pooled[~np.isfinite(pooled)] = 0.0
    return pooled.astype(np.float32)


def _colorize(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = depth[valid & np.isfinite(depth) & (depth > 0)]
    canvas = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not values.size:
        return canvas
    lo, hi = np.percentile(values, [1.0, 99.0])
    log_depth = np.log(np.clip(depth, max(float(lo), 1e-3), max(float(hi), float(lo) + 1e-3)))
    normalized = np.clip((log_depth - np.log(max(float(lo), 1e-3))) /
                         max(np.log(max(float(hi), float(lo) + 1e-3)) - np.log(max(float(lo), 1e-3)), 1e-6), 0.0, 1.0)
    canvas = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    canvas[~valid] = 0
    return canvas


def run(probe_root: Path, camera: str, output: Path, model_dir: Path, source_root: Path, device: str) -> dict:
    summary = json.loads((probe_root / "summary.json").read_text())
    entry = next(item for item in summary["cameras"] if item["name"] == camera)
    image_path = probe_root / f"{camera}.rgb_undistorted.jpg"
    raw_lidar = np.load(probe_root / f"{camera}.depth.npy").astype(np.float32)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    estimator = DepthAnything3Estimator(
        model_dir=model_dir,
        source_root=source_root,
        device=device,
    )
    prediction = estimator.predict(
        image_path,
        np.asarray(entry["newK"], dtype=np.float32),
        output_size=(width, height),
    )
    lidar = _min_pool(raw_lidar, height, width)
    result = align_monocular_depth(
        prediction.depth,
        lidar,
        prediction.confidence,
        min_fit_points=24,
    )
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "da3_depth.npy", prediction.depth)
    np.save(output / "da3_confidence.npy", prediction.confidence)
    np.save(output / "lidar_depth.npy", lidar)
    np.save(output / "aligned_depth.npy", result.aligned_monocular_depth)
    np.save(output / "fused_depth.npy", result.depth)
    np.save(output / "lidar_anchor_depth.npy", result.lidar_anchor_depth)
    panel = np.concatenate([
        image,
        _colorize(lidar, result.lidar_validity),
        _colorize(result.aligned_monocular_depth, result.monocular_validity),
        _colorize(result.depth, result.depth > 0),
    ], axis=1)
    cv2.imwrite(str(output / "depth_alignment_panel.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
    report = {
        "camera": camera,
        "image_shape_hw": [height, width],
        "model_dir": str(model_dir),
        "source_root": str(source_root),
        "prediction_is_metric": prediction.is_metric,
        "lidar_coverage": float(result.lidar_validity.mean()),
        "monocular_coverage": float(result.monocular_validity.mean()),
        "fused_coverage": float((result.depth > 0).mean()),
        "fit_count": result.fit_count,
        "inverse_depth_scale": result.scale,
        "inverse_depth_shift": result.shift,
        "holdout": result.holdout,
        "panel_columns": ["rectified_rgb", "lidar", "da3_aligned", "continuous_fused_depth"],
        "fused_depth_role": "continuous_da3_aligned_surface",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_root", type=Path)
    parser.add_argument("--camera", default="CAM_FRONT_NARROW")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/heqing/models/depth-anything-3/DA3Metric-Large"))
    parser.add_argument("--source-root", type=Path, default=Path("/home/heqing/Depth-Anything-3"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.probe_root, args.camera, args.output, args.model_dir, args.source_root, args.device), indent=2))


if __name__ == "__main__":
    main()
