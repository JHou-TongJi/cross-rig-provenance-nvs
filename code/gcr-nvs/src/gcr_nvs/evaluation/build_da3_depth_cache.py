"""Build rectified RGB + DA3/LiDAR depth caches for selected frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest
from gcr_nvs.geometry.calibration import load_calibrations, rectify_image
from gcr_nvs.geometry.camera import sparse_depth_from_points
from gcr_nvs.geometry.pcd import read_pcd
from gcr_nvs.geometry.dense_depth_alignment import align_monocular_depth
from gcr_nvs.models.monocular_depth import DepthAnything3Estimator


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path, quality=95)


def build_frame(
    root: Path,
    sequence: str,
    frame_id: int,
    cameras: tuple[str, ...],
    output_root: Path,
    distortion: Path,
    model_dir: Path,
    source_root: Path,
    device: str,
    output_size: tuple[int, int],
    overwrite: bool = False,
    estimator: DepthAnything3Estimator | None = None,
) -> dict:
    unknown = sorted(set(cameras) - set(CAMERA_NAMES))
    if unknown:
        raise ValueError(f"unknown cameras: {unknown}")
    sequence_dir = root / sequence
    records = build_sequence_manifest(sequence_dir, compute_quality=False)
    record = next((item for item in records if item.frame_id == frame_id), None)
    if record is None:
        raise ValueError(f"frame {frame_id} does not exist in {sequence}")
    calibrations = load_calibrations(sequence_dir / record.camera_config, distortion)
    points = read_pcd(sequence_dir / record.lidar, fields=("x", "y", "z"))
    width, height = map(int, output_size)
    frame_root = output_root / sequence / f"{frame_id:06d}"
    estimator = estimator or DepthAnything3Estimator(
        model_dir=model_dir, source_root=source_root, device=device,
    )
    rows = []
    for camera in cameras:
        target = frame_root / camera
        report_path = target / "report.json"
        if report_path.exists() and not overwrite:
            rows.append(json.loads(report_path.read_text()))
            continue
        target.mkdir(parents=True, exist_ok=True)
        raw_path = sequence_dir / record.cameras[camera]
        with Image.open(raw_path) as image:
            raw_rgb = np.asarray(image.convert("RGB"))
        rectified_rgb, rectified_calibration = rectify_image(
            raw_rgb,
            calibrations[camera],
            output_size=(width, height),
            alpha=0.0,
        )
        rgb_path = target / "rgb_rectified.jpg"
        _save_rgb(rgb_path, rectified_rgb)
        sparse_depth, sparse_validity, _ = sparse_depth_from_points(
            points,
            rectified_calibration,
            (width, height),
        )
        prediction = estimator.predict(
            rgb_path,
            rectified_calibration.intrinsic.astype(np.float32),
            output_size=(width, height),
        )
        alignment = align_monocular_depth(
            prediction.depth,
            sparse_depth,
            prediction.confidence,
            min_fit_points=24,
        )
        np.save(target / "da3_depth.npy", prediction.depth)
        np.save(target / "da3_confidence.npy", prediction.confidence)
        np.save(target / "lidar_sparse_depth.npy", sparse_depth)
        np.save(target / "lidar_validity.npy", sparse_validity)
        np.save(target / "aligned_dense_depth.npy", alignment.aligned_monocular_depth)
        np.save(target / "fused_depth.npy", alignment.depth)
        np.save(target / "fused_confidence.npy", alignment.confidence)
        np.save(target / "lidar_anchor_depth.npy", alignment.lidar_anchor_depth)
        metadata = {
            "sequence": sequence,
            "frame_id": frame_id,
            "camera": camera,
            "rgb_rectified": str(rgb_path),
            "raw_rgb": str(raw_path),
            "source_image_size": [int(raw_rgb.shape[1]), int(raw_rgb.shape[0])],
            "output_size": [width, height],
            "intrinsic_newK": rectified_calibration.intrinsic.tolist(),
            "external_world_to_camera": rectified_calibration.external.tolist(),
            "distortion_after_rectification": rectified_calibration.distortion.tolist(),
            "model_dir": str(model_dir),
            "model_source_root": str(source_root),
            "prediction_is_metric": prediction.is_metric,
            "lidar_coverage": float(sparse_validity.mean()),
            "fused_coverage": float((alignment.depth > 0).mean()),
            "fit_count": alignment.fit_count,
            "inverse_depth_scale": alignment.scale,
            "inverse_depth_shift": alignment.shift,
            "holdout": alignment.holdout,
            "contract": {
                "rgb_color_source": "rectified_source_rgb",
                "lidar_direct_rgb_override": False,
                "lidar_role": ["metric_scale", "depth_constraint", "visibility"],
                "monocular_role": "dense_depth_candidate",
                "fused_depth_role": "continuous_da3_aligned_surface",
                "lidar_anchor_depth": "separate_visibility_only",
            },
        }
        report_path.write_text(json.dumps(metadata, indent=2))
        rows.append(metadata)
    manifest = {
        "sequence": sequence,
        "frame_id": frame_id,
        "cameras": list(cameras),
        "output_size": [width, height],
        "rows": rows,
    }
    (frame_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--camera", action="append", dest="cameras")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/da3_depth_cache"))
    parser.add_argument("--distortion", type=Path, default=Path("camera_intric.yaml"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/heqing/models/depth-anything-3/DA3Metric-Large"))
    parser.add_argument("--source-root", type=Path, default=Path("/home/heqing/Depth-Anything-3"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cameras = tuple(args.cameras or ("CAM_FRONT_NARROW", "CAM_FRONT_WIDE", "CAM_BACK_LEFT"))
    manifest = build_frame(
        root=args.root,
        sequence=args.sequence,
        frame_id=args.frame_id,
        cameras=cameras,
        output_root=args.output_root,
        distortion=args.distortion,
        model_dir=args.model_dir,
        source_root=args.source_root,
        device=args.device,
        output_size=(args.width, args.height),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
