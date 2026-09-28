"""Render target-camera geometry diagnostics from a Sparse Geometry Student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gcr_nvs.datasets.sparse_geometry_dataset import SparseGeometryDistillationDataset
from gcr_nvs.geometry.calibration import load_calibrations
from gcr_nvs.geometry.camera import TargetCamera
from gcr_nvs.models.sparse_geometry import SparseGeometryStudent
from gcr_nvs.rendering.sparse_geometry_raymarch import raymarch_sparse_geometry


def _depth_rgb(depth: np.ndarray, valid: np.ndarray, far_m: float) -> np.ndarray:
    normalized = np.clip(depth / far_m, 0.0, 1.0)
    red = np.clip(1.5 - 2.0 * normalized, 0.0, 1.0)
    green = np.clip(1.0 - np.abs(2.0 * normalized - 1.0), 0.0, 1.0)
    blue = np.clip(2.0 * normalized - 0.5, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    rgb[~valid] = 0.0
    return np.rint(rgb * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--teacher-root", type=Path, default=Path("outputs/geometry_teacher"))
    parser.add_argument("--sequence", default="2026-05-22-15-09-17")
    parser.add_argument("--frame-id", type=int, default=75)
    parser.add_argument("--camera", default="CAM_FRONT_NARROW")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--near-m", type=float, default=1.0)
    parser.add_argument("--far-m", type=float, default=80.0)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--occupied-support-stride", type=int, default=4)
    parser.add_argument("--max-input-points", type=int, default=8000)
    parser.add_argument("--max-positive-queries", type=int, default=8000)
    parser.add_argument("--max-negative-queries", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/geometry_diagnostics/sparse_raymarch"),
    )
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    include_free_input = bool(config.get("include_free_input", False))
    dataset = SparseGeometryDistillationDataset(
        args.teacher_root,
        args.sequence,
        max_input_points=args.max_input_points,
        max_positive_queries=args.max_positive_queries,
        max_negative_queries=args.max_negative_queries,
        include_free_input=include_free_input,
    )
    sample_index = next(
        index for index, artifact in enumerate(dataset.artifacts)
        if int(artifact.stem) == args.frame_id
    )
    sample = dataset[sample_index]
    device = torch.device(args.device)
    model = SparseGeometryStudent(
        input_dim=7 if include_free_input else 6,
        query_position_encoding=bool(config.get("query_position_encoding", False)),
    ).to(device).eval()
    model.load_state_dict(state["model"], strict=True)

    sequence_dir = Path(args.sequence)
    calibration_path = sequence_dir / "Key_frames" / "camera_config" / f"{args.frame_id:06d}.json"
    calibrations = load_calibrations(calibration_path, Path("camera_intric.yaml"))
    target = TargetCamera(calibrations[args.camera])
    with torch.inference_mode():
        result = raymarch_sparse_geometry(
            model,
            sample["input_features"].to(device),
            sample["input_indices"].to(device),
            tuple(int(value) for value in sample["spatial_shape"].tolist()),
            target,
            dataset.grid,
            width=args.width,
            height=args.height,
            near_m=args.near_m,
            far_m=args.far_m,
            sample_count=args.sample_count,
            occupancy_threshold=args.threshold,
            occupied_support_stride=args.occupied_support_stride,
        )

    depth = result.depth.cpu().numpy()
    valid = result.structure_validity.cpu().numpy()
    coverage = result.query_coverage.cpu().numpy()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_prefix.with_suffix(".npz"),
        depth=depth,
        structure_validity=valid,
        query_coverage=coverage,
        occupancy=result.occupancy.cpu().numpy(),
        confidence=result.confidence.cpu().numpy(),
        normal=result.normal.cpu().numpy(),
        surface_features=result.surface_features.cpu().numpy(),
        surface_points=result.surface_points.cpu().numpy(),
        occupied_support=result.occupied_support.cpu().numpy(),
    )
    depth_image = _depth_rgb(depth, valid, args.far_m)
    validity_image = np.repeat((valid[..., None] * 255).astype(np.uint8), 3, axis=2)
    coverage_image = np.repeat(
        np.rint(coverage[..., None] * 255.0).astype(np.uint8), 3, axis=2,
    )
    panel = np.concatenate([depth_image, validity_image, coverage_image], axis=1)
    Image.fromarray(panel).resize(
        (panel.shape[1] * 4, panel.shape[0] * 4), Image.Resampling.NEAREST,
    ).save(args.output_prefix.with_suffix(".png"))

    valid_depth = depth[valid]
    report = {
        "checkpoint": str(args.checkpoint),
        "sequence": args.sequence,
        "frame_id": args.frame_id,
        "camera": args.camera,
        "resolution": [args.width, args.height],
        "structure_coverage": float(valid.mean()),
        "mean_query_coverage": float(coverage.mean()),
        "unknown_structure_ratio": float((~valid).mean()),
        "valid_depth_p10_m": float(np.percentile(valid_depth, 10)) if len(valid_depth) else None,
        "valid_depth_p50_m": float(np.percentile(valid_depth, 50)) if len(valid_depth) else None,
        "valid_depth_p90_m": float(np.percentile(valid_depth, 90)) if len(valid_depth) else None,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
