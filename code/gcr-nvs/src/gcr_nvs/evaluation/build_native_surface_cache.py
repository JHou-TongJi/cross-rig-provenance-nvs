"""Build a dense native-resolution LiDAR/RGB surface cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from scipy.spatial import cKDTree

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest
from gcr_nvs.datasets.sparse_geometry_dataset import SparseGeometryDistillationDataset
from gcr_nvs.geometry.appearance import project_surfaces_to_sources
from gcr_nvs.geometry.calibration import load_calibrations
from gcr_nvs.geometry.camera import TargetCamera
from gcr_nvs.geometry.temporal_lidar import EXACT_CURRENT
from gcr_nvs.models.lidar_appearance import LiDARSurfaceAppearanceLift
from gcr_nvs.models.sparse_geometry import SparseGeometryStudent
from gcr_nvs.rendering.sparse_geometry_raymarch import _points_to_query_indices


def _load_rgb(path: Path, width: int, height: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize(
            (width, height), Image.Resampling.LANCZOS,
        ),
        dtype=np.float32,
    ) / 255.0


def _sample_rgb(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return cv2.remap(
        image,
        uv[:, 0].reshape(-1, 1).astype(np.float32),
        uv[:, 1].reshape(-1, 1).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )[:, 0]


def estimate_lidar_normals(
    points: np.ndarray,
    camera_center: np.ndarray,
    neighbors: int = 16,
) -> np.ndarray:
    """Estimate oriented surface normals from the LiDAR neighborhood only."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) < neighbors + 1:
        raise ValueError("normal estimation needs more points than neighbors")
    _, indices = cKDTree(points).query(points, k=neighbors + 1)
    neighborhoods = points[indices[:, 1:]]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(neighbors - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    toward_camera = np.asarray(camera_center, dtype=np.float32)[None] - points
    flip = (normals * toward_camera).sum(axis=1) < 0.0
    normals[flip] *= -1.0
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-6)
    return normals.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry_checkpoint", type=Path)
    parser.add_argument("--teacher-root", type=Path, default=Path("outputs/geometry_teacher"))
    parser.add_argument("--sequence", default="2026-05-22-15-09-17")
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--target-camera", default="CAM_FRONT_NARROW")
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--feature-width", type=int, default=960)
    parser.add_argument("--feature-height", type=int, default=540)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--allow-future-lidar", action="store_true")
    parser.add_argument("--projection-offsets", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/geometry_diagnostics/native_surface_cache.npz"),
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device(args.device)
    sequence_dir = Path(args.sequence)
    record = next(
        row for row in build_sequence_manifest(sequence_dir, compute_quality=False)
        if row.frame_id == args.frame_id
    )
    calibrations = load_calibrations(
        sequence_dir / record.camera_config, Path("camera_intric.yaml"),
    )
    target_calibration = calibrations[args.target_camera]

    teacher_path = args.teacher_root / args.sequence / f"{args.frame_id:06d}.npz"
    with np.load(teacher_path) as payload:
        teacher_points = payload["points"].astype(np.float32)
        source_kind = payload["source_kind"].astype(np.uint8)
        teacher_confidence = payload["confidence"].astype(np.float32)
        time_offset = payload["time_offset_s"].astype(np.float32)
    causal = np.ones(len(teacher_points), dtype=bool)
    if not args.allow_future_lidar:
        causal &= time_offset <= 0.0
    teacher_points = teacher_points[causal]
    source_kind = source_kind[causal]
    teacher_confidence = teacher_confidence[causal]
    time_offset = time_offset[causal]

    target_uv, target_visible, target_depth, _ = project_surfaces_to_sources(
        teacher_points[:, :3], [target_calibration],
    )
    target_visible = target_visible[0]
    points = teacher_points[target_visible]
    source_kind = source_kind[target_visible]
    teacher_confidence = teacher_confidence[target_visible]
    time_offset = time_offset[target_visible]
    scan_time_ms = points[:, 5].copy()
    target_uv = target_uv[0, target_visible]
    target_depth = target_depth[0, target_visible]

    geometry_state = torch.load(
        args.geometry_checkpoint, map_location="cpu", weights_only=False,
    )
    geometry_config = geometry_state.get("config", {})
    include_free = bool(geometry_config.get("include_free_input", False))
    geometry_dataset = SparseGeometryDistillationDataset(
        args.teacher_root,
        args.sequence,
        max_input_points=int(geometry_config.get("max_input_points", 8000)),
        max_positive_queries=int(geometry_config.get("max_positive_queries", 8000)),
        max_negative_queries=int(geometry_config.get("max_negative_queries", 8000)),
        include_free_input=include_free,
    )
    sample_index = next(
        index for index, artifact in enumerate(geometry_dataset.artifacts)
        if int(artifact.stem) == args.frame_id
    )
    sample = geometry_dataset[sample_index]
    geometry_model = SparseGeometryStudent(
        input_dim=7 if include_free else 6,
        query_position_encoding=bool(geometry_config.get("query_position_encoding", False)),
    ).to(device).eval()
    geometry_model.load_state_dict(geometry_state["model"], strict=True)
    point_tensor = torch.from_numpy(points[:, :3]).to(device)
    query_indices, in_grid = _points_to_query_indices(point_tensor, geometry_dataset.grid)
    points = points[in_grid.cpu().numpy()]
    source_kind = source_kind[in_grid.cpu().numpy()]
    teacher_confidence = teacher_confidence[in_grid.cpu().numpy()]
    time_offset = time_offset[in_grid.cpu().numpy()]
    scan_time_ms = scan_time_ms[in_grid.cpu().numpy()]
    target_uv = target_uv[in_grid.cpu().numpy()]
    target_depth = target_depth[in_grid.cpu().numpy()]
    unique_queries, inverse = torch.unique(query_indices, dim=0, return_inverse=True)
    with torch.inference_mode():
        geometry = geometry_model(
            sample["input_features"].to(device),
            sample["input_indices"].to(device),
            tuple(int(value) for value in sample["spatial_shape"].tolist()),
            batch_size=1,
            query_indices=unique_queries,
        )
    query_valid = geometry["query_validity"][inverse, 0] > 0.5
    keep = query_valid.cpu().numpy()
    points = points[keep]
    source_kind = source_kind[keep]
    teacher_confidence = teacher_confidence[keep]
    time_offset = time_offset[keep]
    scan_time_ms = scan_time_ms[keep]
    target_uv = target_uv[keep]
    target_depth = target_depth[keep]
    geometry_features = geometry["surface_features"][inverse][query_valid]
    geometry_occupancy = geometry["occupancy"][inverse, 0][query_valid]
    model_normal = geometry["normal"][inverse][query_valid]
    geometry_normal = torch.from_numpy(
        estimate_lidar_normals(points[:, :3], target_calibration.camera_center),
    ).to(device=device, dtype=torch.float32)
    geometry_confidence = torch.from_numpy(
        np.where(
            source_kind == EXACT_CURRENT,
            1.0,
            teacher_confidence,
        ).astype(np.float32),
    ).to(device=device)
    del geometry_model, geometry
    torch.cuda.empty_cache()

    source_names = tuple(name for name in CAMERA_NAMES if name != args.target_camera)
    feature_images = np.stack([
        _load_rgb(
            sequence_dir / record.cameras[name],
            args.feature_width,
            args.feature_height,
        ).transpose(2, 0, 1)
        for name in source_names
    ])
    rgb_images = np.stack([
        _load_rgb(
            sequence_dir / record.cameras[name],
            args.source_width,
            args.source_height,
        ).transpose(2, 0, 1)
        for name in source_names
    ])
    feature_calibrations = [
        TargetCamera(calibrations[name]).resized(
            args.feature_width, args.feature_height,
        ).calibration
        for name in source_names
    ]
    rgb_calibrations = [
        TargetCamera(calibrations[name]).resized(
            args.source_width, args.source_height,
        ).calibration
        for name in source_names
    ]
    feature_uv, feature_valid, _, source_centers = project_surfaces_to_sources(
        points[:, :3], feature_calibrations,
    )
    rgb_uv, rgb_valid, _, _ = project_surfaces_to_sources(
        points[:, :3], rgb_calibrations,
    )
    applied_offsets = {}
    if args.projection_offsets is not None:
        offset_payload = json.loads(args.projection_offsets.read_text())
        offset_width, offset_height = offset_payload["offset_resolution"]
        for source_index, source_name in enumerate(source_names):
            correction = offset_payload["offsets"].get(source_name, {})
            dx = float(correction.get("dx", 0.0))
            dy = float(correction.get("dy", 0.0))
            feature_uv[source_index, :, 0] += dx * args.feature_width / offset_width
            feature_uv[source_index, :, 1] += dy * args.feature_height / offset_height
            rgb_uv[source_index, :, 0] += dx * args.source_width / offset_width
            rgb_uv[source_index, :, 1] += dy * args.source_height / offset_height
            applied_offsets[source_name] = [dx, dy]
    feature_valid &= rgb_valid
    source_axes = np.stack([
        calibration.external[:3, :3].T @ np.asarray([0.0, 0.0, 1.0])
        for calibration in feature_calibrations
    ]).astype(np.float32)
    target_center = target_calibration.camera_center.astype(np.float32)
    target_axis = (
        target_calibration.external[:3, :3].T @ np.asarray([0.0, 0.0, 1.0])
    ).astype(np.float32)
    appearance_model = LiDARSurfaceAppearanceLift(
        feature_dim=64,
        top_k=args.top_k,
        use_dino=True,
        dino_pretrained=True,
    ).to(device).eval()
    with torch.inference_mode():
        appearance = appearance_model(
            torch.from_numpy(feature_images[None]).to(device),
            torch.from_numpy(points[None, :, :3]).to(device),
            torch.from_numpy(feature_uv[None]).to(device),
            torch.from_numpy(feature_valid[None]).to(device),
            torch.from_numpy(source_centers[None]).to(device),
            target_camera_center=torch.from_numpy(target_center[None]).to(device),
            source_optical_axes=torch.from_numpy(source_axes[None]).to(device),
            target_optical_axis=torch.from_numpy(target_axis[None]).to(device),
            rgb_source_images=torch.from_numpy(rgb_images[None]).to(device),
            rgb_source_uv=torch.from_numpy(rgb_uv[None]).to(device),
        )
    target_rgb = _load_rgb(
        sequence_dir / record.cameras[args.target_camera],
        target_calibration.width,
        target_calibration.height,
    )
    target_point_rgb = _sample_rgb(target_rgb, target_uv)
    output = {
        "surface_points": points[:, :3],
        "point_geometry_features": geometry_features.cpu().numpy(),
        "per_source_features": appearance.per_source_features[0].cpu().numpy(),
        "per_source_rgb": appearance.per_source_rgb[0].cpu().numpy(),
        "per_source_validity": appearance.per_source_validity[0].cpu().numpy(),
        "per_source_scores": appearance.per_source_scores[0].cpu().numpy(),
        "point_appearance_validity": appearance.appearance_validity[0].cpu().numpy(),
        "point_source_count": appearance.source_count[0].cpu().numpy().astype(np.float32),
        "point_rgb": appearance.rgb[0].cpu().numpy(),
        "target_point_rgb": target_point_rgb,
        "point_normal": geometry_normal.cpu().numpy(),
        "point_model_normal": model_normal.cpu().numpy(),
        "point_occupancy": geometry_occupancy.cpu().numpy(),
        "point_geometry_confidence": geometry_confidence.cpu().numpy(),
        "point_query_coverage": np.ones(len(points), dtype=np.float32),
        "point_depth": target_depth,
        "point_source_kind": source_kind,
        "point_teacher_confidence": teacher_confidence,
        "point_time_offset_s": time_offset,
        "point_scan_time_ms": scan_time_ms.astype(np.float32),
        "source_camera_names": np.asarray(source_names),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    report = {
        "sequence": args.sequence,
        "frame_id": args.frame_id,
        "target_camera": args.target_camera,
        "native_resolution": [target_calibration.width, target_calibration.height],
        "causal_lidar_only": not args.allow_future_lidar,
        "surface_count": len(points),
        "exact_surface_count": int((source_kind == EXACT_CURRENT).sum()),
        "appearance_coverage": float(output["point_appearance_validity"].mean()),
        "geometry_feature_dim": int(output["point_geometry_features"].shape[-1]),
        "projection_offsets": applied_offsets,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
