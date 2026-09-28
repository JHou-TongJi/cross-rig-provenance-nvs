"""Seven-camera combined-translation stress test for LiDAR-only depth NVS."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest
from gcr_nvs.evaluation.build_native_surface_cache import estimate_lidar_normals
from gcr_nvs.geometry.appearance import project_surfaces_to_sources
from gcr_nvs.geometry.calibration import CameraCalibration, load_calibrations, project_world_points
from gcr_nvs.geometry.camera import TargetCamera
from gcr_nvs.geometry.temporal_lidar import EXACT_CURRENT
from gcr_nvs.models.lidar_depth_completion import LidarDenseDepthCompletion
from gcr_nvs.rendering.dense_lidar_reprojection import (
    DenseLidarGeometry,
    densify_lidar_surfaces,
    rasterize_lidar_range,
    reproject_far_field_rgb,
    reproject_source_rgb,
)


SCENARIOS = (
    ("base", (0.0, 0.0, 0.0)),
    ("forward_left_up", (0.35, 0.30, 0.25)),
    ("forward_right_down", (0.35, -0.30, -0.20)),
    ("backward_left_down", (-0.35, 0.30, -0.20)),
    ("backward_right_up", (-0.35, -0.30, 0.25)),
)


def translated_calibration(
    calibration: CameraCalibration,
    offset_xyz: tuple[float, float, float],
) -> CameraCalibration:
    """Translate a camera center in vehicle axes while preserving its K/D/R."""
    external = calibration.external.copy()
    rotation = external[:3, :3]
    center = calibration.camera_center + np.asarray(offset_xyz, dtype=np.float64)
    external[:3, 3] = -rotation @ center
    return replace(calibration, external=external)


def _load_rgb(path: Path, width: int, height: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize(
            (width, height), Image.Resampling.LANCZOS,
        ),
        dtype=np.float32,
    ) / 255.0


def _view_weights(
    points: np.ndarray,
    source_calibrations: list[CameraCalibration],
    source_validity: np.ndarray,
    target: CameraCalibration,
) -> np.ndarray:
    target_center = target.camera_center.astype(np.float32)
    target_axis = target.external[:3, :3].T @ np.asarray([0.0, 0.0, 1.0])
    target_direction = target_center[None] - points
    target_direction /= np.linalg.norm(target_direction, axis=1, keepdims=True).clip(min=1e-6)
    scores = []
    for source_index, source in enumerate(source_calibrations):
        source_center = source.camera_center.astype(np.float32)
        source_axis = source.external[:3, :3].T @ np.asarray([0.0, 0.0, 1.0])
        source_direction = source_center[None] - points
        source_direction /= np.linalg.norm(source_direction, axis=1, keepdims=True).clip(min=1e-6)
        surface_angle = np.sum(source_direction * target_direction, axis=1)
        optical_angle = float(np.dot(source_axis, target_axis))
        baseline = float(np.linalg.norm(source_center - target_center))
        score = 5.0 * surface_angle + 4.0 * optical_angle - baseline / 5.0
        score = np.where(source_validity[source_index], score, -np.inf)
        scores.append(score.astype(np.float32))
    return np.stack(scores, axis=1)


def _complete_geometry(
    model: LidarDenseDepthCompletion | None,
    points: np.ndarray,
    normals: np.ndarray,
    target: CameraCalibration,
    width: int,
    height: int,
    device: torch.device,
    completion_min_seed_distance_px: float = 0.0,
    completion_confidence_threshold: float = 0.0,
) -> tuple[DenseLidarGeometry, np.ndarray]:
    base = densify_lidar_surfaces(
        points, normals, target, width, height,
        maximum_seed_distance_px=12.0,
        maximum_relative_plane_deviation=0.15,
    )
    if model is None:
        return base, np.ones((height, width), dtype=np.float32)
    measured, measured_valid = rasterize_lidar_range(points, target, width, height)
    rays = TargetCamera(target).ray_map(width, height)
    inputs = [
        torch.from_numpy(value)[None].to(device)
        for value in (
            base.range_m[None],
            base.validity[None],
            base.normal.transpose(2, 0, 1),
            base.seed_distance_px[None],
            rays,
            measured[None],
            measured_valid[None],
        )
    ]
    with torch.inference_mode():
        output = model(*inputs)
    completed_range = output.range_m[0, 0].cpu().numpy()
    confidence = output.confidence[0, 0].cpu().numpy()
    use_completion = (
        (base.seed_distance_px >= completion_min_seed_distance_px)
        & (confidence >= completion_confidence_threshold)
        & base.validity
    )
    completed_range = np.where(
        use_completion, completed_range, base.range_m,
    ).astype(np.float32)
    ray_map = rays.transpose(1, 2, 0)
    surface_points = target.camera_center.astype(np.float32) + ray_map * completed_range[..., None]
    surface_points[~base.validity] = 0.0
    return replace(
        base,
        surface_points=surface_points.astype(np.float32),
        range_m=np.where(base.validity, completed_range, 0.0).astype(np.float32),
    ), confidence


def _render_rgb(
    geometry: DenseLidarGeometry,
    points: np.ndarray,
    source_images: list[np.ndarray],
    source_calibrations: list[CameraCalibration],
    source_validity: np.ndarray,
    target: CameraCalibration,
    hard_source_selection: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = _view_weights(points, source_calibrations, source_validity, target)
    render = reproject_source_rgb(
        geometry, source_images, source_calibrations,
        weights, source_validity.T,
        hard_selection=hard_source_selection,
        spatial_smoothing_sigma=0.0,
    )
    rgb = render.rgb.copy()
    output_validity = render.validity.copy()
    far_field = np.zeros_like(output_validity)
    target_axis = target.external[:3, :3].T @ np.asarray([0.0, 0.0, 1.0])
    order = sorted(range(len(source_calibrations)), key=lambda index: float(
        -(source_calibrations[index].external[:3, :3].T @ np.asarray([0.0, 0.0, 1.0])) @ target_axis
    ))
    for source_index in order:
        environment, valid = reproject_far_field_rgb(
            target, source_images[source_index], source_calibrations[source_index],
            target.width, target.height,
        )
        fill = ~output_validity & valid
        rgb[fill] = environment[fill]
        output_validity[fill] = True
        far_field[fill] = True
    return rgb, render.validity, render.provenance, far_field


def _depth_image(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(depth, dtype=np.uint8)
    normalized[valid] = np.rint(
        np.clip(np.log1p(depth[valid]) / np.log(81.0), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    color = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


def _provenance_image(provenance: np.ndarray) -> np.ndarray:
    colors = np.asarray([
        [230, 80, 70], [80, 180, 90], [70, 130, 230], [230, 190, 60],
        [180, 80, 220], [50, 200, 210], [240, 130, 40],
    ], dtype=np.uint8)
    output = np.zeros((*provenance.shape, 3), dtype=np.uint8)
    valid = provenance >= 0
    output[valid] = colors[provenance[valid]]
    return output


def _fragmentation_metrics(
    rgb: np.ndarray,
    depth: np.ndarray,
    validity: np.ndarray,
    provenance: np.ndarray,
) -> dict[str, float]:
    edge_valid_x = validity[:, 1:] & validity[:, :-1]
    edge_valid_y = validity[1:, :] & validity[:-1, :]
    provenance_jump_x = edge_valid_x & (
        provenance[:, 1:] != provenance[:, :-1]
    )
    provenance_jump_y = edge_valid_y & (
        provenance[1:, :] != provenance[:-1, :]
    )
    depth_difference_x = np.abs(depth[:, 1:] - depth[:, :-1])
    depth_difference_y = np.abs(depth[1:, :] - depth[:-1, :])
    depth_threshold_x = np.maximum(
        0.5, 0.05 * np.minimum(depth[:, 1:], depth[:, :-1]),
    )
    depth_threshold_y = np.maximum(
        0.5, 0.05 * np.minimum(depth[1:, :], depth[:-1, :]),
    )
    depth_jump_x = edge_valid_x & (depth_difference_x > depth_threshold_x)
    depth_jump_y = edge_valid_y & (depth_difference_y > depth_threshold_y)
    rgb_difference_x = np.abs(rgb[:, 1:] - rgb[:, :-1]).mean(axis=2)
    rgb_difference_y = np.abs(rgb[1:, :] - rgb[:-1, :]).mean(axis=2)
    provenance_edges = int(provenance_jump_x.sum() + provenance_jump_y.sum())
    all_edges = int(edge_valid_x.sum() + edge_valid_y.sum())
    rgb_on_provenance = np.concatenate([
        rgb_difference_x[provenance_jump_x],
        rgb_difference_y[provenance_jump_y],
    ])
    return {
        "provenance_neighbor_jump_ratio": provenance_edges / max(all_edges, 1),
        "depth_discontinuity_ratio": float(
            (depth_jump_x.sum() + depth_jump_y.sum()) / max(all_edges, 1)
        ),
        "rgb_l1_at_provenance_boundary": float(
            rgb_on_provenance.mean() if len(rgb_on_provenance) else 0.0
        ),
    }


def _cycle_metrics(
    base_geometry: DenseLidarGeometry,
    shifted_geometry: DenseLidarGeometry,
    shifted_target: CameraCalibration,
    stride: int = 4,
) -> dict[str, float | int]:
    selected = np.zeros_like(base_geometry.validity)
    selected[::stride, ::stride] = base_geometry.validity[::stride, ::stride]
    points = base_geometry.surface_points[selected]
    uv, projected = project_world_points(points, shifted_target)
    points = points[projected]
    uv = uv[projected].astype(np.float32)
    sampled_range = cv2.remap(
        shifted_geometry.range_m,
        uv[:, 0, None], uv[:, 1, None],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )[:, 0]
    sampled_valid = cv2.remap(
        shifted_geometry.validity.astype(np.uint8),
        uv[:, 0, None], uv[:, 1, None],
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )[:, 0].astype(bool)
    expected = np.linalg.norm(points - shifted_target.camera_center, axis=1)
    valid = sampled_valid & (sampled_range > 0.1)
    error = np.abs(sampled_range[valid] - expected[valid])
    return {
        "count": int(valid.sum()),
        "median_range_cycle_error_m": float(np.median(error)) if len(error) else float("nan"),
        "p90_range_cycle_error_m": float(np.percentile(error, 90)) if len(error) else float("nan"),
    }


def _parallax_metrics(
    points: np.ndarray,
    base: CameraCalibration,
    shifted: CameraCalibration,
) -> dict[str, dict[str, float | int]]:
    base_uv, base_valid = project_world_points(points, base)
    shifted_uv, shifted_valid = project_world_points(points, shifted)
    valid = base_valid & shifted_valid
    depth = np.linalg.norm(points - base.camera_center, axis=1)
    displacement = np.linalg.norm(shifted_uv - base_uv, axis=1)
    result = {}
    for name, lower, upper in (
        ("near", 0.0, 20.0), ("mid", 20.0, 50.0), ("far", 50.0, np.inf),
    ):
        selected = valid & (depth >= lower) & (depth < upper)
        result[name] = {
            "count": int(selected.sum()),
            "median_displacement_px": float(np.median(displacement[selected])) if selected.any() else float("nan"),
        }
    return result


def _save_camera_panel(
    root: Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    provenance: np.ndarray,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(rgb.clip(0, 1) * 255).astype(np.uint8)).save(root / "rgb.png")
    Image.fromarray(_depth_image(depth, valid)).save(root / "depth.png")
    Image.fromarray(np.repeat(valid[..., None], 3, axis=2).astype(np.uint8) * 255).save(root / "validity.png")
    Image.fromarray(_provenance_image(provenance)).save(root / "provenance.png")
    panel = Image.new("RGB", (rgb.shape[1] * 4, rgb.shape[0]), "black")
    for index, name in enumerate(("rgb.png", "depth.png", "validity.png", "provenance.png")):
        panel.paste(Image.open(root / name).convert("RGB"), (index * rgb.shape[1], 0))
    panel.save(root / "diagnostic.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--teacher-root", type=Path, default=Path("outputs/geometry_teacher"))
    parser.add_argument("--sequence", default="2026-05-22-15-09-17")
    parser.add_argument("--frame-id", type=int, default=75)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-depth-completion", action="store_true")
    parser.add_argument(
        "--no-depth-completion", action="store_true", help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-log-residual", type=float, default=0.7)
    parser.add_argument("--hard-source-selection", action="store_true")
    parser.add_argument("--completion-min-seed-distance", type=float, default=3.0)
    parser.add_argument("--completion-confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/geometry_diagnostics/lidar_depth_se3_stress"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    model = None
    depth_completion_enabled = (
        args.use_depth_completion and not args.no_depth_completion
    )
    if depth_completion_enabled:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = LidarDenseDepthCompletion(
            maximum_log_residual=args.max_log_residual,
        ).to(device).eval()
        model.load_state_dict(state["model"], strict=True)
    sequence_dir = Path(args.sequence)
    record = next(
        row for row in build_sequence_manifest(sequence_dir, compute_quality=False)
        if row.frame_id == args.frame_id
    )
    native_calibrations = load_calibrations(
        sequence_dir / record.camera_config, Path("camera_intric.yaml"),
    )
    source_calibrations = [
        TargetCamera(native_calibrations[name]).resized(args.width, args.height).calibration
        for name in CAMERA_NAMES
    ]
    source_images = [
        _load_rgb(sequence_dir / record.cameras[name], args.width, args.height)
        for name in CAMERA_NAMES
    ]
    with np.load(args.teacher_root / args.sequence / f"{args.frame_id:06d}.npz") as payload:
        teacher_points = payload["points"].astype(np.float32)
        source_kind = payload["source_kind"].astype(np.uint8)
        time_offset = payload["time_offset_s"].astype(np.float32)
    causal = (source_kind == EXACT_CURRENT) | (time_offset <= 0.0)
    teacher_points = teacher_points[causal, :3]
    _, source_visible, _, _ = project_surfaces_to_sources(
        teacher_points, source_calibrations,
    )
    union_visible = source_visible.any(axis=0)
    points = teacher_points[union_visible]
    source_visible = source_visible[:, union_visible]
    normals = estimate_lidar_normals(points, np.zeros(3, dtype=np.float32))
    tree = cKDTree(points)

    args.output.mkdir(parents=True, exist_ok=True)
    base_geometry = {}
    base_targets = {}
    reports = []
    for scenario, offset in SCENARIOS:
        scenario_rows = {}
        for camera_name, base_source in zip(CAMERA_NAMES, source_calibrations):
            target = translated_calibration(base_source, offset)
            geometry, confidence = _complete_geometry(
                model, points, normals, target, args.width, args.height, device,
                completion_min_seed_distance_px=args.completion_min_seed_distance,
                completion_confidence_threshold=args.completion_confidence_threshold,
            )
            rgb, appearance_valid, provenance, far_field = _render_rgb(
                geometry, points, source_images, source_calibrations,
                source_visible, target,
                hard_source_selection=args.hard_source_selection,
            )
            valid_points = geometry.surface_points[geometry.validity][::8]
            lidar_distance = tree.query(valid_points, k=1, workers=1)[0]
            row = {
                "structure_coverage": float(geometry.validity.mean()),
                "appearance_coverage": float(appearance_valid.mean()),
                "far_field_ratio": float(far_field.mean()),
                "mean_depth_confidence": float(confidence[geometry.validity].mean()),
                "depth_p10_m": float(np.percentile(geometry.range_m[geometry.validity], 10)),
                "depth_p50_m": float(np.percentile(geometry.range_m[geometry.validity], 50)),
                "depth_p90_m": float(np.percentile(geometry.range_m[geometry.validity], 90)),
                "surface_to_lidar_median_m": float(np.median(lidar_distance)),
                "surface_to_lidar_p90_m": float(np.percentile(lidar_distance, 90)),
                "fragmentation": _fragmentation_metrics(
                    rgb, geometry.range_m, geometry.validity, provenance,
                ),
            }
            if scenario == "base":
                base_geometry[camera_name] = geometry
                base_targets[camera_name] = target
            else:
                row["cycle"] = _cycle_metrics(
                    base_geometry[camera_name], geometry, target,
                )
                row["parallax"] = _parallax_metrics(
                    points, base_targets[camera_name], target,
                )
            scenario_rows[camera_name] = row
            camera_root = args.output / scenario / camera_name
            _save_camera_panel(
                camera_root, rgb, geometry.range_m, geometry.validity, provenance,
            )
            np.savez_compressed(
                camera_root / "geometry.npz",
                depth_range_m=geometry.range_m,
                structure_validity=geometry.validity,
                confidence=confidence,
                provenance=provenance,
                far_field_mask=far_field,
            )
        reports.append({
            "scenario": scenario,
            "vehicle_offset_m": list(offset),
            "per_camera": scenario_rows,
            "mean_structure_coverage": float(np.mean([
                row["structure_coverage"] for row in scenario_rows.values()
            ])),
            "mean_surface_to_lidar_p90_m": float(np.mean([
                row["surface_to_lidar_p90_m"] for row in scenario_rows.values()
            ])),
        })

    cell_width, cell_height = 256, 144
    overview = Image.new(
        "RGB", (cell_width * len(CAMERA_NAMES), (cell_height + 24) * len(reports)), "black",
    )
    draw = ImageDraw.Draw(overview)
    for row_index, scenario in enumerate(reports):
        y = row_index * (cell_height + 24)
        draw.text((4, y + 4), f"{scenario['scenario']} xyz={scenario['vehicle_offset_m']}", fill="yellow")
        for column, camera in enumerate(CAMERA_NAMES):
            image = Image.open(args.output / scenario["scenario"] / camera / "rgb.png").convert("RGB")
            image = image.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            overview.paste(image, (column * cell_width, y + 24))
            draw.text((column * cell_width + 3, y + 27), camera.replace("CAM_", ""), fill="yellow", stroke_width=1, stroke_fill="black")
    overview.save(args.output / "rgb_stress_overview.png")

    report = {
        "checkpoint": str(args.checkpoint),
        "sequence": args.sequence,
        "frame_id": args.frame_id,
        "resolution": [args.width, args.height],
        "sources": list(CAMERA_NAMES),
        "target_cameras": list(CAMERA_NAMES),
        "geometry_contract": "causal LiDAR teacher + LiDAR-only dense depth completion",
        "depth_completion_enabled": depth_completion_enabled,
        "max_log_residual": args.max_log_residual,
        "completion_min_seed_distance_px": args.completion_min_seed_distance,
        "completion_confidence_threshold": args.completion_confidence_threshold,
        "appearance_contract": "all seven real RGB sources; no RGB-to-depth path",
        "source_fusion": (
            "hard_argmax" if args.hard_source_selection
            else "surface-consistent_soft_blending"
        ),
        "ground_truth_warning": "shifted target RGB has no real ground truth; use geometry cycle/parallax diagnostics",
        "scenarios": reports,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "scenario_count": len(reports),
        "camera_count": len(CAMERA_NAMES),
        "mean_coverages": {
            row["scenario"]: row["mean_structure_coverage"] for row in reports
        },
    }, indent=2))


if __name__ == "__main__":
    main()
