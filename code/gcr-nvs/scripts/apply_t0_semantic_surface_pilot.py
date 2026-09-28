"""Apply semantic target-depth completion to real T0 holes and pull source RGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import torch

from gcr_nvs.datasets.manifest import appearance_source_cameras, build_sequence_manifest
from gcr_nvs.geometry.calibration import CameraCalibration, load_calibrations, rectify_image
from gcr_nvs.geometry.camera import TargetCamera, sparse_depth_from_points
from gcr_nvs.models.dino_anyup_descriptor import FrozenDinoAnyUpDescriptor
from gcr_nvs.models.semantic_surface_completion import SemanticSurfaceCompletionNet
from gcr_nvs.rendering.target_surface_completion import inverse_sample_source_rgb


POSES = {
    "mixed_5_10cm": (0.06, -0.08, 0.05),
    "mixed_10_20cm": (-0.14, 0.12, 0.18),
    "mixed_20_50cm": (0.32, -0.24, 0.42),
}
BLUE = np.asarray([25, 90, 235], np.float32) / 255.0


def _perturb(calibration: CameraCalibration, translation) -> CameraCalibration:
    camera_to_world = np.linalg.inv(calibration.external)
    camera_to_world[:3, 3] += np.asarray(translation, np.float64)
    return CameraCalibration(
        calibration.name, calibration.intrinsic.copy(), calibration.distortion.copy(),
        np.linalg.inv(camera_to_world), calibration.width, calibration.height,
    )


def _nearest(depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = valid & np.isfinite(depth) & (depth > 0.1)
    distance, indices = ndimage.distance_transform_edt(~finite, return_indices=True)
    output = depth.copy()
    output[~finite] = depth[indices[0][~finite], indices[1][~finite]]
    return np.clip(output, 0.5, 250.0).astype(np.float32), distance.astype(np.float32)


def _edges(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.uint8(np.clip(rgb, 0, 1) * 255), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3) / 4.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3) / 4.0
    return np.stack([gx, gy, np.sqrt(gx * gx + gy * gy)], axis=0).astype(np.float32)


def _tensor(array: np.ndarray, device: torch.device, channels: bool = False) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=torch.float32)
    return value[None] if channels else value[None, None]


def _depth_vis(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    value = np.zeros(depth.shape, np.uint8)
    okay = valid & np.isfinite(depth) & (depth > 0.1)
    value[okay] = np.uint8(np.clip((np.log(np.clip(depth[okay], 0.5, 250.0)) - np.log(0.5)) / (np.log(250.0) - np.log(0.5)), 0, 1) * 255)
    rgb = cv2.applyColorMap(255 - value, cv2.COLORMAP_TURBO)[..., ::-1]
    rgb[~valid] = np.uint8(BLUE * 255)
    return rgb


def _semantic_pca(features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Render the actual dense descriptor as a deterministic RGB PCA image."""
    channels, height, width = features.shape
    flat = features.reshape(channels, -1).T.astype(np.float64)
    fit = flat[valid.reshape(-1)]
    if len(fit) > 50000:
        fit = fit[np.linspace(0, len(fit) - 1, 50000, dtype=np.int64)]
    mean = fit.mean(axis=0, keepdims=True)
    covariance = np.cov(fit - mean, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    projected = (flat - mean) @ vectors[:, -3:]
    projected = projected.reshape(height, width, 3)
    for channel in range(3):
        low, high = np.percentile(projected[..., channel][valid], (2.0, 98.0))
        projected[..., channel] = (projected[..., channel] - low) / max(high - low, 1e-6)
    return np.clip(projected, 0.0, 1.0).astype(np.float32)


def _component_stats(mask: np.ndarray) -> dict[str, float | int]:
    labels, count = ndimage.label(mask)
    if count == 0:
        return {"count": 0, "largest_pixels": 0, "largest_fraction": 0.0, "p95_pixels": 0.0}
    areas = np.bincount(labels.reshape(-1))[1:]
    return {
        "count": int(count),
        "largest_pixels": int(areas.max()),
        "largest_fraction": float(areas.max() / mask.size),
        "p95_pixels": float(np.percentile(areas, 95.0)),
    }


def _panel(path: Path, entries: list[tuple[str, np.ndarray]]) -> None:
    thumb = (960, 540); title = 42
    canvas = Image.new("RGB", (thumb[0] * 3, (thumb[1] + title) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    fp = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    font = ImageFont.truetype(str(fp), 24) if fp.exists() else ImageFont.load_default()
    for index, (label, array) in enumerate(entries):
        x = index % 3 * thumb[0]; y = index // 3 * (thumb[1] + title)
        draw.text((x + 10, y + 7), label, fill="black", font=font)
        value = np.asarray(array)
        if value.dtype != np.uint8:
            value = np.uint8(np.clip(value, 0, 1) * 255)
        canvas.paste(Image.fromarray(value).resize(thumb, Image.Resampling.LANCZOS), (x, y + title))
    canvas.save(path, quality=95, subsampling=0)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path)
    parser.add_argument("--sequence", default="2026-05-22-10-25-21")
    parser.add_argument("--frame", type=int, default=73)
    parser.add_argument("--camera", default="CAM_FRONT_NARROW")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    width, height = int(config["width"]), int(config["height"])
    model = SemanticSurfaceCompletionNet(semantic_dim=32, channels=32).to(device).eval()
    model.load_state_dict(state["model"])
    descriptor = FrozenDinoAnyUpDescriptor(
        model_name="dinov2_vitl14", output_channels=32, dino_input_size=(364, 644),
        anyup_checkpoint=project / "outputs/checkpoints/anyup_multi_backbone.pth",
        anyup_root=project / "third_party/anyup", q_chunk_size=4096,
    ).to(device).eval()
    record = next(row for row in build_sequence_manifest(args.bundle / "datasets" / args.sequence, compute_quality=False) if row.frame_id == args.frame)
    raw_cals = load_calibrations(args.bundle / "datasets" / args.sequence / record.camera_config, args.bundle / "camera_intric.yaml")
    topology = appearance_source_cameras(args.camera, include_target=True)
    source_rgbs, source_depths, source_cals = [], [], []
    for name in topology:
        raw = np.asarray(Image.open(args.bundle / "datasets" / args.sequence / record.cameras[name]).convert("RGB"))
        rgb, calibration = rectify_image(raw, raw_cals[name], output_size=(3840, 2160), alpha=0.0)
        source_rgbs.append(rgb.astype(np.float32) / 255.0)
        source_cals.append(calibration)
        source_depths.append(np.load(args.audit_root / "source_sidecars" / name / "depth.npy").astype(np.float32))
    with np.load(args.teacher_root / args.sequence / f"{args.frame:06d}.npz") as payload:
        teacher_points = payload["points"].astype(np.float32)
    reports = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for pose, translation in POSES.items():
        folder = args.output / pose / args.camera; folder.mkdir(parents=True, exist_ok=True)
        input_root = args.audit_root / pose / args.camera
        t0 = np.asarray(Image.open(input_root / "t0_highres_blue.png").convert("RGB"), np.float32) / 255.0
        valid_hd = np.load(input_root / "validity.npy").astype(bool)
        depth_hd = np.load(input_root / "depth.npy").astype(np.float32)
        target_hd = _perturb(source_cals[0], translation)
        target_low = TargetCamera(target_hd).resized(width, height).calibration
        hole_fraction = cv2.resize((~valid_hd).astype(np.float32), (width, height), cv2.INTER_AREA)
        hole = hole_fraction >= 0.03
        valid = ~hole
        depth_low = cv2.resize(depth_hd, (width, height), cv2.INTER_NEAREST)
        depth_low[~valid] = np.nan
        base, distance = _nearest(depth_low, valid)
        rgb_low = cv2.resize(t0, (width, height), interpolation=cv2.INTER_AREA)
        rgb_low[hole] = BLUE
        lidar_depth, lidar_valid, _ = sparse_depth_from_points(teacher_points, target_low)
        lidar_condition = np.where(lidar_valid, lidar_depth, base).astype(np.float32)
        image = _tensor(rgb_low.transpose(2, 0, 1), device, channels=True)
        with torch.inference_mode():
            semantic = descriptor(image, (height, width))
            output = model(
                _tensor(base, device), _tensor(valid.astype(np.float32), device),
                _tensor(lidar_condition, device), _tensor(lidar_valid.astype(np.float32), device),
                _tensor(distance, device),
                _tensor(TargetCamera(target_low).ray_map(width, height), device, channels=True),
                _tensor(_edges(rgb_low), device, channels=True), semantic,
            )
        predicted_low = output.depth_m[0, 0].cpu().numpy()
        confidence_low = output.confidence[0, 0].cpu().numpy()
        semantic_low = semantic[0].cpu().numpy()
        semantic_vis = _semantic_pca(semantic_low, valid)
        predicted_hd = cv2.resize(predicted_low, (3840, 2160), interpolation=cv2.INTER_LINEAR)
        confidence_hd = cv2.resize(confidence_low, (3840, 2160), interpolation=cv2.INTER_LINEAR)
        candidate = (~valid_hd) & (confidence_hd >= 0.35)
        completed_depth = depth_hd.copy()
        completed_depth[candidate] = predicted_hd[candidate]
        inverse = inverse_sample_source_rgb(
            completed_depth, target_hd, source_rgbs, source_depths, source_cals,
            candidate, maximum_log_depth_error=0.04,
        )
        recovered = t0.copy()
        recovered[inverse.validity] = inverse.rgb[inverse.validity]
        remaining = (~valid_hd) & ~inverse.validity
        recovered[remaining] = BLUE
        Image.fromarray(np.uint8(np.clip(recovered, 0, 1) * 255)).save(folder / "semantic_surface_rgb_blue.png")
        np.save(folder / "completed_target_depth.npy", completed_depth.astype(np.float32))
        np.save(folder / "semantic_surface_confidence.npy", confidence_hd.astype(np.float32))
        np.save(folder / "recovered_rgb_mask.npy", inverse.validity)
        np.save(folder / "source_log_depth_error.npy", inverse.log_depth_error.astype(np.float32))
        Image.fromarray(np.uint8(semantic_vis * 255)).save(folder / "dino_anyup_semantic_pca.png")
        _panel(folder / "semantic_surface_pipeline.jpg", [
            ("昨天 T0 蓝洞", t0),
            ("T0 目标深度（空洞）", _depth_vis(depth_hd, valid_hd)),
            ("DINOv2-L + AnyUP 特征 PCA", cv2.resize(semantic_vis, (3840, 2160))),
            ("语义补全后目标深度", _depth_vis(completed_depth, valid_hd | candidate)),
            ("通过源深度验证的 RGB", np.repeat(inverse.validity[..., None], 3, axis=2).astype(np.float32)),
            ("只取真实源色后的结果", recovered),
        ])
        repair_coverage = None
        if args.repair_root is not None:
            repair_path = args.repair_root / pose / args.camera / "t0_surface_2px_repaired_blue.png"
            repair_valid_path = args.repair_root / pose / args.camera / "validity_after_2px_surface_repair.npy"
            if repair_path.exists():
                repaired = np.asarray(Image.open(repair_path).convert("RGB"), np.float32) / 255.0
                _panel(folder / "t0_vs_2px_vs_semantic_comparison.jpg", [
                    ("昨天 T0（真实空洞）", t0),
                    ("旧 2px 最近深度修补", repaired),
                    ("新语义目标深度补全", recovered),
                    ("T0 目标深度", _depth_vis(depth_hd, valid_hd)),
                    ("语义特征 PCA", cv2.resize(semantic_vis, (3840, 2160))),
                    ("新补全目标深度", _depth_vis(completed_depth, valid_hd | candidate)),
                ])
            if repair_valid_path.exists():
                repair_coverage = float(np.load(repair_valid_path, mmap_mode="r").mean())
        accepted_errors = inverse.log_depth_error[inverse.validity]
        boundary = (~valid_hd) & (ndimage.distance_transform_edt(~valid_hd) <= 8.0)
        reports[pose] = {
            "original_coverage": float(valid_hd.mean()),
            "depth_candidate_fraction_of_hole": float(candidate.sum() / max((~valid_hd).sum(), 1)),
            "rgb_recovered_fraction_of_hole": float(inverse.validity.sum() / max((~valid_hd).sum(), 1)),
            "final_coverage": float((valid_hd | inverse.validity).mean()),
            "observed_rgb_modified": int(np.any(recovered[valid_hd] != t0[valid_hd], axis=1).sum()),
            "boundary_8px_rgb_recovered_fraction": float(
                (inverse.validity & boundary).sum() / max(boundary.sum(), 1)
            ),
            "source_log_depth_error_mean": float(accepted_errors.mean()) if accepted_errors.size else None,
            "source_log_depth_error_p95": float(np.percentile(accepted_errors, 95.0)) if accepted_errors.size else None,
            "remaining_hole_components": _component_stats(remaining),
            "two_px_repair_coverage": repair_coverage,
        }
    (args.output / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(args.output), "reports": reports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
