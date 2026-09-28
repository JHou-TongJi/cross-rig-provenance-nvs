"""Visualize every intermediate of the dense RGB/LiDAR/semantic pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from gcr_nvs.datasets.manifest import CAMERA_NAMES, appearance_source_cameras, build_sequence_manifest
from gcr_nvs.geometry.calibration import CameraCalibration, load_calibrations, rectify_image
from gcr_nvs.rendering.dense_depth_reprojection import reproject_dense_rgb
from gcr_nvs.models.dense_warp_completion import DenseWarpCompletionGenerator


def _font(size: int = 18):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _label(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    canvas = np.asarray(image).copy()
    if canvas.dtype != np.uint8:
        canvas = np.rint(np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 31), (0, 0, 0), -1)
    cv2.putText(canvas, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (8, canvas.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _depth_color(depth: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(depth) & (depth > 0)
    values = depth[valid & np.isfinite(depth) & (depth > 0)]
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not values.size:
        return output
    lo, hi = np.percentile(values, [1.0, 99.0])
    q = (np.log(np.clip(depth, max(lo, 1e-3), max(hi, lo + 1e-3))) - np.log(max(lo, 1e-3))) / max(np.log(max(hi, lo + 1e-3)) - np.log(max(lo, 1e-3)), 1e-6)
    output = cv2.cvtColor(cv2.applyColorMap(np.rint(np.clip(q, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    output[~valid] = (12, 12, 12)
    return output


def _confidence_color(confidence: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
    return cv2.cvtColor(cv2.applyColorMap(np.rint(q * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def _validity(validity: np.ndarray, provenance: np.ndarray | None = None) -> np.ndarray:
    output = np.zeros((*validity.shape, 3), dtype=np.uint8)
    output[validity] = (55, 205, 85)
    output[~validity] = (224, 26, 220)
    if provenance is not None:
        for index, color in enumerate(((40, 190, 80), (40, 150, 230), (230, 150, 40), (180, 80, 220))):
            output[provenance == index] = color
    return output


def _feature_color(features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    # PCA-free stable visualization: first three normalized channels, with
    # feature magnitude shown by saturation.
    channels = features[..., :3]
    lo, hi = np.percentile(channels, [2, 98])
    image = np.clip((channels - lo) / max(float(hi - lo), 1e-6), 0, 1)
    image[~valid] = 0
    return np.rint(image * 255).astype(np.uint8)


def _panel(items: list[tuple[str, str, np.ndarray]], cell_size=(320, 180), columns=4) -> Image.Image:
    width, height = cell_size
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (width * columns, height * rows), (18, 18, 18))
    for index, (stage, io, image) in enumerate(items):
        image = Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 30), fill=(0, 0, 0))
        draw.text((8, 4), stage, fill=(255, 255, 255), font=_font(16))
        draw.text((8, height - 22), io, fill=(220, 220, 220), font=_font(12))
        canvas.paste(image, ((index % columns) * width, (index // columns) * height))
    return canvas


def _flow_diagram(output: Path) -> None:
    width, height = 1800, 760
    canvas = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    title = _font(30)
    body = _font(19)
    small = _font(16)
    draw.text((48, 24), "GCR-NVS Dense Geometry + Semantic RGB Reconstruction", fill=(22, 29, 38), font=title)
    boxes = [
        (50, 150, 285, 330, "Raw RGB +\nBrown calibration", "input: image, K, D"),
        (360, 150, 595, 330, "Rectify image", "output: RGB_rectified, newK"),
        (670, 70, 950, 220, "DA3Metric-Large", "output: dense depth, confidence"),
        (670, 270, 950, 420, "LiDAR geometry", "output: sparse depth, time, mask"),
        (1030, 150, 1320, 330, "Metric alignment", "output: continuous depth + uncertainty"),
        (1400, 150, 1725, 330, "Dense 3D + SE(3) + z-buffer", "output: visibility, RGB warp, semantic warp"),
        (1030, 500, 1320, 680, "DINOv2 + AnyUP", "output: high-resolution semantic features"),
        (1400, 500, 1725, 680, "Controlled completion", "observed: small residual\ndisocclusion: generate"),
    ]
    for x0, y0, x1, y1, label, detail in boxes:
        fill = (226, 238, 250) if "RGB" in label or "DINO" in label else (232, 244, 232) if "LiDAR" in label or "alignment" in label else (248, 236, 220)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=fill, outline=(50, 70, 90), width=3)
        lines = label.split("\n")
        draw.multiline_text((x0 + 16, y0 + 22), "\n".join(lines), fill=(20, 32, 44), font=body, spacing=4)
        draw.multiline_text((x0 + 16, y0 + 100), detail, fill=(65, 76, 88), font=small, spacing=3)
    arrows = [((285, 240), (360, 240)), ((595, 190), (670, 145)), ((595, 280), (670, 350)), ((950, 145), (1030, 210)), ((950, 350), (1030, 270)), ((1320, 240), (1400, 240)), ((595, 240), (1030, 570)), ((1320, 570), (1400, 570)), ((1560, 330), (1560, 500))]
    for (x0, y0), (x1, y1) in arrows:
        draw.line((x0, y0, x1, y1), fill=(55, 70, 85), width=5)
        dx, dy = x1 - x0, y1 - y0
        norm = max((dx * dx + dy * dy) ** 0.5, 1)
        ux, uy = dx / norm, dy / norm
        px, py = -uy, ux
        tip = (x1, y1)
        left = (x1 - 16 * ux + 8 * px, y1 - 16 * uy + 8 * py)
        right = (x1 - 16 * ux - 8 * px, y1 - 16 * uy - 8 * py)
        draw.polygon((tip, left, right), fill=(55, 70, 85))
    draw.text((48, 710), "LiDAR constrains metric geometry and visibility; RGB/DINOv2/AnyUP provide appearance and semantics; generator never repaints reliable observed RGB.", fill=(55, 65, 75), font=small)
    canvas.save(output)


def run(root: Path, cache_root: Path, sequence: str, frame_id: int, target_camera: str, translation: tuple[float, float, float], output: Path, semantic_dir: Path | None = None, completion_checkpoint: Path | None = None, device: str = "cuda") -> dict:
    records = build_sequence_manifest(root / sequence, compute_quality=False)
    record = next(item for item in records if item.frame_id == frame_id)
    raw_calibrations = load_calibrations(root / sequence / record.camera_config, Path("camera_intric.yaml"))
    names = appearance_source_cameras(target_camera, include_target=True)
    sources = []
    for name in names:
        camera_root = cache_root / sequence / f"{frame_id:06d}" / name
        if not (camera_root / "report.json").exists():
            continue
        metadata = json.loads((camera_root / "report.json").read_text())
        sources.append({
            "name": name,
            "rgb": np.asarray(Image.open(camera_root / "rgb_rectified.jpg").convert("RGB")),
            "da3": np.load(camera_root / "da3_depth.npy").astype(np.float32),
            "aligned": np.load(camera_root / "aligned_dense_depth.npy").astype(np.float32),
            "fused": np.load(camera_root / "fused_depth.npy").astype(np.float32),
            "confidence": np.load(camera_root / "fused_confidence.npy").astype(np.float32),
            "lidar": np.load(camera_root / "lidar_sparse_depth.npy").astype(np.float32),
            "lidar_valid": np.load(camera_root / "lidar_validity.npy").astype(bool),
            "calibration": CameraCalibration(name, np.asarray(metadata["intrinsic_newK"]), np.zeros(5), np.asarray(metadata["external_world_to_camera"]), 960, 540),
        })
    if not sources:
        raise RuntimeError(f"missing dense cache for {target_camera}: {names}")
    target_source = next(item for item in sources if item["name"] == target_camera)
    camera_to_world = np.linalg.inv(target_source["calibration"].external)
    camera_to_world[:3, 3] += np.asarray(translation, dtype=np.float64)
    target_calibration = CameraCalibration(target_camera, target_source["calibration"].intrinsic, np.zeros(5), np.linalg.inv(camera_to_world), 960, 540)
    rgbs = [item["rgb"].astype(np.float32) / 255.0 for item in sources]
    depths = [item["fused"] for item in sources]
    confs = [item["confidence"] for item in sources]
    calibrations = [item["calibration"] for item in sources]
    warped, valid, provenance, target_depth = reproject_dense_rgb(
        rgbs, depths, calibrations, target_calibration,
        source_confidences=confs, splat_radius=1, exclusive_source_priority=True, return_depth=True,
    )
    target_raw_path = root / sequence / record.cameras[target_camera]
    with Image.open(target_raw_path) as image:
        raw_rgb = np.asarray(image.convert("RGB"))
    rectified_rgb, _ = rectify_image(raw_rgb, raw_calibrations[target_camera], (960, 540), alpha=0.0)
    sparse = target_source["lidar"]
    da3 = target_source["da3"]
    aligned = target_source["aligned"]
    fused = target_source["fused"]
    confidence = target_source["confidence"]
    items = [
        ("01 Raw RGB", "input: distorted image", raw_rgb),
        ("02 Rectified RGB", "output: undistorted image + newK", rectified_rgb),
        ("03 LiDAR sparse depth", f"input: sparse geometry / {target_source['lidar_valid'].mean()*100:.2f}% valid", _depth_color(sparse, target_source["lidar_valid"])),
        ("04 DA3 depth", "output: monocular dense candidate", _depth_color(da3)),
        ("05 Metric aligned", "output: inverse-depth scale/shift", _depth_color(aligned)),
        ("06 Fused depth", "output: depth used for 3D surface", _depth_color(fused)),
        ("07 Confidence", "output: depth confidence", _confidence_color(confidence)),
        ("08 3D surface / SE(3)", f"output: target z-buffer depth / {translation}m", _depth_color(target_depth, np.isfinite(target_depth))),
        ("09 RGB warp", "output: source RGB sampled at target surface", np.rint(np.clip(warped, 0, 1) * 255).astype(np.uint8)),
        ("10 Visibility / provenance", "output: green=valid, magenta=hole", _validity(valid, provenance)),
    ]
    semantic_valid = None
    if semantic_dir is not None:
        semantic_path = semantic_dir / "target_semantic_features.npy"
        semantic_valid_path = semantic_dir / "target_semantic_validity.npy"
        if semantic_path.exists() and semantic_valid_path.exists():
            semantic = np.load(semantic_path).astype(np.float32)
            semantic_valid = np.load(semantic_valid_path).astype(bool)
            items.append(("11 DINOv2 + AnyUP", "output: projected semantic feature", _feature_color(semantic, semantic_valid)))
    completion_loaded = False
    if completion_checkpoint is not None and semantic_valid is not None:
        state = torch.load(completion_checkpoint, map_location="cpu", weights_only=False)
        config = state.get("config", {})
        train_width = int(config.get("width", 320))
        train_height = int(config.get("height", 180))
        model = DenseWarpCompletionGenerator(semantic_dim=semantic.shape[-1], decoder_dim=int(config.get("decoder_dim", 32))).to(device).eval()
        model.load_state_dict(state["model"], strict=True)
        def resize_tensor(array, channels=False):
            tensor = torch.from_numpy(array.astype(np.float32))
            if not channels:
                tensor = tensor[None, None]
            else:
                tensor = tensor.permute(2, 0, 1)[None]
            return torch.nn.functional.interpolate(tensor.to(device), (train_height, train_width), mode="bilinear", align_corners=False)
        with torch.inference_mode():
            completion = model(
                resize_tensor(warped, True),
                resize_tensor(semantic, True),
                resize_tensor(np.nan_to_num(target_depth, nan=0.0, posinf=0.0)),
                resize_tensor(valid.astype(np.float32)),
                resize_tensor(target_source["confidence"]),
                resize_tensor(np.zeros_like(warped), True),
            )["rgb"][0].permute(1, 2, 0).cpu().numpy()
        completion = cv2.resize(completion, (960, 540), interpolation=cv2.INTER_CUBIC)
        items.append(("12 Completion output", "output: controlled residual / hole fill", np.rint(np.clip(completion, 0, 1) * 255).astype(np.uint8)))
        completion_loaded = True
    pipeline_panel = _panel(items, columns=4)
    output.mkdir(parents=True, exist_ok=True)
    pipeline_panel.save(output / "pipeline_intermediates.jpg", quality=95)
    for index, (stage, _, image) in enumerate(items, 1):
        safe = stage.lower().replace(" ", "_").replace("/", "_").replace("+", "plus").replace(".", "")
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(output / f"{index:02d}_{safe}.png")
    _flow_diagram(output / "pipeline_flow_diagram.png")
    report = {
        "sequence": sequence,
        "frame_id": frame_id,
        "target_camera": target_camera,
        "source_topology": list(names),
        "available_sources": [item["name"] for item in sources],
        "translation_xyz_m": list(translation),
        "resolution": [960, 540],
        "lidar_coverage": float(target_source["lidar_valid"].mean()),
        "fused_depth_coverage": float((fused > 0).mean()),
        "target_surface_coverage": float(valid.mean()),
        "stage_order": [item[0] for item in items],
        "semantic_stage": "DINOv2 + AnyUP is extracted by audit_dense_semantic_pilot and intentionally not mixed into RGB color buffers",
        "completion_stage": "included from checkpoint" if completion_loaded else "not loaded; pass --completion-checkpoint to visualize it",
        "semantic_visualization_loaded": semantic_valid is not None,
        "completion_checkpoint_loaded": completion_loaded,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-id", type=int, default=1)
    parser.add_argument("--camera", choices=CAMERA_NAMES, default="CAM_FRONT_NARROW")
    parser.add_argument("--translate-xyz", type=float, nargs=3, default=(0.0, 0.1, 0.0))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path)
    parser.add_argument("--completion-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.cache_root, args.sequence, args.frame_id, args.camera, tuple(args.translate_xyz), args.output, args.semantic_dir, args.completion_checkpoint, args.device), indent=2))


if __name__ == "__main__":
    main()
