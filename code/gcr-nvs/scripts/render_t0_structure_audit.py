"""Render original T0 and T0+ learned structure on mixed SE(3) translations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from gcr_nvs.datasets.manifest import CAMERA_NAMES, appearance_source_cameras
from gcr_nvs.models.source_encoder import SourceImageEncoder
from gcr_nvs.models.t0_structure_constraint import T0StructureConstraintNet
from gcr_nvs.inference.t0_inputs import Ref, load_case, semantic_features, to_device
from gcr_nvs.rendering.dense_depth_reprojection import reproject_dense_rgb


POSES = {
    "zero": (0.0, 0.0, 0.0),
    "mixed_5_10cm": (0.06, -0.08, 0.05),
    "mixed_10_20cm": (-0.14, 0.12, 0.18),
    "mixed_20_50cm": (0.32, -0.24, 0.42),
}


def _cache_item(cache_root: Path, sequence: str, frame: int, camera: str):
    root = cache_root / sequence / f"{frame:06d}" / camera
    report = json.loads((root / "report.json").read_text())
    rgb = np.asarray(Image.open(root / "rgb_rectified.jpg").convert("RGB"), np.float32) / 255.0
    depth = np.load(root / "aligned_dense_depth.npy").astype(np.float32)
    confidence = np.load(root / "da3_confidence.npy").astype(np.float32)
    from gcr_nvs.geometry.calibration import CameraCalibration
    calibration = CameraCalibration(
        camera, np.asarray(report["intrinsic_newK"], np.float64),
        np.zeros_like(np.asarray(report["distortion_after_rectification"], np.float64)),
        np.asarray(report["external_world_to_camera"], np.float64),
        int(report["output_size"][0]), int(report["output_size"][1]),
    )
    return rgb, depth, confidence, calibration


def _perturb(calibration, translation):
    camera_to_world = np.linalg.inv(calibration.external)
    camera_to_world[:3, 3] += np.asarray(translation, np.float64)
    from gcr_nvs.geometry.calibration import CameraCalibration
    return CameraCalibration(
        calibration.name, calibration.intrinsic.copy(), calibration.distortion.copy(),
        np.linalg.inv(camera_to_world), calibration.width, calibration.height,
    )


def _save_image(array: np.ndarray, path: Path):
    Image.fromarray(np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def _panel(images: list[tuple[str, np.ndarray]], width: int = 480, height: int = 270) -> Image.Image:
    canvas = Image.new("RGB", (width * len(images), height + 28), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        resized = cv2.resize(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8), (width, height))
        canvas.paste(Image.fromarray(resized), (index * width, 28))
        draw.text((index * width + 8, 7), label, fill=(20, 20, 20))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    bundle = Path(__file__).resolve().parents[1]
    parser.add_argument("--sequence", default="2026-05-22-10-25-21")
    parser.add_argument("--frame", type=int, default=73)
    parser.add_argument("--cache-root", type=Path, default=bundle / "data/da3_depth_cache_dense_train_v1")
    parser.add_argument("--teacher-root", type=Path, default=bundle / "data/geometry_teacher_full_v1")
    parser.add_argument("--semantic-cache", type=Path, default=bundle / "data/t0_semantic_cache_v1")
    parser.add_argument("--root", type=Path, default=bundle / "datasets")
    parser.add_argument("--dynamic-root", type=Path, default=bundle / "data/dynamic_masks_full_v1")
    parser.add_argument("--distortion", type=Path, default=bundle / "camera_intric.yaml")
    parser.add_argument("--checkpoint", type=Path, default=bundle / "checkpoints/t0_structure_rectified_v2_dinob14_320x180_c48_best.pt")
    parser.add_argument("--output", type=Path, default=bundle / "runs/t0_structure_audit_rectified_v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anyup-checkpoint", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823/outputs/checkpoints/anyup_multi_backbone.pth"))
    parser.add_argument("--anyup-root", type=Path, default=bundle / "third_party/anyup")
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    model = T0StructureConstraintNet(int(config.get("channels", 48)), float(config.get("maximum_log_residual", 0.22))).to(device).eval()
    model.load_state_dict(state["model"])
    dino_model = str(config.get("dino_model", "dinov2_vitb14"))
    dino_width = int(config.get("dino_input_width", 448))
    dino_height = int(config.get("dino_input_height", 252))
    encoder = SourceImageEncoder(
        feature_dim=32, use_dino=True, dino_model_name=dino_model, dino_pretrained=True,
        dino_input_size=(dino_height, dino_width), semantic_upsampling="anyup",
        anyup_checkpoint=args.anyup_checkpoint,
        anyup_root=args.anyup_root, anyup_q_chunk_size=4096,
    ).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    source = {}
    for camera in CAMERA_NAMES:
        source[camera] = _cache_item(args.cache_root, args.sequence, args.frame, camera)
    adapted = {}
    for camera in CAMERA_NAMES:
        rgb, _, confidence, calibration = source[camera]
        ref = Ref(args.sequence, args.frame, camera, "", str(args.teacher_root / args.sequence / f"{args.frame:06d}.npz"))
        case = load_case(ref, root=args.root, cache_root=args.cache_root,
                         teacher_root=args.teacher_root, dynamic_root=args.dynamic_root,
                         distortion=args.distortion, width=int(config.get("width", 320)),
                         height=int(config.get("height", 180)), keep_ratio=0.58,
                         seed=20260824 + CAMERA_NAMES.index(camera))
        scalar, edges, rays, _, _ = to_device(case, device)
        semantic = semantic_features(case, encoder, device, args.semantic_cache)
        with torch.inference_mode():
            depth_range = model(*scalar, edges, rays, semantic).depth_range_m[0, 0].cpu().numpy()
        # The network predicts camera-ray range.  Convert back to rectified
        # camera z with the same world-to-camera rotation used during train.
        camera_rays = np.einsum("ij,jhw->ihw", calibration.external[:3, :3].astype(np.float32), case["rays"])
        ray_z = np.maximum(camera_rays[2], 1e-4)
        adapted_depth = cv2.resize(depth_range * ray_z, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        adapted[camera] = (rgb, adapted_depth.astype(np.float32), confidence, calibration)

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"sequence": args.sequence, "frame": args.frame, "checkpoint": str(args.checkpoint), "poses": {}}
    for pose_name, translation in POSES.items():
        pose_root = args.output / pose_name
        pose_root.mkdir(exist_ok=True)
        rows = []
        for target_camera in CAMERA_NAMES:
            names = appearance_source_cameras(target_camera, include_target=True)
            base_items = [source[name] for name in names]
            adapted_items = [adapted[name] for name in names]
            target = _perturb(source[target_camera][3], translation)
            base_rgb, base_valid, base_prov = reproject_dense_rgb(
                [x[0] for x in base_items], [x[1] for x in base_items], [x[3] for x in base_items], target,
                source_confidences=[x[2] for x in base_items], splat_radius=0, exclusive_source_priority=True,
            )
            new_rgb, new_valid, new_prov = reproject_dense_rgb(
                [x[0] for x in adapted_items], [x[1] for x in adapted_items], [x[3] for x in adapted_items], target,
                source_confidences=[x[2] for x in adapted_items], splat_radius=0, exclusive_source_priority=True,
            )
            target_rgb = source[target_camera][0]
            folder = pose_root / target_camera; folder.mkdir(exist_ok=True)
            _save_image(target_rgb, folder / "source_rgb.png")
            _save_image(base_rgb * base_valid[..., None], folder / "t0_original.png")
            _save_image(new_rgb * new_valid[..., None], folder / "t0_structure.png")
            _save_image(target_rgb, folder / "current_camera_reference.png")
            _save_image(base_valid[..., None].repeat(3, axis=2), folder / "t0_validity.png")
            _save_image(new_valid[..., None].repeat(3, axis=2), folder / "t0_structure_validity.png")
            _panel([
                ("source", target_rgb), ("T0", base_rgb * base_valid[..., None]),
                ("T0+structure", new_rgb * new_valid[..., None]),
                ("validity", new_valid[..., None].repeat(3, axis=2)),
            ]).save(folder / "audit_panel.jpg", quality=94)
            rows.append({"camera": target_camera, "base_coverage": float(base_valid.mean()), "structure_coverage": float(new_valid.mean()), "validity_delta": float(new_valid.mean() - base_valid.mean()), "source_topology": list(names)})
        (pose_root / "summary.json").write_text(json.dumps(rows, indent=2))
        summary["poses"][pose_name] = rows
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output": str(args.output), "poses": list(POSES), "cameras": len(CAMERA_NAMES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
