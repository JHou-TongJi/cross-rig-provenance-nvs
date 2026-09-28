"""Audit DINOv2 + AnyUP semantic reprojection on dense-depth pilot caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gcr_nvs.datasets.manifest import CAMERA_NAMES, appearance_source_cameras
from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.rendering.dense_semantic_reprojection import reproject_dense_maps
from gcr_nvs.models.source_encoder import SourceImageEncoder


def _load(cache_root: Path, sequence: str, frame: int, camera: str):
    root = cache_root / sequence / f"{frame:06d}" / camera
    metadata = json.loads((root / "report.json").read_text())
    if metadata.get("contract", {}).get("fused_depth_role") != "continuous_da3_aligned_surface":
        raise RuntimeError(
            f"legacy depth cache for {camera}; use a continuous DA3-aligned cache"
        )
    image = np.asarray(Image.open(root / "rgb_rectified.jpg").convert("RGB"), dtype=np.float32) / 255.0
    depth = np.load(root / "fused_depth.npy").astype(np.float32)
    confidence = np.load(root / "fused_confidence.npy").astype(np.float32)
    calibration = CameraCalibration(
        camera,
        np.asarray(metadata["intrinsic_newK"], dtype=np.float64),
        np.asarray(metadata["distortion_after_rectification"], dtype=np.float64),
        np.asarray(metadata["external_world_to_camera"], dtype=np.float64),
        int(metadata["output_size"][0]), int(metadata["output_size"][1]),
    )
    return image, depth, confidence, calibration


def run(cache_root: Path, sequence: str, frame: int, target_camera: str, output: Path, device: str, dino_model: str) -> dict:
    names = appearance_source_cameras(target_camera, include_target=True)
    loaded = [_load(cache_root, sequence, frame, name) for name in names]
    images = torch.from_numpy(np.stack([item[0].transpose(2, 0, 1) for item in loaded])).to(device)
    encoder = SourceImageEncoder(
        feature_dim=64,
        use_dino=True,
        dino_model_name=dino_model,
        dino_pretrained=True,
        semantic_upsampling="anyup",
        anyup_checkpoint=Path("outputs/checkpoints/anyup_multi_backbone.pth"),
        anyup_root=Path("third_party/anyup"),
        anyup_q_chunk_size=8192,
    ).to(device).eval()
    with torch.inference_mode():
        features = encoder(images[None])[0].permute(0, 2, 3, 1).float().cpu().numpy()
    maps, depths, confidences, calibrations = [], [], [], []
    for index, item in enumerate(loaded):
        maps.append(features[index])
        depths.append(item[1])
        confidences.append(item[2])
        calibrations.append(item[3])
    target = calibrations[0]
    projected, validity, provenance = reproject_dense_maps(
        maps, depths, calibrations, target,
        source_confidences=confidences,
        exclusive_source_priority=True,
    )
    target_features = features[0]
    target_norm = target_features / np.maximum(np.linalg.norm(target_features, axis=-1, keepdims=True), 1e-6)
    projected_norm = projected / np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), 1e-6)
    cosine = (target_norm * projected_norm).sum(axis=-1)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "source_features.npy", features)
    np.save(output / "target_semantic_features.npy", projected)
    np.save(output / "target_semantic_validity.npy", validity)
    np.save(output / "target_semantic_provenance.npy", provenance)
    report = {
        "sequence": sequence,
        "frame_id": frame,
        "target_camera": target_camera,
        "topology_sources": list(names),
        "dino_model": dino_model,
        "semantic_upsampling": "anyup",
        "feature_shape": list(features.shape),
        "validity": float(validity.mean()),
        "mean_cosine_zero_pose": float(cosine[validity].mean()) if validity.any() else None,
        "p05_cosine_zero_pose": float(np.percentile(cosine[validity], 5)) if validity.any() else None,
        "provenance_counts": {str(int(index)): int((provenance == index).sum()) for index in np.unique(provenance) if index >= 0},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--camera", choices=CAMERA_NAMES, default="CAM_FRONT_NARROW")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dino-model", default="dinov2_vitb14")
    args = parser.parse_args()
    print(json.dumps(run(args.cache_root, args.sequence, args.frame, args.camera, args.output, args.device, args.dino_model), indent=2))


if __name__ == "__main__":
    main()
