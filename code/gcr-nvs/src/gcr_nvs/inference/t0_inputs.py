"""Inference-side T0 input preparation.

This module intentionally contains no optimizer, loss, or training import.
It loads the frozen geometry teacher/cache contract and produces the tensors
expected by ``T0StructureConstraintNet``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from gcr_nvs.datasets.manifest import CAMERA_NAMES, appearance_source_cameras, build_sequence_manifest
from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.geometry.camera import TargetCamera
from gcr_nvs.geometry.temporal_lidar import EXACT_CURRENT, TEMPORAL_FILLED
from gcr_nvs.models.source_encoder import SourceImageEncoder


@dataclass(frozen=True)
class Ref:
    sequence: str
    frame: int
    camera: str
    calibration_path: str
    teacher_path: str


def _rasterize(points: np.ndarray, values: np.ndarray, calibration, width: int, height: int):
    from gcr_nvs.geometry.calibration import project_world_points
    if len(points) == 0:
        return np.zeros((height, width), np.float32), np.zeros((height, width), bool)
    pixels, valid = project_world_points(points[:, :3], calibration, undistort=False)
    homogeneous = np.c_[points[:, :3], np.ones(len(points), np.float32)]
    depth = (calibration.external @ homogeneous.T).T[:, 2]
    xy = np.zeros((len(points), 2), dtype=np.int64)
    finite = np.isfinite(pixels).all(axis=1)
    xy[finite] = np.rint(pixels[finite]).astype(np.int64)
    valid &= np.isfinite(values) & (depth > 1e-4)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    indices = np.flatnonzero(valid)
    output = np.zeros((height, width), np.float32)
    occupied = np.full((height, width), np.inf, np.float32)
    if len(indices):
        flat = xy[indices, 1] * width + xy[indices, 0]
        order = indices[np.lexsort((depth[indices], flat))]
        ordered_flat = flat[np.lexsort((depth[indices], flat))]
        first = np.r_[True, ordered_flat[1:] != ordered_flat[:-1]]
        winners = order[first]
        winner_flat = ordered_flat[first]
        occupied.reshape(-1)[winner_flat] = depth[winners]
        output.reshape(-1)[winner_flat] = values[winners]
    return output, output > 0.0


def _support_distance(validity: np.ndarray) -> np.ndarray:
    if not validity.any():
        return np.full(validity.shape, 100.0, np.float32)
    return cv2.distanceTransform((~validity).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)


def _rgb_edges(path: Path, width: int, height: int) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (width, height), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    magnitude = np.sqrt(gx * gx + gy * gy)
    return np.stack([np.clip(gx, -1, 1), np.clip(gy, -1, 1), np.clip(magnitude, 0, 1)], axis=0).astype(np.float32)


def load_case(ref: Ref, *, root: Path, cache_root: Path, teacher_root: Path,
              dynamic_root: Path, distortion: Path, width: int, height: int,
              keep_ratio: float = 1.0, seed: int = 0) -> dict[str, np.ndarray]:
    from gcr_nvs.geometry.calibration import load_calibrations
    rng = np.random.default_rng(seed)
    with np.load(ref.teacher_path) as payload:
        points = payload["points"].astype(np.float32)
        source_kind = payload["source_kind"].astype(np.uint8)
        confidence = payload["confidence"].astype(np.float32)
        time_offset = payload["time_offset_s"].astype(np.float32)
    cache = cache_root / ref.sequence / f"{ref.frame:06d}" / ref.camera
    metadata = json.loads((cache / "report.json").read_text())
    ow, oh = (int(value) for value in metadata["output_size"])
    rectified = CameraCalibration(ref.camera, np.asarray(metadata["intrinsic_newK"], np.float64),
        np.zeros_like(np.asarray(metadata["distortion_after_rectification"], np.float64)),
        np.asarray(metadata["external_world_to_camera"], np.float64), ow, oh)
    calibration = TargetCamera(rectified).resized(width, height).calibration
    exact = points[source_kind == EXACT_CURRENT]
    temporal_mask = source_kind == TEMPORAL_FILLED
    temporal = points[temporal_mask]
    temporal_conf_values = confidence[temporal_mask]
    temporal_time_values = np.abs(time_offset[temporal_mask])
    keep = rng.random(len(exact)) < keep_ratio if len(exact) else np.empty(0, bool)
    input_exact = exact[keep]
    ranges = lambda value: np.linalg.norm(value[:, :3] - calibration.camera_center[None], axis=1).astype(np.float32)
    current_range, current_valid = _rasterize(input_exact, ranges(input_exact), calibration, width, height)
    exact_target, exact_valid = _rasterize(exact, ranges(exact), calibration, width, height)
    temporal_range, temporal_valid = _rasterize(temporal, ranges(temporal), calibration, width, height)
    temporal_conf, temporal_conf_valid = _rasterize(temporal, temporal_conf_values, calibration, width, height)
    temporal_conf = np.where(temporal_conf_valid, temporal_conf, 0.0).astype(np.float32)
    temporal_offset, temporal_offset_valid = _rasterize(temporal, temporal_time_values, calibration, width, height)
    temporal_offset = np.where(temporal_offset_valid, temporal_offset, 0.0).astype(np.float32)
    da3_z = cv2.resize(np.load(cache / "aligned_dense_depth.npy").astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    da3_conf = cv2.resize(np.load(cache / "da3_confidence.npy").astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    ray_map = TargetCamera(calibration).ray_map(width, height).astype(np.float32)
    camera_rays = np.einsum("ij,jhw->ihw", calibration.external[:3, :3].astype(np.float32), ray_map)
    da3_range = np.where(da3_z > 0.1, da3_z / np.maximum(camera_rays[2], 0.1), 0.0).astype(np.float32)
    dynamic = np.zeros((height, width), np.float32)
    dynamic_path = dynamic_root / ref.sequence / f"{ref.frame:06d}" / f"{ref.camera}.dynamic.npy"
    if dynamic_path.exists():
        dynamic = cv2.resize(np.load(dynamic_path).astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    target_rgb = cv2.resize(np.asarray(Image.open(cache / "rgb_rectified.jpg").convert("RGB"), np.float32) / 255.0, (width, height), interpolation=cv2.INTER_AREA)
    source_rgb, source_intrinsic, source_external = [], [], []
    for source_camera in appearance_source_cameras(ref.camera, include_target=False):
        source_cache = cache_root / ref.sequence / f"{ref.frame:06d}" / source_camera
        if not (source_cache / "report.json").exists():
            continue
        report = json.loads((source_cache / "report.json").read_text())
        image = cv2.resize(np.asarray(Image.open(source_cache / "rgb_rectified.jpg").convert("RGB"), np.float32) / 255.0, (width, height), interpolation=cv2.INTER_AREA)
        intrinsic = np.asarray(report["intrinsic_newK"], np.float32)
        intrinsic[0] *= width / float(report["output_size"][0]); intrinsic[1] *= height / float(report["output_size"][1])
        source_rgb.append(image.transpose(2, 0, 1).astype(np.float32)); source_intrinsic.append(intrinsic)
        source_external.append(np.asarray(report["external_world_to_camera"], np.float32))
    return {"da3_range": da3_range, "da3_confidence": np.clip(da3_conf, 0, 1), "current_range": current_range,
        "current_valid": current_valid.astype(np.float32), "temporal_range": temporal_range,
        "temporal_valid": temporal_valid.astype(np.float32), "temporal_confidence": temporal_conf,
        "temporal_offset": temporal_offset, "seed_distance": _support_distance(current_valid | temporal_valid),
        "dynamic": dynamic, "rgb_edges": _rgb_edges(cache / "rgb_rectified.jpg", width, height), "rays": ray_map,
        "target_range": exact_target, "rgb_path": str(cache / "rgb_rectified.jpg"),
        "semantic_key": f"{ref.sequence}/{ref.frame:06d}/{ref.camera}", "target_valid": exact_valid,
        "target_rgb": target_rgb.transpose(2, 0, 1).astype(np.float32), "target_external": rectified.external.astype(np.float32),
        "target_center": np.linalg.inv(rectified.external)[:3, 3].astype(np.float32),
        "photometric_source_rgb": np.asarray(source_rgb, np.float32), "photometric_source_intrinsic": np.asarray(source_intrinsic, np.float32),
        "photometric_source_external": np.asarray(source_external, np.float32)}


def to_device(case: dict[str, np.ndarray], device: torch.device):
    names = ("da3_range", "da3_confidence", "current_range", "current_valid", "temporal_range", "temporal_valid", "temporal_confidence", "temporal_offset", "seed_distance", "dynamic")
    scalar = [torch.from_numpy(case[name])[None, None].to(device) for name in names]
    edges = torch.from_numpy(case["rgb_edges"])[None].to(device); rays = torch.from_numpy(case["rays"])[None].to(device)
    target = torch.from_numpy(case["target_range"])[None, None].to(device)
    valid = torch.from_numpy(case["target_valid"])[None, None].to(device).bool()
    return scalar, edges, rays, target, valid


def semantic_features(case: dict[str, np.ndarray], encoder: SourceImageEncoder, device: torch.device, cache_root: Path | None = None) -> torch.Tensor:
    cache_path = cache_root / (case["semantic_key"] + ".npy") if cache_root is not None else None
    if cache_path is not None and cache_path.exists():
        value = np.load(cache_path, mmap_mode="r").astype(np.float32, copy=False)
        if value.ndim == 3 and value.shape[1:] == case["rays"].shape[1:]:
            return torch.from_numpy(value.copy())[None].to(device)
    rgb = cv2.resize(np.asarray(Image.open(case["rgb_path"]).convert("RGB"), np.float32) / 255.0, (case["rays"].shape[2], case["rays"].shape[1]), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(rgb.transpose(2, 0, 1)).to(device)[None]
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        features = encoder(image).float()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True); temp = cache_path.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(temp, features[0].detach().cpu().numpy().astype(np.float16)); os.replace(temp, cache_path)
    return features
