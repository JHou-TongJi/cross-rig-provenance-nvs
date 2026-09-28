"""Sequence-disjoint training data for semantic T0 residual completion."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import yaml

from gcr_nvs.datasets.manifest import CAMERA_NAMES, FrameRecord, build_sequence_manifest
from gcr_nvs.geometry.calibration import load_calibrations, rectify_image


@dataclass(frozen=True)
class CompletionReference:
    sequence: str
    record: FrameRecord
    camera: str


def completion_references(
    dataset_root: Path,
    split_manifest: Path,
    split: str,
) -> list[CompletionReference]:
    payload = yaml.safe_load(split_manifest.read_text(encoding="utf-8"))
    sequences = tuple(payload[split])
    references = []
    for sequence in sequences:
        for record in build_sequence_manifest(dataset_root / sequence, compute_quality=False):
            references.extend(CompletionReference(sequence, record, camera) for camera in CAMERA_NAMES)
    return references


def evenly_subsample_references(
    references: list[CompletionReference], maximum: int,
) -> list[CompletionReference]:
    """Keep a deterministic sample spread across all split sequences."""
    if maximum <= 0 or maximum >= len(references):
        return references
    indices = np.linspace(0, len(references) - 1, maximum, dtype=np.int64)
    return [references[int(index)] for index in indices]


def load_t0_mask_bank(audit_root: Path, output_size: tuple[int, int]) -> dict[str, list[np.ndarray]]:
    width, height = output_size
    bank = {camera: [] for camera in CAMERA_NAMES}
    for path in sorted(audit_root.glob("*/CAM_*/validity.npy")):
        camera = path.parent.name
        validity = np.load(path, mmap_mode="r").astype(np.uint8)
        hole = cv2.resize(1 - validity, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        bank[camera].append(hole)
    if any(not masks for masks in bank.values()):
        raise RuntimeError("T0 mask bank must contain every camera")
    return bank


class T0SemanticCompletionDataset(Dataset):
    """Real rectified RGB targets with exact T0-shaped synthetic holes."""

    def __init__(
        self,
        references: list[CompletionReference],
        *,
        dataset_root: Path,
        distortion_config: Path,
        rgb_cache_root: Path,
        token_cache_root: Path,
        audit_root: Path,
        output_size: tuple[int, int] = (512, 288),
        seed: int = 20260824,
    ) -> None:
        self.references = references
        self.dataset_root = dataset_root
        self.distortion_config = distortion_config
        self.rgb_cache_root = rgb_cache_root
        self.token_cache_root = token_cache_root
        self.output_size = tuple(int(value) for value in output_size)
        self.seed = int(seed)
        self.mask_bank = load_t0_mask_bank(audit_root, self.output_size)
        self._calibrations: dict[tuple[str, int], dict] = {}

    def __len__(self) -> int:
        return len(self.references)

    def _rgb_cache_path(self, reference: CompletionReference) -> Path:
        return self.rgb_cache_root / reference.sequence / f"{reference.record.frame_id:06d}" / f"{reference.camera}.jpg"

    def _token_cache_path(self, reference: CompletionReference) -> Path:
        return self.token_cache_root / reference.sequence / f"{reference.record.frame_id:06d}" / f"{reference.camera}.npy"

    def _rectified_rgb(self, reference: CompletionReference) -> np.ndarray:
        path = self._rgb_cache_path(reference)
        if path.exists():
            return np.asarray(Image.open(path).convert("RGB"))
        sequence_root = self.dataset_root / reference.sequence
        key = (reference.sequence, reference.record.frame_id)
        calibrations = self._calibrations.get(key)
        if calibrations is None:
            calibrations = load_calibrations(
                sequence_root / reference.record.camera_config, self.distortion_config,
            )
            self._calibrations[key] = calibrations
        raw = np.asarray(Image.open(sequence_root / reference.record.cameras[reference.camera]).convert("RGB"))
        rectified, _ = rectify_image(
            raw, calibrations[reference.camera], output_size=self.output_size, alpha=0.0,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp.jpg")
        Image.fromarray(rectified).save(temporary, quality=95, subsampling=0)
        os.replace(temporary, path)
        return rectified

    @staticmethod
    def _retrieval_proxy(rgb: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        height, width = rgb.shape[:2]
        dx = float(rng.uniform(-0.035, 0.035) * width)
        dy = float(rng.uniform(-0.035, 0.035) * height)
        angle = float(rng.uniform(-1.2, 1.2))
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        matrix[:, 2] += (dx, dy)
        proxy = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
        gain = float(rng.uniform(0.90, 1.10))
        bias = float(rng.uniform(-8.0, 8.0))
        proxy = np.clip(proxy.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        confidence = np.full((height, width), float(rng.uniform(0.65, 0.95)), np.float32)
        return proxy, confidence

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        reference = self.references[index]
        rng = np.random.default_rng(self.seed + index * 104729)
        rgb = self._rectified_rgb(reference)
        hole = self.mask_bank[reference.camera][int(rng.integers(len(self.mask_bank[reference.camera])))].copy()
        if rng.random() < 0.5:
            # Vary topology while preserving the camera-specific T0 outline.
            kernel = np.ones((int(rng.choice([3, 5, 7])),) * 2, np.uint8)
            hole = cv2.dilate(hole, kernel, iterations=1)
        retrieved, retrieval_confidence = self._retrieval_proxy(rgb, rng)
        # Training follows inference order: source retrieval first, generation
        # only on the residual. Small splat gaps nearest reliable T0 evidence
        # are the most likely to be recoverable from a topology source.
        distance = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
        hole_distances = distance[hole > 0.5]
        recoverable = np.zeros_like(hole, dtype=bool)
        if hole_distances.size:
            recovery_fraction = float(rng.uniform(0.20, 0.42))
            threshold = float(np.quantile(hole_distances, recovery_fraction))
            recoverable = (hole > 0.5) & (distance <= threshold)
            # Reject a subset to mimic cycle/geometry consistency filtering.
            recoverable &= rng.random(hole.shape) > 0.18
        residual_hole = (hole > 0.5) & ~recoverable
        source_prefill = np.clip(
            rgb.astype(np.float32) * float(rng.uniform(0.97, 1.03))
            + float(rng.uniform(-2.0, 2.0)), 0, 255,
        ).astype(np.uint8)
        t0_prefilled = rgb.copy()
        t0_prefilled[hole > 0.5] = 0
        t0_prefilled[recoverable] = source_prefill[recoverable]

        cache = self.rgb_cache_root.parent / "da3_depth_cache_dense_train_v1" / reference.sequence / f"{reference.record.frame_id:06d}" / reference.camera
        if cache.exists():
            depth = cv2.resize(np.load(cache / "aligned_dense_depth.npy").astype(np.float32), self.output_size)
            confidence = cv2.resize(np.load(cache / "fused_confidence.npy").astype(np.float32), self.output_size)
        else:
            depth = np.ones(self.output_size[::-1], np.float32)
            confidence = np.zeros_like(depth)
        depth = np.log1p(np.clip(depth, 0.0, 160.0)) / np.log(161.0)
        yy, xx = np.indices(depth.shape, dtype=np.float32)
        ray_x = (xx / max(depth.shape[1] - 1, 1)) * 2.0 - 1.0
        ray_y = (yy / max(depth.shape[0] - 1, 1)) * 2.0 - 1.0
        geometry = np.stack([
            depth, confidence, 1.0 - residual_hole.astype(np.float32), ray_x, ray_y,
        ]).astype(np.float32)
        token_path = self._token_cache_path(reference)
        tokens = np.load(token_path).astype(np.float32) if token_path.exists() else np.empty((0, 0, 0), np.float32)
        return {
            "target_rgb": torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0,
            "masked_rgb": torch.from_numpy(t0_prefilled.transpose(2, 0, 1).copy()).float() / 255.0,
            "hole_mask": torch.from_numpy(residual_hole[None].astype(np.float32)),
            "original_hole_mask": torch.from_numpy(hole[None]),
            "recoverable_mask": torch.from_numpy(recoverable[None]),
            "retrieved_rgb": torch.from_numpy(retrieved.transpose(2, 0, 1).copy()).float() / 255.0,
            "retrieval_confidence": torch.from_numpy(retrieval_confidence[None]),
            "geometry": torch.from_numpy(geometry.copy()),
            "tokens": torch.from_numpy(tokens),
            "token_path": str(token_path),
            "camera": reference.camera,
        }
