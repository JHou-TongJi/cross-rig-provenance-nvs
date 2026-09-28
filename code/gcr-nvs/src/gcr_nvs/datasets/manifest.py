"""Manifest generation and loading for the captured multi-sensor sequences."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageStat


CAMERA_NAMES = (
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_NARROW",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_WIDE",
)

# Cameras in one family share an optical axis but may use different focal lengths.
CAMERA_FAMILIES = {
    "CAM_FRONT_WIDE": "front_forward",
    "CAM_FRONT_NARROW": "front_forward",
}

CAMERA_FOCAL_ROLES = {
    "CAM_FRONT_WIDE": "wide",
    "CAM_FRONT_NARROW": "narrow",
}

# Hard appearance topology for the physical camera rig. Cameras outside a
# target's local optical neighborhood must never be offered as RGB sources.
# Geometry remains shared globally through the LiDAR field.
CAMERA_APPEARANCE_NEIGHBORS = {
    "CAM_BACK": ("CAM_BACK_LEFT", "CAM_BACK_RIGHT"),
    "CAM_BACK_LEFT": ("CAM_BACK",),
    "CAM_BACK_RIGHT": ("CAM_BACK",),
    "CAM_FRONT_LEFT": ("CAM_FRONT_WIDE",),
    "CAM_FRONT_NARROW": ("CAM_FRONT_WIDE",),
    "CAM_FRONT_RIGHT": ("CAM_FRONT_WIDE",),
    "CAM_FRONT_WIDE": ("CAM_FRONT_NARROW", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"),
}


def appearance_source_cameras(
    target_camera: str,
    *,
    include_target: bool,
) -> tuple[str, ...]:
    """Return same-direction RGB sources in deterministic priority order."""
    if target_camera not in CAMERA_APPEARANCE_NEIGHBORS:
        raise ValueError(f"unknown target camera: {target_camera}")
    neighbors = CAMERA_APPEARANCE_NEIGHBORS[target_camera]
    return ((target_camera,) + neighbors) if include_target else neighbors


@dataclass(frozen=True)
class FrameRecord:
    sequence_id: str
    frame_id: int
    timestamp: float
    camera_config: str
    cameras: dict[str, str]
    lidar: str
    camera_timestamps: dict[str, float] | None = None
    lidar_timestamp: float | None = None
    image_quality: dict[str, dict[str, float | int | bool]] | None = None
    lidar_point_count: int | None = None
    quality_flags: tuple[str, ...] = ()


def _timestamp(path: Path) -> float:
    return float(path.name.rsplit(".", 1)[0])


def sensor_timestamp(record: FrameRecord) -> float:
    """Return the precise LiDAR time used for temporal ordering."""
    return float(
        record.lidar_timestamp
        if record.lidar_timestamp is not None
        else record.timestamp
    )


def sort_records_by_sensor_time(records: Iterable[FrameRecord]) -> list[FrameRecord]:
    """Sort records by LiDAR acquisition time while preserving frame identity."""
    return sorted(records, key=lambda record: (sensor_timestamp(record), record.frame_id))


def _image_quality(path: Path) -> dict[str, float | int | bool]:
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            return {
                "decode_ok": True,
                "width": int(image.width),
                "height": int(image.height),
                "brightness_mean": float(ImageStat.Stat(image).mean[0]),
                "brightness_p01": float(np.percentile(gray, 1)),
                "brightness_p99": float(np.percentile(gray, 99)),
                "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            }
    except Exception:
        return {"decode_ok": False, "width": 0, "height": 0, "brightness_mean": 0.0, "brightness_p01": 0.0, "brightness_p99": 0.0, "blur_score": 0.0}


def _pcd_point_count(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            for raw in handle:
                line = raw.decode("ascii", errors="ignore").strip()
                if line.startswith("POINTS"):
                    return int(line.split()[1])
                if line.startswith("DATA"):
                    break
    except Exception:
        pass
    return None


def discover_sequences(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "Key_frames").is_dir()
    )


def build_sequence_manifest(sequence_dir: Path, compute_quality: bool = True) -> list[FrameRecord]:
    key_frames = sequence_dir / "Key_frames"
    config_dir = key_frames / "camera_config"
    records: list[FrameRecord] = []
    for config_path in sorted(config_dir.glob("*.json"), key=lambda p: int(p.stem)):
        payload = json.loads(config_path.read_text())
        sensors = payload["sensors"]
        missing = [name for name in (*CAMERA_NAMES, "LIDAR_CONCAT") if name not in sensors]
        if missing:
            raise ValueError(f"{config_path}: missing sensors: {', '.join(missing)}")
        camera_paths = {
            name: (key_frames / name / sensors[name]) for name in CAMERA_NAMES
        }
        lidar_path = key_frames / "LIDAR_CONCAT" / sensors["LIDAR_CONCAT"]
        camera_timestamps = {name: _timestamp(path) for name, path in camera_paths.items()}
        lidar_timestamp = _timestamp(lidar_path)
        image_quality = ({name: _image_quality(path) for name, path in camera_paths.items()} if compute_quality else None)
        quality_flags = []
        if image_quality is not None:
            for name, quality in image_quality.items():
                if not quality["decode_ok"]:
                    quality_flags.append(f"{name}:decode_failed")
                if quality["width"] == 0 or quality["height"] == 0:
                    quality_flags.append(f"{name}:invalid_size")
        records.append(
            FrameRecord(
                sequence_id=sequence_dir.name,
                frame_id=int(config_path.stem),
                timestamp=float(payload["timestamp"]),
                camera_config=str(config_path.relative_to(sequence_dir)),
                cameras={name: str(path.relative_to(sequence_dir)) for name, path in camera_paths.items()},
                lidar=str(lidar_path.relative_to(sequence_dir)),
                camera_timestamps=camera_timestamps,
                lidar_timestamp=lidar_timestamp,
                image_quality=image_quality,
                lidar_point_count=_pcd_point_count(lidar_path) if compute_quality else None,
                quality_flags=tuple(quality_flags),
            )
        )
    if not records:
        raise ValueError(f"No camera_config JSON files found in {config_dir}")
    return sort_records_by_sensor_time(records)


def build_manifests(root: Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for sequence_dir in discover_sequences(root):
        records.extend(build_sequence_manifest(sequence_dir))
    return records


def write_jsonl(records: Iterable[FrameRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[FrameRecord]:
    records = [
        FrameRecord(**json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return sort_records_by_sensor_time(records)


def split_sequences(
    sequence_ids: list[str], train: int = 4, val: int = 1, test: int = 1
) -> dict[str, list[str]]:
    if len(sequence_ids) != train + val + test:
        raise ValueError(f"expected {train + val + test} sequences, got {len(sequence_ids)}")
    return {
        "train": sequence_ids[:train],
        "val": sequence_ids[train : train + val],
        "test": sequence_ids[train + val :],
    }
