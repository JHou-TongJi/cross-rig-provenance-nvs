"""Explicit target sensor profiles used to condition novel-view rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from gcr_nvs.geometry.calibration import CameraCalibration


@dataclass(frozen=True)
class TargetCameraSpec:
    name: str
    model: str
    width: int
    height: int
    inherit_source_camera: str | None
    intrinsic: np.ndarray | None
    distortion: np.ndarray
    external: np.ndarray
    vehicle_offset_m: np.ndarray
    vehicle_rotation_rpy_deg: np.ndarray
    intrinsic_focal_scale: float = 1.0
    principal_point_offset_px: np.ndarray = None


@dataclass(frozen=True)
class TargetRigSpec:
    rig_id: str
    vehicle_convention: str
    cameras: tuple[TargetCameraSpec, ...]

    def camera(self, name: str) -> TargetCameraSpec:
        for camera in self.cameras:
            if camera.name == name:
                return camera
        raise KeyError(f"unknown target camera {name!r}")


def _matrix(value, shape):
    if value == "identity":
        return np.eye(shape[0], dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    return array.reshape(shape)


def load_target_rig(path: Path) -> TargetRigSpec:
    payload = yaml.safe_load(path.read_text()) or {}
    vehicle = payload.get("vehicle_frame", {})
    cameras = []
    for item in payload.get("cameras", []):
        intrinsic = item.get("K")
        cameras.append(
            TargetCameraSpec(
                name=str(item["name"]),
                model=str(item.get("model", "pinhole_brown")),
                width=int(item["width"]),
                height=int(item["height"]),
                inherit_source_camera=item.get("inherit_source_camera"),
                intrinsic=None if intrinsic is None else _matrix(intrinsic, (3, 3)),
                distortion=np.asarray(item.get("D", [0, 0, 0, 0, 0]), dtype=np.float64),
                external=_matrix(item.get("T_camera_from_vehicle", "identity"), (4, 4)),
                vehicle_offset_m=np.asarray(item.get("vehicle_offset_m", [0.0, 0.0, 0.0]), dtype=np.float64),
                vehicle_rotation_rpy_deg=np.asarray(
                    item.get("vehicle_rotation_rpy_deg", [0.0, 0.0, 0.0]),
                    dtype=np.float64,
                ),
                intrinsic_focal_scale=float(item.get("intrinsic_focal_scale", 1.0)),
                principal_point_offset_px=np.asarray(
                    item.get("principal_point_offset_px", [0.0, 0.0]), dtype=np.float64,
                ),
            )
        )
    if not cameras:
        raise ValueError(f"{path}: no cameras declared")
    return TargetRigSpec(
        rig_id=str(payload["rig_id"]),
        vehicle_convention=str(vehicle.get("convention", "x_forward_y_left_z_up")),
        cameras=tuple(cameras),
    )


def resolve_target_calibration(spec: TargetCameraSpec, source: dict[str, CameraCalibration]) -> CameraCalibration:
    """Resolve an explicit target camera, inheriting source K/D when requested."""
    if spec.inherit_source_camera:
        base = source[spec.inherit_source_camera]
        if spec.intrinsic is None:
            intrinsic = base.intrinsic.copy()
            intrinsic[0] *= spec.width / base.width
            intrinsic[1] *= spec.height / base.height
        else:
            intrinsic = spec.intrinsic.copy()
        distortion = base.distortion.copy() if not np.any(spec.distortion) else spec.distortion.copy()
        external = base.external.copy() if np.allclose(spec.external, np.eye(4)) else spec.external.copy()
    else:
        if spec.intrinsic is None:
            raise ValueError(f"target camera {spec.name} requires K or inherit_source_camera")
        intrinsic, distortion, external = spec.intrinsic, spec.distortion, spec.external.copy()

    if spec.intrinsic_focal_scale <= 0:
        raise ValueError(f"target camera {spec.name} intrinsic_focal_scale must be positive")
    intrinsic = intrinsic.copy()
    intrinsic[0, 0] *= spec.intrinsic_focal_scale
    intrinsic[1, 1] *= spec.intrinsic_focal_scale
    principal_offset = np.asarray(spec.principal_point_offset_px, dtype=np.float64).reshape(-1)
    if principal_offset.size != 2:
        raise ValueError(f"target camera {spec.name} principal_point_offset_px must have two values")
    intrinsic[0, 2] += principal_offset[0]
    intrinsic[1, 2] += principal_offset[1]

    roll, pitch, yaw = np.deg2rad(spec.vehicle_rotation_rpy_deg)
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rotation_delta = np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])
    base_rotation = external[:3, :3].copy()
    base_center = -base_rotation.T @ external[:3, 3]
    target_center = base_center + spec.vehicle_offset_m
    target_rotation = base_rotation @ rotation_delta.T
    external[:3, :3] = target_rotation
    external[:3, 3] = -target_rotation @ target_center
    return CameraCalibration(spec.name, intrinsic, distortion, external, spec.width, spec.height)
