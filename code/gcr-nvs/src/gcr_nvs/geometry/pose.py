"""Explicit vehicle-pose interfaces used by the geometry front-end."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def identity_pose() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


@dataclass(frozen=True)
class PoseQuality:
    fitness: float = 1.0
    inlier_ratio: float = 1.0
    static_residual_median: float = 0.0
    static_residual_p95: float = 0.0
    pose_quality: float = 1.0
    failure_flag: bool = False


@dataclass(frozen=True)
class PoseSample:
    timestamp: float
    world_from_vehicle: np.ndarray
    quality: PoseQuality


class PoseProvider:
    """Pose provider with a safe single-frame fallback.

    External odometry can be added by supplying timestamped poses. Until an
    odometry source is verified, every frame remains in its own vehicle frame
    and temporal fusion must be disabled by the caller.
    """

    def __init__(self, samples: list[PoseSample] | None = None):
        self.samples = sorted(samples or [], key=lambda sample: sample.timestamp)

    @property
    def temporal_valid(self) -> bool:
        return len(self.samples) > 1

    def at(self, timestamp: float) -> PoseSample:
        if not self.samples:
            return PoseSample(timestamp, identity_pose(), PoseQuality(pose_quality=0.0, failure_flag=True))
        index = min(range(len(self.samples)), key=lambda i: abs(self.samples[i].timestamp - timestamp))
        return self.samples[index]

    def relative(self, source_timestamp: float, target_timestamp: float) -> np.ndarray:
        source = self.at(source_timestamp).world_from_vehicle
        target = self.at(target_timestamp).world_from_vehicle
        return np.linalg.inv(target) @ source
