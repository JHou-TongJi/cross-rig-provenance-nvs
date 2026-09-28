"""Lightweight frame-to-frame dynamic instance association."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    track_id: int
    centroid: np.ndarray
    last_frame: int
    confidence: float


class CentroidTracker:
    def __init__(self, max_distance_m: float = 4.0, max_age: int = 3):
        self.max_distance_m = max_distance_m
        self.max_age = max_age
        self.next_id = 1
        self.tracks: list[Track] = []

    def update(self, centroids: np.ndarray, frame_id: int) -> np.ndarray:
        centroids = np.asarray(centroids, dtype=np.float32).reshape(-1, 3)
        assignments = np.full(len(centroids), -1, dtype=np.int64)
        unused = set(range(len(self.tracks)))
        for detection_id, centroid in enumerate(centroids):
            if not unused:
                break
            candidates = list(unused)
            distances = [float(np.linalg.norm(self.tracks[index].centroid - centroid)) for index in candidates]
            best = candidates[int(np.argmin(distances))]
            if min(distances) <= self.max_distance_m:
                track = self.tracks[best]
                track.centroid = centroid
                track.last_frame = frame_id
                track.confidence = min(1.0, track.confidence + 0.1)
                assignments[detection_id] = track.track_id
                unused.remove(best)
        for detection_id, centroid in enumerate(centroids):
            if assignments[detection_id] >= 0:
                continue
            track = Track(self.next_id, centroid, frame_id, 0.5)
            self.next_id += 1
            self.tracks.append(track)
            assignments[detection_id] = track.track_id
        self.tracks = [track for track in self.tracks if frame_id - track.last_frame <= self.max_age]
        return assignments
