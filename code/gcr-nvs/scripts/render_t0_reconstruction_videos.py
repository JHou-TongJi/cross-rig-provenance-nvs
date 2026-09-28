"""Create timestamp-aware videos from T0 reconstructed frame anchors.

Unlike ``render_30fps_videos.py``, this script never reads source RGB as the
video content. It reads ``run_t0_native_reconstruction_video.py`` outputs for
one pose, keeps all real reconstructed anchors, and linearly inserts frames at
the requested FPS. The 1920x1080 raster is checked for every input frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest, sensor_timestamp


def _writer(path: Path, size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError(f"cannot open video writer: {path}")


def _read(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if (image.shape[1], image.shape[0]) != size:
        raise ValueError(f"reconstruction frame {path} has {(image.shape[1], image.shape[0])}, expected {size}")
    return image


def _contact(output: Path, cameras: list[str], pose: str, fps: float, size: tuple[int, int]) -> None:
    paths = [output / f"{camera}_{int(fps)}fps.mp4" for camera in cameras]
    readers = [cv2.VideoCapture(str(path)) for path in paths]
    contact_size = (size[0] * 2, size[1] * 4)
    writer = _writer(output / f"seven_view_{pose}_{int(fps)}fps.mp4", contact_size, fps)
    try:
        while True:
            frames = []
            for reader in readers:
                ok, frame = reader.read()
                if not ok:
                    return
                frames.append(frame)
            blank = np.zeros_like(frames[0])
            rows = []
            for start in range(0, len(frames), 2):
                pair = frames[start:start + 2]
                if len(pair) == 1:
                    pair.append(blank.copy())
                rows.append(np.concatenate(pair, axis=1))
            writer.write(np.concatenate(rows, axis=0))
    finally:
        writer.release()
        for reader in readers:
            reader.release()


def render_pose(sequence: Path, reconstruction_root: Path, output: Path, pose: str,
                fps: float, cameras: list[str], size: tuple[int, int],
                records_override=None) -> dict:
    records = records_override or build_sequence_manifest(sequence, compute_quality=False)
    if len(records) < 2:
        raise ValueError("sequence must contain at least two frames")
    writers = {camera: _writer(output / f"{camera}_{int(fps)}fps.mp4", size, fps) for camera in cameras}
    frame_count = 0
    try:
        for index in range(len(records) - 1):
            left, right = records[index], records[index + 1]
            dt = sensor_timestamp(right) - sensor_timestamp(left)
            if dt <= 0:
                raise ValueError(f"non-increasing timestamp {left.frame_id}->{right.frame_id}")
            slices = max(1, int(round(dt * fps)))
            left_images = {camera: _read(reconstruction_root / "frames" / sequence.name / camera / pose / f"{left.frame_id:06d}.jpg", size) for camera in cameras}
            right_images = {camera: _read(reconstruction_root / "frames" / sequence.name / camera / pose / f"{right.frame_id:06d}.jpg", size) for camera in cameras}
            if index == 0:
                for camera in cameras:
                    writers[camera].write(left_images[camera])
                frame_count += 1
            for step in range(1, slices):
                alpha = step / slices
                for camera in cameras:
                    frame = cv2.addWeighted(left_images[camera], 1.0 - alpha, right_images[camera], alpha, 0.0)
                    writers[camera].write(frame)
                frame_count += 1
            for camera in cameras:
                writers[camera].write(right_images[camera])
            frame_count += 1
    finally:
        for writer in writers.values():
            writer.release()
    timestamps = [sensor_timestamp(record) for record in records]
    report = {
        "sequence": sequence.name,
        "pose": pose,
        "fps": fps,
        "resolution": [size[0], size[1]],
        "real_reconstructed_frames": len(records),
        "output_frames": frame_count,
        "duration_s": (frame_count - 1) / fps,
        "cameras": cameras,
        "anchor_policy": "all real anchors are T0 reconstructed frames; intermediate frames are timestamp-aware linear RGB interpolation",
        "first_timestamp": timestamps[0],
        "last_timestamp": timestamps[-1],
        "metric_lidar_policy": "LiDAR supervision exists only at real anchors; interpolated frames have no synthetic LiDAR truth",
    }
    (output / "render_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if cameras == list(CAMERA_NAMES):
        _contact(output, cameras, pose, fps, size)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pose", choices=("mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm"), required=True)
    parser.add_argument("--fps", type=float, default=180.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera", action="append", choices=CAMERA_NAMES, default=None)
    parser.add_argument("--frame-list", type=int, nargs="+", default=None,
                        help="Explicit reconstructed anchor frame IDs; useful when only keyframe caches exist.")
    args = parser.parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("fps and output dimensions must be positive")
    cameras = args.camera or list(CAMERA_NAMES)
    output = args.output_root / args.pose
    output.mkdir(parents=True, exist_ok=True)
    if args.frame_list is not None:
        # Keep only explicit anchor records. Intermediate output is still
        # timestamp-aware linear RGB interpolation between real anchors.
        original = build_sequence_manifest(args.sequence, compute_quality=False)
        wanted = set(args.frame_list)
        selected = [record for record in original if record.frame_id in wanted]
        if len(selected) < 2:
            parser.error("--frame-list requires at least two anchor frames")
        import tempfile
        # render_pose accepts a sequence path and discovers its manifest. A
        # filtered manifest is represented by a temporary symlink-free subset
        # directory only for the manifest lookup, while image paths remain in
        # the original reconstruction root.
        report = render_pose(args.sequence, args.reconstruction_root, output, args.pose, args.fps, cameras, (args.width, args.height), selected)
    else:
        report = render_pose(args.sequence, args.reconstruction_root, output, args.pose, args.fps, cameras, (args.width, args.height))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
