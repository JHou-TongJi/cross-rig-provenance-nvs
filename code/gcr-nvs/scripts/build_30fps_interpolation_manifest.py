"""Build a timestamp-aware 30 FPS interpolation manifest for one sequence.

The captured sequence remains the source of truth. Real frames are emitted as
anchors; intermediate records only describe a temporal interpolation request
between two adjacent real frames. They intentionally carry no fabricated LiDAR
path or metric-depth GT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gcr_nvs.datasets.manifest import build_sequence_manifest, sensor_timestamp


def build_manifest(sequence_dir: Path, fps: float) -> tuple[list[dict], dict]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    records = build_sequence_manifest(sequence_dir, compute_quality=False)
    if len(records) < 2:
        raise ValueError("at least two real frames are required")

    output: list[dict] = []
    synthetic_count = 0
    interval_counts: list[int] = []

    for index in range(len(records) - 1):
        left = records[index]
        right = records[index + 1]
        t0 = sensor_timestamp(left)
        t1 = sensor_timestamp(right)
        dt = t1 - t0
        if not dt > 0:
            raise ValueError(
                f"non-increasing sensor timestamps at frames {left.frame_id} -> {right.frame_id}: {t0}, {t1}"
            )

        # Number of equal-duration time slices in this real interval. For a
        # nominal 0.5 s interval at 30 FPS this is 15, i.e. 14 synthetic frames.
        slices = max(1, int(round(dt * fps)))
        interval_counts.append(slices)

        if index == 0:
            output.append(
                {
                    "kind": "real",
                    "synthetic": False,
                    "sequence_id": left.sequence_id,
                    "frame_id": left.frame_id,
                    "timestamp_s": t0,
                    "source_frame_id": left.frame_id,
                    "camera_config": left.camera_config,
                    "cameras": left.cameras,
                    "lidar": left.lidar,
                    "lidar_available": True,
                }
            )

        for step in range(1, slices):
            alpha = step / slices
            synthetic_count += 1
            output.append(
                {
                    "kind": "interpolated",
                    "synthetic": True,
                    "sequence_id": left.sequence_id,
                    "frame_id": None,
                    "timestamp_s": t0 + alpha * dt,
                    "source_frame_id": left.frame_id,
                    "source_next_frame_id": right.frame_id,
                    "source_timestamp_s": t0,
                    "source_next_timestamp_s": t1,
                    "alpha": alpha,
                    "camera_config": None,
                    "cameras": None,
                    "lidar": None,
                    "lidar_available": False,
                    "metric_gt_available": False,
                    "interpolation_contract": (
                        "RGB/geometry must be generated or motion-compensated from the two real anchors; "
                        "do not treat this record as a measured sensor frame"
                    ),
                }
            )

        output.append(
            {
                "kind": "real",
                "synthetic": False,
                "sequence_id": right.sequence_id,
                "frame_id": right.frame_id,
                "timestamp_s": t1,
                "source_frame_id": right.frame_id,
                "camera_config": right.camera_config,
                "cameras": right.cameras,
                "lidar": right.lidar,
                "lidar_available": True,
            }
        )

    timestamps = [row["timestamp_s"] for row in output]
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    summary = {
        "sequence_id": records[0].sequence_id,
        "real_frames": len(records),
        "output_records": len(output),
        "synthetic_records": synthetic_count,
        "requested_fps": fps,
        "real_timestamp_start_s": sensor_timestamp(records[0]),
        "real_timestamp_end_s": sensor_timestamp(records[-1]),
        "real_duration_s": sensor_timestamp(records[-1]) - sensor_timestamp(records[0]),
        "interval_slices_min": min(interval_counts),
        "interval_slices_median": sorted(interval_counts)[len(interval_counts) // 2],
        "interval_slices_max": max(interval_counts),
        "output_delta_min_s": min(deltas),
        "output_delta_max_s": max(deltas),
        "real_anchor_policy": "all real frames retained",
        "synthetic_lidar_policy": "none; synthetic rows have no metric LiDAR GT",
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path, help="sequence directory containing Key_frames/")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True, help="output JSON manifest path")
    args = parser.parse_args()

    rows, summary = build_manifest(args.sequence, args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "frames": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
