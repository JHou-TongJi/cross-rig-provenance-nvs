"""Encode source/T0/A-only anchor videos and aligned comparison videos.

This is the reproducible FFmpeg-backed video demo for the T0 20 cm result.
It consumes existing anchor artifacts; it does not run reconstruction again.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2

from gcr_nvs.datasets.manifest import build_sequence_manifest, sensor_timestamp


def _encoder(path: Path, width: int, height: int, fps: int, crf: int = 18):
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
         "-preset", "ultrafast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE,
    )


def _read(path: Path, size: tuple[int, int]):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)


def _write_variant(records, paths, output: Path, width: int, height: int, fps: int) -> int:
    process = _encoder(output, width, height, fps)
    count = 0
    try:
        for index in range(len(records) - 1):
            left = _read(paths[index], (width, height))
            right = _read(paths[index + 1], (width, height))
            slices = max(1, round((sensor_timestamp(records[index + 1]) - sensor_timestamp(records[index])) * fps))
            if index == 0:
                process.stdin.write(left.tobytes())
                count += 1
            for step in range(1, slices):
                alpha = step / slices
                process.stdin.write(cv2.addWeighted(left, 1.0 - alpha, right, alpha, 0.0).tobytes())
                count += 1
            process.stdin.write(right.tobytes())
            count += 1
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"FFmpeg failed for {output}")
    except Exception:
        process.kill()
        process.wait()
        raise
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="T0 source/T0-blue/A-only video demo")
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True,
                        help="t0_a_only_video.../anchor_work directory")
    parser.add_argument("--a-only-root", type=Path, required=True,
                        help="t0_a_only_video.../frames directory")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pose", default="mixed_10_20cm")
    parser.add_argument("--fps", type=int, choices=(120, 240), default=120)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera", action="append", required=True)
    args = parser.parse_args()
    records = build_sequence_manifest(args.sequence, compute_quality=False)
    out = args.output_root / args.pose
    out.mkdir(parents=True, exist_ok=True)
    variants = {}
    for camera in args.camera:
        base = args.anchor_root
        variants["source"] = [base / f"{r.frame_id:06d}" / args.pose / camera / "source_native_rectified.png" for r in records]
        variants["t0_blue"] = [base / f"{r.frame_id:06d}" / args.pose / camera / "t0_surface_2px_repaired_blue.png" for r in records]
        variants["a_only"] = [args.a_only_root / args.sequence.name / camera / args.pose / f"{r.frame_id:06d}.jpg" for r in records]
        for name, paths in variants.items():
            _write_variant(records, paths, out / f"{camera}_{name}_{args.fps}fps.mp4", args.width, args.height, args.fps)
        comparison = out / f"{camera}_comparison_source_blue_a_only_{args.fps}fps.mp4"
        inputs = [str(out / f"{camera}_{name}_{args.fps}fps.mp4") for name in ("source", "t0_blue", "a_only")]
        filter_graph = (
            "[0:v]scale=640:360,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            "text='SOURCE RGB':x=16:y=16:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65[s];"
            "[1:v]scale=640:360,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            "text='T0 BLUE HOLE':x=16:y=16:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65[b];"
            "[2:v]scale=640:360,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            "text='A-ONLY FINAL | +20cm':x=16:y=16:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65[a];"
            "[s][b][a]hstack=inputs=3[out]"
        )
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", inputs[0], "-i", inputs[1], "-i", inputs[2],
                         "-filter_complex", filter_graph, "-map", "[out]", "-r", str(args.fps),
                         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p", str(comparison)], check=True)
    report = {
        "sequence": args.sequence.name, "pose": args.pose, "cameras": args.camera,
        "fps": args.fps, "resolution": [args.width, args.height], "real_anchor_frames": len(records),
        "interpolation": "timestamp-aware linear RGB between real anchors",
        "variants": {"source": "rectified source RGB", "t0_blue": "T0 RGB with invalid pixels shown blue", "a_only": "A-only final RGB"},
        "note": "This demo encodes variants independently; it does not perform reconstruction or diffusion inference.",
    }
    (out / "demo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
