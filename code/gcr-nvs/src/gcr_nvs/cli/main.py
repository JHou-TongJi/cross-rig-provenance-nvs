"""Command line entry points for the first GCR-NVS geometry baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gcr_nvs.datasets.manifest import build_manifests, read_jsonl, split_sequences, write_jsonl
from gcr_nvs.geometry.calibration import load_calibrations, project_world_points
from gcr_nvs.geometry.pcd import read_pcd, voxel_downsample
from gcr_nvs.geometry.local_points import estimate_normals
from gcr_nvs.rendering.surfel import load_rgb, render_points, save_render
from gcr_nvs.rendering.fixed_splat import FixedSplatRenderer, save_render_result
from gcr_nvs.evaluation.calibration_report import build_calibration_report
from gcr_nvs.evaluation.pose_report import build_pose_report


def command_manifest(args: argparse.Namespace) -> None:
    records = build_manifests(args.root)
    write_jsonl(records, args.output)
    sequence_ids = sorted({record.sequence_id for record in records})
    split = split_sequences(sequence_ids, args.train, args.val, args.test)
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    args.split_output.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"frames": len(records), "sequences": sequence_ids, "split": split}, ensure_ascii=False, indent=2))


def command_project(args: argparse.Namespace) -> None:
    records = read_jsonl(args.manifest)
    record = next(record for record in records if record.sequence_id == args.sequence and record.frame_id == args.frame)
    sequence_dir = args.root / record.sequence_id
    calibration = load_calibrations(sequence_dir / record.camera_config, args.distortion)
    points = read_pcd(sequence_dir / record.lidar, fields=("x", "y", "z"))
    points = voxel_downsample(points, args.voxel)
    report = {}
    for name, camera in calibration.items():
        _, valid = project_world_points(points, camera)
        report[name] = {"points": int(len(points)), "projected_points": int(valid.sum()), "coverage_ratio": float(valid.mean())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_render(args: argparse.Namespace) -> None:
    records = read_jsonl(args.manifest)
    record = next(record for record in records if record.sequence_id == args.sequence and record.frame_id == args.frame)
    sequence_dir = args.root / record.sequence_id
    calibration = load_calibrations(sequence_dir / record.camera_config, args.distortion)
    target = calibration[args.target_camera]
    points = read_pcd(sequence_dir / record.lidar, fields=("x", "y", "z"))
    points = voxel_downsample(points, args.voxel)
    colors = np.full((len(points), 3), 0.5, dtype=np.float32)
    provenance = np.full(len(points), -1, dtype=np.int16)
    observation_count = np.zeros(len(points), dtype=np.float32)
    for source_index, (source_name, relative_path) in enumerate(record.cameras.items()):
        if source_name == args.target_camera:
            continue
        source = load_rgb(sequence_dir / relative_path)
        source_calibration = calibration[source_name]
        pixels, valid = project_world_points(points, source_calibration)
        xy = np.rint(pixels[valid]).astype(np.int64)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < source.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < source.shape[0])
        selected = np.flatnonzero(valid)[inside]
        colors[selected] = source[xy[inside, 1], xy[inside, 0]]
        provenance[selected] = source_index
        observation_count[selected] += 1
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be provided together")
    render = FixedSplatRenderer().render(
        points,
        colors,
        target,
        normals=estimate_normals(points),
        source_observation_count=observation_count,
        source_provenance=provenance,
        output_size=(args.width, args.height) if args.width else None,
    )
    stem = f"{record.sequence_id}_{record.frame_id:06d}_{args.target_camera}"
    save_render_result(render, args.output, stem)
    print(json.dumps({"stem": stem, "valid_pixels": int((render.opacity[0] > 0).sum()), "channels": 12}, indent=2))


def command_calibration_report(args: argparse.Namespace) -> None:
    sequences = args.sequences or [path.name for path in sorted(args.root.iterdir()) if path.is_dir() and (path / "Key_frames").is_dir()]
    report = build_calibration_report(args.root, sequences, args.distortion, args.output, args.samples_per_sequence, not args.no_overlays)
    print(json.dumps({"sequences": list(report["sequences"]), "output": str(args.output)}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gcr-nvs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("root", type=Path)
    manifest.add_argument("--output", type=Path, default=Path("outputs/manifests/all.jsonl"))
    manifest.add_argument("--split-output", type=Path, default=Path("outputs/manifests/split.json"))
    manifest.add_argument("--train", type=int, default=4)
    manifest.add_argument("--val", type=int, default=1)
    manifest.add_argument("--test", type=int, default=1)
    manifest.set_defaults(function=command_manifest)
    project = subparsers.add_parser("project")
    project.add_argument("root", type=Path)
    project.add_argument("manifest", type=Path)
    project.add_argument("sequence")
    project.add_argument("--frame", type=int, default=1)
    project.add_argument("--distortion", type=Path, default=None)
    project.add_argument("--voxel", type=float, default=0.08)
    project.add_argument("--output", type=Path, default=Path("outputs/calibration_reports/report.json"))
    project.set_defaults(function=command_project)
    render = subparsers.add_parser("render")
    render.add_argument("root", type=Path)
    render.add_argument("manifest", type=Path)
    render.add_argument("sequence")
    render.add_argument("--frame", type=int, default=1)
    render.add_argument("--target-camera", default="CAM_FRONT_WIDE")
    render.add_argument("--distortion", type=Path, default=None)
    render.add_argument("--voxel", type=float, default=0.08)
    render.add_argument("--width", type=int, default=None)
    render.add_argument("--height", type=int, default=None)
    render.add_argument("--output", type=Path, default=Path("outputs/coarse"))
    render.set_defaults(function=command_render)
    calibration = subparsers.add_parser("calibration-report")
    calibration.add_argument("root", type=Path)
    calibration.add_argument("--sequences", nargs="*", default=None)
    calibration.add_argument("--distortion", type=Path, default=None)
    calibration.add_argument("--samples-per-sequence", type=int, default=20)
    calibration.add_argument("--no-overlays", action="store_true")
    calibration.add_argument("--output", type=Path, default=Path("outputs/calibration_reports/pilot_6seq_v1.json"))
    calibration.set_defaults(function=command_calibration_report)
    pose = subparsers.add_parser("pose-report")
    pose.add_argument("root", type=Path)
    pose.add_argument("--sequences", nargs="+", required=True)
    pose.add_argument("--output", type=Path, default=Path("outputs/poses/pilot_6seq_v1.json"))
    pose.set_defaults(function=lambda args: print(json.dumps({"output": str(args.output), "sequences": args.sequences}, indent=2)) or build_pose_report(args.root, args.sequences, args.output))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
