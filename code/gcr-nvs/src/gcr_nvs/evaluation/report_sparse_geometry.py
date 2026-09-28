"""Evaluate Sparse Geometry Student on teacher-query occupancy contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gcr_nvs.datasets.sparse_geometry_dataset import (
    FREE_SPACE,
    SparseGeometryDistillationDataset,
)
from gcr_nvs.geometry.temporal_lidar import EXACT_CURRENT, TEMPORAL_FILLED
from gcr_nvs.models.sparse_geometry import SparseGeometryStudent


def _binary_metrics(predicted, positive, negative) -> dict[str, float]:
    true_positive = int((predicted & positive).sum())
    false_negative = int((~predicted & positive).sum())
    false_positive = int((predicted & negative).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": 2.0 * precision * recall / max(precision + recall, 1e-8),
        "occupancy_iou": true_positive / max(
            true_positive + false_positive + false_negative, 1,
        ),
        "free_space_accuracy": float(
            (~predicted[negative]).float().mean() if negative.any() else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--teacher-root", type=Path, default=Path("outputs/geometry_teacher"))
    parser.add_argument("--sequence", default="2026-05-22-15-09-17")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--exclude-frame-id", type=int)
    parser.add_argument("--frame-min", type=int)
    parser.add_argument("--frame-max", type=int)
    parser.add_argument("--max-input-points", type=int, default=8000)
    parser.add_argument("--max-positive-queries", type=int, default=8000)
    parser.add_argument("--max-negative-queries", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/sparse_geometry_report.json"),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    include_free_input = bool(
        state.get("config", {}).get("include_free_input", False)
    )
    include_temporal_input = bool(
        state.get("config", {}).get("include_temporal_input", False)
    )
    dataset = SparseGeometryDistillationDataset(
        args.teacher_root,
        args.sequence,
        max_input_points=args.max_input_points,
        max_positive_queries=args.max_positive_queries,
        max_negative_queries=args.max_negative_queries,
        include_free_input=include_free_input,
        include_temporal_input=include_temporal_input,
        temporal_input_fraction=float(
            state.get("config", {}).get("temporal_input_fraction", 0.40)
        ),
    )
    model = SparseGeometryStudent(
        input_dim=7 if include_free_input else 6,
        query_position_encoding=bool(
            state.get("config", {}).get("query_position_encoding", False)
        ),
    ).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    rows = []
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            frame_id = int(sample["frame_id"])
            if args.exclude_frame_id is not None and frame_id == args.exclude_frame_id:
                continue
            if args.frame_min is not None and frame_id < args.frame_min:
                continue
            if args.frame_max is not None and frame_id > args.frame_max:
                continue
            batch = {
                key: value.to(device)
                for key, value in sample.items()
                if isinstance(value, torch.Tensor)
            }
            outputs = model(
                batch["input_features"],
                batch["input_indices"],
                tuple(int(value) for value in batch["spatial_shape"].tolist()),
                batch_size=1,
                query_indices=batch["query_indices"],
            )
            valid = outputs["query_validity"] > 0.5
            target = batch["occupancy_target"] > 0.5
            positive = valid & target
            negative = valid & ~target
            source_kind = batch["query_source_kind"]
            distance = batch["query_distance"]
            threshold_metrics = {}
            for threshold in thresholds:
                predicted = outputs["occupancy"] > threshold
                threshold_metrics[str(threshold)] = _binary_metrics(
                    predicted, positive, negative,
                )
            default_metrics = threshold_metrics[str(0.5)] if "0.5" in threshold_metrics else threshold_metrics[str(thresholds[0])]
            default_threshold = 0.5 if "0.5" in threshold_metrics else thresholds[0]
            predicted = outputs["occupancy"] > default_threshold
            source_metrics = {}
            for name, value in (
                ("exact", EXACT_CURRENT),
                ("temporal", TEMPORAL_FILLED),
                ("free_space", FREE_SPACE),
            ):
                selected = valid & (source_kind == value)
                source_metrics[name] = {
                    "count": int(selected.sum()),
                    "occupancy_rate": float(
                        predicted[selected].float().mean() if selected.any() else 0.0
                    ),
                }
            distance_metrics = {}
            for name, lower, upper in (
                ("near", 0.0, 20.0),
                ("mid", 20.0, 50.0),
                ("far", 50.0, float("inf")),
            ):
                selected = valid & (distance >= lower) & (distance < upper)
                distance_metrics[name] = {
                    "count": int(selected.sum()),
                    **_binary_metrics(
                        predicted,
                        positive & selected,
                        negative & selected,
                    ),
                }
            rows.append({
                "frame_id": frame_id,
                "query_coverage": float(valid.float().mean()),
                "occupancy_bce": float(F.binary_cross_entropy(
                    outputs["occupancy"][valid],
                    batch["occupancy_target"][valid],
                )),
                **default_metrics,
                "source_metrics": source_metrics,
                "distance_metrics": distance_metrics,
                "threshold_metrics": threshold_metrics,
            })
            if len(rows) >= args.max_samples:
                break
    if not rows:
        raise RuntimeError("no evaluation samples were selected")
    aggregate = {
        name: sum(row[name] for row in rows) / len(rows)
        for name in (
            "query_coverage",
            "occupancy_bce",
            "positive_precision",
            "positive_recall",
            "positive_f1",
            "occupancy_iou",
            "free_space_accuracy",
        )
    }
    threshold_aggregate = {
        threshold: {
            name: sum(row["threshold_metrics"][threshold][name] for row in rows) / len(rows)
            for name in (
                "positive_precision",
                "positive_recall",
                "positive_f1",
                "occupancy_iou",
                "free_space_accuracy",
            )
        }
        for threshold in rows[0]["threshold_metrics"]
    }
    source_aggregate = {
        source: {
            "count": sum(row["source_metrics"][source]["count"] for row in rows),
            "occupancy_rate": sum(
                row["source_metrics"][source]["occupancy_rate"]
                * row["source_metrics"][source]["count"]
                for row in rows
            ) / max(sum(row["source_metrics"][source]["count"] for row in rows), 1),
        }
        for source in rows[0]["source_metrics"]
    }
    distance_aggregate = {
        band: {
            name: sum(row["distance_metrics"][band][name] for row in rows) / len(rows)
            for name in (
                "positive_precision",
                "positive_recall",
                "positive_f1",
                "occupancy_iou",
                "free_space_accuracy",
            )
        }
        for band in rows[0]["distance_metrics"]
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "sequence_id": args.sequence,
        "excluded_frame_id": args.exclude_frame_id,
        "sample_count": len(rows),
        "aggregate": aggregate,
        "source_aggregate": source_aggregate,
        "distance_aggregate": distance_aggregate,
        "threshold_sweep": threshold_aggregate,
        "frames": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
