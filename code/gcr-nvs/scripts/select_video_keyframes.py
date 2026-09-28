"""Deterministically select real and virtual keyframes for video-diffusion clips.

The selector intentionally treats measured LiDAR frames as privileged anchors.
When a candidate manifest contains 80 real frames and synthetic intermediate
frames, all real frames are retained by default and the remaining budget is
filled with high-novelty virtual frames. Virtual rows must be marked
``synthetic=true`` and are never treated as metric geometry supervision.

Input JSON may be either a list of candidate rows or ``{"candidates": [...]}``.
Each row should contain normalized [0, 1] metrics when available:
``pose_novelty``, ``visibility_novelty``, ``semantic_novelty``,
``lidar_coverage``, ``quality``, ``dynamic_ratio`` and ``timestamp_s``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


WEIGHTS = {
    "pose_novelty": 0.30,
    "visibility_novelty": 0.25,
    "semantic_novelty": 0.20,
    "lidar_coverage": 0.15,
    "quality": 0.10,
}


def _clip(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(0.0, min(1.0, value))


def keyframe_score(row: dict[str, Any]) -> float:
    score = sum(weight * _clip(row.get(name)) for name, weight in WEIGHTS.items())
    return score - 0.15 * _clip(row.get("dynamic_ratio"))


def _is_real(row: dict[str, Any]) -> bool:
    return not bool(row.get("synthetic", row.get("is_real") is False))


def _quality_gate(row: dict[str, Any]) -> bool:
    if row.get("decode_ok", True) is False:
        return False
    if _clip(row.get("invalid_fraction")) > 0.70:
        return False
    if _clip(row.get("overexposure_fraction")) > 0.50:
        return False
    if _clip(row.get("underexposure_fraction")) > 0.50:
        return False
    if row.get("temporal_alignment_valid", True) is False and _is_real(row):
        return False
    return True


def _time(row: dict[str, Any]) -> float:
    value = row.get("timestamp_s", row.get("timestamp", row.get("frame_id", 0)))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def select_keyframes(
    candidates: list[dict[str, Any]],
    total: int = 100,
    virtual_min_gap_s: float = 0.20,
) -> list[dict[str, Any]]:
    if total < 1:
        raise ValueError("total must be positive")
    eligible = [row for row in candidates if _quality_gate(row)]
    real = sorted((row for row in eligible if _is_real(row)), key=_time)
    virtual = sorted(
        (row for row in eligible if not _is_real(row)),
        key=lambda row: (-keyframe_score(row), _time(row)),
    )
    if len(real) > total:
        # The normal contract is 80 real frames and total=100. If a caller
        # requests a smaller budget, preserve endpoints and rank the rest.
        endpoints = real[:1] + (real[-1:] if len(real) > 1 else [])
        middle = [row for row in real if row not in endpoints]
        chosen = endpoints + sorted(middle, key=lambda row: (-keyframe_score(row), _time(row)))[: total - len(endpoints)]
    else:
        chosen = list(real)
        for row in virtual:
            if len(chosen) >= total:
                break
            timestamp = _time(row)
            if any(abs(timestamp - _time(other)) < virtual_min_gap_s for other in chosen if not _is_real(other)):
                continue
            chosen.append(row)
    chosen = sorted(chosen, key=_time)
    for rank, row in enumerate(chosen):
        row["keyframe_rank"] = rank
        row["selection_score"] = keyframe_score(row)
        row["geometry_supervision"] = "metric_lidar" if _is_real(row) else "none_synthetic_appearance_only"
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Candidate JSON list or object with candidates")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--virtual-min-gap-s", type=float, default=0.20)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    candidates = payload["candidates"] if isinstance(payload, dict) else payload
    if not isinstance(candidates, list) or not all(isinstance(row, dict) for row in candidates):
        raise ValueError("input must contain a list of candidate objects")
    selected = select_keyframes(candidates, args.total, args.virtual_min_gap_s)
    result = {
        "total_candidates": len(candidates),
        "eligible_candidates": sum(_quality_gate(row) for row in candidates),
        "selected_count": len(selected),
        "real_selected": sum(_is_real(row) for row in selected),
        "synthetic_selected": sum(not _is_real(row) for row in selected),
        "selection_rule": {
            "weights": WEIGHTS,
            "dynamic_penalty": 0.15,
            "virtual_min_gap_s": args.virtual_min_gap_s,
            "real_frames_are_metric_anchors": True,
        },
        "keyframes": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("total_candidates", "eligible_candidates", "selected_count", "real_selected", "synthetic_selected")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
