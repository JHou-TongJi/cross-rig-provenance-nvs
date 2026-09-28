"""Register, deduplicate, and split the expanded dense-route dataset.

This script never copies or mutates source data. It writes only manifests and
audit metadata to the requested output directory, normally on the 2T SSD.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import yaml

from gcr_nvs.datasets.manifest import build_sequence_manifest, discover_sequences


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_signature(sequence_dir: Path) -> tuple[str, list[dict[str, str | int]]]:
    records = build_sequence_manifest(sequence_dir, compute_quality=False)
    if not records:
        raise ValueError(f"empty sequence: {sequence_dir}")
    indices = sorted({0, len(records) // 2, len(records) - 1})
    sampled: list[dict[str, str | int]] = []
    digest = hashlib.sha256()
    for index in indices:
        record = records[index]
        relative_paths = [record.camera_config, record.lidar, record.cameras["CAM_FRONT_NARROW"]]
        for relative in relative_paths:
            path = sequence_dir / relative
            if not path.exists():
                raise FileNotFoundError(path)
            file_hash = _sha256(path)
            stat = path.stat()
            sampled.append({
                "frame_id": int(record.frame_id),
                "path": relative,
                "size": int(stat.st_size),
                "sha256": file_hash,
            })
            digest.update(f"{relative}:{stat.st_size}:{file_hash}\n".encode("utf-8"))
    return digest.hexdigest(), sampled


def _session_key(sequence_id: str) -> str:
    parts = sequence_id.split("-")
    if len(parts) < 4:
        return sequence_id
    return "-".join(parts[:3]) + "-hour-" + parts[3][:2]


def _choose_session_splits(groups: dict[str, list[str]], total: int) -> dict[str, list[str]]:
    names = sorted(groups)
    sizes = [len(groups[name]) for name in names]
    target_val = round(total * 0.11)
    target_test = round(total * 0.11)
    best = None
    for assignment in itertools.product(("train", "val", "test"), repeat=len(names)):
        counts = {"train": 0, "val": 0, "test": 0}
        group_counts = {"train": 0, "val": 0, "test": 0}
        for size, split in zip(sizes, assignment):
            counts[split] += size
            group_counts[split] += 1
        if not all(group_counts[name] for name in ("train", "val", "test")):
            continue
        score = abs(counts["val"] - target_val) + abs(counts["test"] - target_test)
        score += 0.05 * abs(group_counts["val"] - group_counts["test"])
        if best is None or score < best[0]:
            best = (score, assignment, counts)
    if best is None:
        raise RuntimeError("could not assign session groups to train/val/test")
    assignment = best[1]
    result = {"train": [], "val": [], "test": []}
    for name, split in zip(names, assignment):
        result[split].extend(sorted(groups[name]))
    for values in result.values():
        values.sort()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--incoming-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    roots = [("formal", args.formal_root.resolve()), ("incoming", args.incoming_root.resolve())]
    candidates = []
    for source_kind, root in roots:
        for sequence_dir in discover_sequences(root):
            signature, sampled = _sequence_signature(sequence_dir)
            records = build_sequence_manifest(sequence_dir, compute_quality=False)
            candidates.append({
                "sequence_id": sequence_dir.name,
                "source_kind": source_kind,
                "source_root": str(root),
                "path": str(sequence_dir),
                "signature": signature,
                "sampled_files": sampled,
                "frames": len(records),
                "session": _session_key(sequence_dir.name),
            })

    by_signature: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_signature[row["signature"]].append(row)

    canonical = []
    duplicate_groups = []
    for signature, rows in sorted(by_signature.items()):
        rows.sort(key=lambda row: (row["source_kind"] != "formal", row["sequence_id"]))
        selected = rows[0]
        canonical.append(selected)
        duplicate_groups.append({
            "signature": signature,
            "canonical": selected["sequence_id"],
            "members": [
                {"sequence_id": row["sequence_id"], "source_kind": row["source_kind"], "path": row["path"]}
                for row in rows
            ],
        })

    sequence_ids = sorted(row["sequence_id"] for row in canonical)
    sessions: dict[str, list[str]] = defaultdict(list)
    by_id = {row["sequence_id"]: row for row in canonical}
    for sequence_id in sequence_ids:
        sessions[by_id[sequence_id]["session"]].append(sequence_id)
    split = _choose_session_splits(sessions, len(sequence_ids))

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "dedup_groups.json").write_text(
        json.dumps(duplicate_groups, ensure_ascii=False, indent=2) + "\n",
    )
    (args.output_root / "sequence_index.json").write_text(
        json.dumps({"canonical_sequences": canonical, "split": split}, ensure_ascii=False, indent=2) + "\n",
    )
    with (args.output_root / "all_sequences.jsonl").open("w", encoding="utf-8") as handle:
        for sequence_id in sequence_ids:
            row = by_id[sequence_id]
            for record in build_sequence_manifest(Path(row["path"]), compute_quality=False):
                payload = {
                    "sequence_id": record.sequence_id,
                    "frame_id": record.frame_id,
                    "timestamp": record.timestamp,
                    "camera_config": record.camera_config,
                    "cameras": record.cameras,
                    "lidar": record.lidar,
                    "camera_timestamps": record.camera_timestamps,
                    "lidar_timestamp": record.lidar_timestamp,
                    "source_root": row["source_root"],
                    "source_kind": row["source_kind"],
                    "split": next(name for name, values in split.items() if sequence_id in values),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    (args.output_root / "dense_route_split_v1.yaml").write_text(
        yaml.safe_dump(split, allow_unicode=True, sort_keys=False),
    )
    audit = {
        "formal_root": str(args.formal_root.resolve()),
        "incoming_root": str(args.incoming_root.resolve()),
        "candidate_sequence_count": len(candidates),
        "canonical_sequence_count": len(canonical),
        "duplicate_group_count": sum(len(group["members"]) > 1 for group in duplicate_groups),
        "duplicate_member_count": sum(max(0, len(group["members"]) - 1) for group in duplicate_groups),
        "split_sequence_counts": {name: len(values) for name, values in split.items()},
        "split_frame_counts": {
            name: sum(by_id[sequence_id]["frames"] for sequence_id in values)
            for name, values in split.items()
        },
        "session_groups": {name: sorted(values) for name, values in sessions.items()},
    }
    (args.output_root / "split_audit_v1.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
