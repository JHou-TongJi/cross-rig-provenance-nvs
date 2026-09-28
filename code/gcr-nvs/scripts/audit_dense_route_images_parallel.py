"""Parallel full-content decode check for all RGB files in a dense manifest."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def _decode(job: tuple[str, str, int, str]) -> dict:
    sequence, camera, frame_id, path_text = job
    path = Path(path_text)
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.convert("RGB").load()
        return {
            "ok": True,
            "sequence": sequence,
            "camera": camera,
            "frame_id": frame_id,
            "width": width,
            "height": height,
        }
    except Exception as error:
        return {
            "ok": False,
            "sequence": sequence,
            "camera": camera,
            "frame_id": frame_id,
            "path": path_text,
            "error": repr(error),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    jobs = []
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        root = Path(row["source_root"]) / row["sequence_id"]
        for camera, relative in row["cameras"].items():
            jobs.append((row["sequence_id"], camera, int(row["frame_id"]), str(root / relative)))

    failures = []
    dimensions = {}
    decoded = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for result in pool.map(_decode, jobs):
            if result["ok"]:
                decoded += 1
                key = f"{result['width']}x{result['height']}"
                dimensions.setdefault(result["camera"], {})[key] = dimensions.setdefault(result["camera"], {}).get(key, 0) + 1
            else:
                failures.append(result)

    report = {
        "manifest": str(args.manifest.resolve()),
        "expected_images": len(jobs),
        "decoded_images": decoded,
        "failure_count": len(failures),
        "dimensions": dimensions,
        "failures": failures,
        "workers": args.workers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary = {key: report[key] for key in ("expected_images", "decoded_images", "failure_count", "workers")}
    summary["output"] = str(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
