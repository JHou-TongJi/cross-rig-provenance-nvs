"""Re-render T0 with depth sidecars and copy them only after RGB hash checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    bundle = Path("/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=bundle)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    original = args.bundle / "runs/t0_highres_source_audit_edge_all"
    temporary = args.bundle / "runs/t0_highres_source_audit_edge_all_depth_tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    renderer = Path(__file__).with_name("render_t0_highres_source_audit.py")
    command = [
        sys.executable, str(renderer),
        "--bundle", str(args.bundle),
        "--target-camera", "all", "--depth-mode", "edge", "--output", str(temporary),
        "--device", args.device,
    ]
    subprocess.run(command, check=True)
    rows = []
    for source in sorted(original.glob("*/CAM_*")):
        candidate = temporary / source.relative_to(original)
        original_rgb = source / "t0_highres_blue.png"
        candidate_rgb = candidate / "t0_highres_blue.png"
        rows.append({
            "relative": str(source.relative_to(original)),
            "original_sha256": sha256(original_rgb),
            "candidate_sha256": sha256(candidate_rgb),
            "match": sha256(original_rgb) == sha256(candidate_rgb),
        })
    if not rows or not all(row["match"] for row in rows):
        raise RuntimeError("T0 RGB sidecar rerender did not reproduce all original SHA256 values")
    for source in sorted(temporary.glob("*/CAM_*")):
        if source.parent.name == "source_sidecars":
            continue
        destination = original / source.relative_to(temporary)
        for name in ("depth.npy", "provenance.npy"):
            shutil.copy2(source / name, destination / name)
    source_sidecars = original / "source_sidecars"
    if source_sidecars.exists():
        shutil.rmtree(source_sidecars)
    shutil.copytree(temporary / "source_sidecars", source_sidecars)
    report = {
        "source": str(original), "temporary": str(temporary), "cases": len(rows),
        "rgb_sha256_all_match": True,
        "sidecars": ["depth.npy", "provenance.npy", "source_sidecars/<camera>/depth.npy"],
        "rows": rows,
    }
    (original / "depth_sidecar_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(rows), "rgb_sha256_all_match": True, "source": str(original)}, indent=2))


if __name__ == "__main__":
    main()
