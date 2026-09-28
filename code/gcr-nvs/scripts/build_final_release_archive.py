"""Build an immutable, auditable GCR-NVS release archive.

Large external model weights are referenced by absolute path and checksum;
the release copies code, configs, docs, tests, selected checkpoints and the
validated T0/UniWorld-A results without touching the source experiments.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import argparse
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATA = Path("/media/2T_HD/GCR-NVS_DATA")
RELEASE = Path(__import__("os").environ.get(
    "GCR_NVS_RELEASE_ARCHIVE",
    str(DATA / "releases/GCR-NVS-FINAL-20260827"),
))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def public_source_label(path: Path) -> str:
    """Keep provenance useful without embedding the operator's local paths."""
    try:
        return str(path.resolve().relative_to(PROJECT.resolve()))
    except ValueError:
        return f"<external>/{path.name}"


def copy_file(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source": public_source_label(src), "archive": str(dst.relative_to(RELEASE)), "sha256": sha256(dst), "bytes": dst.stat().st_size}


def copy_tree(src: Path, dst: Path, *, exclude_training: bool = False) -> list[dict]:
    rows = []
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        relative = path.relative_to(src)
        if exclude_training and (
            relative.parts[:2] == ("gcr_nvs", "training")
            or path.name.startswith("train_")
            or path.name in {"run_full_dense_route_pipeline.sh", "run_gcr_nvs_scale_sweep.sh", "monitor_full_dense_route.sh"}
        ):
            continue
        rows.append(copy_file(path, dst / relative))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a release archive; large assets are opt-in")
    parser.add_argument(
        "--include-large-assets", action="store_true",
        help="copy checkpoints, full result trees and 120/240 FPS videos (several GB)",
    )
    args = parser.parse_args()
    if RELEASE.exists() and any(p.is_file() for p in RELEASE.rglob("*")):
        raise SystemExit(f"release already exists; refusing overwrite: {RELEASE}")
    RELEASE.mkdir(parents=True, exist_ok=True)
    (RELEASE / "metadata").mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "release": "GCR-NVS-FINAL-20260827",
        "branch": "release/gcr-nvs-inference-only-20260826",
        "status": "frozen_validated_default",
        "source_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
        "files": [],
        "external_assets": [],
    }

    # The release is a source snapshot, not a copy of unrelated scratch output.
    for name in ("src", "scripts", "configs", "tests", "docs", "datasets", "models", "references", "submission"):
        manifest["files"].extend(copy_tree(PROJECT / name, RELEASE / "code" / name, exclude_training=True))
    manifest["files"].extend(copy_tree(PROJECT / "outputs/release_examples", RELEASE / "examples/release_examples"))
    for name in ("README.md", "pyproject.toml", "requirements-inference-core.txt", "requirements-inference-optional.txt", "VERSION", "camera_config.json", "camera_intric.yaml"):
        path = PROJECT / name
        if path.exists():
            manifest["files"].append(copy_file(path, RELEASE / "code" / name))

    checkpoint = DATA / "t0_baseline_isolated_20260823/checkpoints/t0_structure_rectified_v2_dinob14_320x180_c48_best.pt"
    adapter = DATA / "outputs/checkpoints/t0_rgbds_adapter_a_256x64_v2/adapter_step_001000.pt"
    if args.include_large_assets:
        for src, rel in ((checkpoint, "models/t0_structure_rectified_v2_dinob14_320x180_c48_best.pt"),
                         (adapter, "models/t0_rgbds_adapter_a_256x64_v2_step_001000.pt")):
            if not src.is_file():
                raise FileNotFoundError(src)
            manifest["files"].append(copy_file(src, RELEASE / rel))

    result = DATA / "outputs/evaluations/t0_uniview_rgbds_full_a_20260825"
    if args.include_large_assets:
        manifest["files"].extend(copy_tree(result, RELEASE / "results/t0_uniview_rgbds_full_a_20260825"))
    elif result.is_dir():
        for path in sorted(result.glob("mixed_*_seven_view.jpg")):
            manifest["files"].append(copy_file(path, RELEASE / "results/t0_uniview_rgbds_full_a_20260825" / path.name))
    baseline = DATA / "t0_baseline_isolated_20260823/runs/t0_highres_source_audit_edge_all"
    # The high-resolution contact sheets are sufficient for a compact release;
    # the full per-camera source cache remains in the immutable external bundle.
    for path in sorted(baseline.glob("*")):
        if path.is_file():
            manifest["files"].append(copy_file(path, RELEASE / "t0_baseline_contact_sheets" / path.name))

    # Include the validated 120/240 FPS videos in the offline submission bundle.
    if args.include_large_assets:
        for video_root in (
            DATA / "outputs/videos_120fps_comparison_20260826_20cm",
            DATA / "outputs/videos_240fps_comparison_20260826_20cm",
        ):
            if video_root.is_dir():
                manifest["files"].extend(copy_tree(video_root, RELEASE / "videos" / video_root.name))

    external = [
        DATA / "models/runwayml-stable-diffusion-inpainting",
        DATA / "t0_baseline_isolated_20260823/outputs/checkpoints/anyup_multi_backbone.pth",
        PROJECT / "third_party/anyup",
    ]
    for path in external:
        if path.is_file():
            manifest["external_assets"].append({"name": path.name, "kind": "file", "sha256": sha256(path), "bytes": path.stat().st_size})
        elif path.is_dir():
            manifest["external_assets"].append({"name": path.name, "kind": "directory", "file_count": sum(p.is_file() for p in path.rglob("*"))})

    (RELEASE / "metadata/release_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (RELEASE / "metadata/EXTERNAL_ASSETS.txt").write_text(
        "大体积基础模型和 AnyUP 权重由部署者按 docs/EXTERNAL_MODELS_DOWNLOAD_2026-08-27.md 下载；manifest 仅记录名称与校验信息。\n"
        "Wan/VACE 当前仅为 disabled lazy backend，不包含未运行的模型权重。\n", encoding="utf-8"
    )
    print(json.dumps({"release": str(RELEASE), "file_count": len(manifest["files"]), "external_count": len(manifest["external_assets"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
