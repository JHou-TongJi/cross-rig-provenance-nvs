"""Project-level checks for the final GCR-NVS branch.

The command is intentionally cheap: it validates repository contracts and
external paths without loading DA3, DINO, Stable Diffusion or Wan weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = (
    "src/gcr_nvs/models/t0_structure_constraint.py",
    "src/gcr_nvs/models/video_diffusion_backend.py",
    "scripts/build_dense_route_manifest.py",
    "scripts/build_da3_depth_cache_split.py",
    "scripts/infer_t0_end_to_end.py",
    "scripts/render_t0_highres_source_audit.py",
    "scripts/demo_t0_reconstruction.py",
    "scripts/reproduce_t0_video.py",
    "scripts/render_t0_reconstruction_videos.py",
    "scripts/render_t0_video_demo.py",
    "scripts/distort_t0_demo_output.py",
    "scripts/build_source_vs_aonly_diff.py",
    "docs/DEMO_MINIMAL_INFERENCE_2026-08-26.md",
    "docs/EXTERNAL_MODELS_DOWNLOAD_2026-08-27.md",
    "datasets/README.md",
    "models/MODEL_REGISTRY.md",
    "configs/releases/t0_uniview_a_final.yaml",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--release-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise SystemExit("missing required project files: " + ", ".join(missing))
    config = (root / "configs/releases/t0_uniview_a_final.yaml").read_text(encoding="utf-8")
    for token in ("observed_pixels_immutable: true", "hole_only: true", "backend: disabled"):
        if token not in config:
            raise SystemExit(f"release config is missing contract: {token}")
    manifest_report = None
    if args.manifest:
        if not args.manifest.is_file():
            raise SystemExit(f"manifest does not exist: {args.manifest}")
        lines = [line for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise SystemExit("manifest is empty")
        first = json.loads(lines[0])
        required_fields = {"sequence_id", "frame_id", "cameras", "lidar", "camera_config", "source_root"}
        absent = sorted(required_fields - set(first))
        if absent:
            raise SystemExit("manifest sample missing fields: " + ", ".join(absent))
        manifest_report = {"records": len(lines), "sample_fields": sorted(first)}
    result = {
        "status": "ok",
        "required_files": len(REQUIRED),
        "manifest": manifest_report,
        "release_root": str(args.release_root) if args.release_root else None,
        "video_diffusion_default": "disabled",
        "contracts": ["rectified_rgb_newK_D0", "observed_pixels_immutable", "hole_only_completion"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
