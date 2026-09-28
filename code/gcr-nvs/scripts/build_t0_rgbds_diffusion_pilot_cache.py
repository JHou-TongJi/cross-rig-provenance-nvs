"""Build fixed, leakage-safe RGB-D-S pilot cases from the 188-sequence data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from gcr_nvs.datasets.t0_completion import (
    T0SemanticCompletionDataset,
    completion_references,
)
from gcr_nvs.datasets.t0_rgbds_diffusion import (
    assert_no_hole_leakage,
    build_rgbds_conditions,
    resize_hole_mask,
)
from gcr_nvs.models.dino_anyup_descriptor import FrozenDinoAnyUpDescriptor


def _available(references, depth_root: Path, dynamic_root: Path):
    return [reference for reference in references if (
        depth_root / reference.sequence / f"{reference.record.frame_id:06d}"
        / reference.camera / "aligned_dense_depth.npy"
    ).exists() and (
        dynamic_root / reference.sequence / f"{reference.record.frame_id:06d}"
        / f"{reference.camera}.dynamic.npy"
    ).exists()]


def _spread(references, maximum: int):
    ranked = sorted(references, key=lambda ref: hashlib.sha1(
        f"{ref.sequence}/{ref.record.frame_id}/{ref.camera}".encode("ascii")
    ).hexdigest())
    if maximum <= 0 or maximum >= len(ranked):
        return ranked
    indices = np.linspace(0, len(ranked) - 1, maximum, dtype=np.int64)
    return [ranked[int(index)] for index in indices]


def _mask_bank(root: Path, size: tuple[int, int]):
    output: dict[str, list[np.ndarray]] = {}
    for path in sorted(root.glob("mixed_*/CAM_*/validity_after_2px_surface_repair.npy")):
        output.setdefault(path.parent.name, []).append(
            resize_hole_mask(~np.load(path, mmap_mode="r").astype(bool), size)
        )
    return output


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    default_bundle = Path("/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=default_bundle)
    parser.add_argument("--depth-root", type=Path, default=Path(
        "/media/2T_HD/GCR-NVS_DATA/outputs/da3_depth_cache_dense_train_t0_v2"
    ))
    parser.add_argument("--dynamic-root", type=Path, default=Path(
        "/media/2T_HD/GCR-NVS_DATA/outputs/dynamic_masks_full_v1"
    ))
    parser.add_argument("--audit-root", type=Path, default=default_bundle / "runs/t0_target_surface_2px_repair_all_20260824")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--maximum", type=int, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    size = (args.width, args.height)
    references = completion_references(
        args.bundle / "datasets", args.bundle / "datasets/split_manifest.yaml", args.split,
    )
    references = _spread(_available(references, args.depth_root, args.dynamic_root), args.maximum)
    loader = T0SemanticCompletionDataset(
        references,
        dataset_root=args.bundle / "datasets",
        distortion_config=args.bundle / "camera_intric.yaml",
        rgb_cache_root=args.bundle / "data/t0_completion_rgb_rectified_512x288",
        token_cache_root=args.bundle / "data/t0_completion_dinov2_l_tokens_64",
        audit_root=args.audit_root,
        output_size=size,
    )
    masks = _mask_bank(args.audit_root, size)
    device = torch.device(args.device)
    descriptor = FrozenDinoAnyUpDescriptor(
        model_name="dinov2_vitl14", output_channels=8, dino_input_size=(364, 644),
        anyup_checkpoint=project / "outputs/checkpoints/anyup_multi_backbone.pth",
        anyup_root=project / "third_party/anyup", q_chunk_size=4096,
    ).to(device).eval()
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for index, reference in enumerate(references):
        folder = args.output / f"{index:06d}"
        folder.mkdir(exist_ok=True)
        rgb_u8 = loader._rectified_rgb(reference)
        rgb = rgb_u8.astype(np.float32) / 255.0
        hole_values = masks[reference.camera]
        hole = hole_values[int(hashlib.sha1(
            f"{reference.sequence}/{reference.record.frame_id}".encode("ascii")
        ).hexdigest(), 16) % len(hole_values)]
        depth = cv2.resize(np.load(
            args.depth_root / reference.sequence / f"{reference.record.frame_id:06d}"
            / reference.camera / "aligned_dense_depth.npy",
        ).astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        dynamic = cv2.resize(np.load(
            args.dynamic_root / reference.sequence / f"{reference.record.frame_id:06d}"
            / f"{reference.camera}.dynamic.npy",
        ).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        masked = rgb.copy(); masked[hole] = 0.0
        image = torch.from_numpy(masked.transpose(2, 0, 1).copy()).to(device)[None]
        with torch.inference_mode():
            semantic = descriptor(image, (args.height, args.width))[0].cpu().numpy()
        first = build_rgbds_conditions(rgb, depth, dynamic, semantic, hole)
        changed = rgb.copy(); changed[hole] = np.random.default_rng(index).random((hole.sum(), 3))
        second = build_rgbds_conditions(changed, depth, dynamic, semantic, hole)
        assert_no_hole_leakage(first, second)
        Image.fromarray(rgb_u8).save(folder / "target.png")
        np.save(folder / "depth.npy", depth.astype(np.float32))
        np.save(folder / "dynamic.npy", dynamic)
        np.save(folder / "semantic.npy", semantic.astype(np.float16))
        np.save(folder / "hole.npy", hole)
        rows.append({
            "sequence": reference.sequence, "frame": reference.record.frame_id,
            "camera": reference.camera, "rgb": f"{index:06d}/target.png",
            "depth": f"{index:06d}/depth.npy", "dynamic": f"{index:06d}/dynamic.npy",
            "semantic": f"{index:06d}/semantic.npy", "hole": f"{index:06d}/hole.npy",
            "hole_fraction": float(hole.mean()),
            "forbid_foreground_fraction": float(first["forbid_foreground_extension"].mean()),
        })
        if (index + 1) % 10 == 0:
            print(json.dumps({"processed": index + 1, "total": len(references)}), flush=True)
    payload = {
        "split": args.split, "size": list(size), "samples": rows,
        "contract": "masked RGB DINO-L+AnyUP; GT hole pixels are loss-only; 2px residual masks",
    }
    (args.output / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
