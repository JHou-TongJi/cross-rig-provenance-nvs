"""Cache compact DINOv2-L patch descriptors for all completion samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from gcr_nvs.datasets.t0_completion import (
    T0SemanticCompletionDataset,
    completion_references,
    evenly_subsample_references,
)
from gcr_nvs.models.dino_anyup_descriptor import FrozenDinoAnyUpDescriptor


def main() -> None:
    bundle = Path("/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=bundle)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    references = completion_references(
        args.bundle / "datasets", args.bundle / "datasets/split_manifest.yaml", args.split,
    )
    references = evenly_subsample_references(references, args.max_cases)
    rgb_cache = args.bundle / "data/t0_completion_rgb_rectified_512x288"
    token_cache = args.bundle / "data/t0_completion_dinov2_l_tokens_64"
    dataset = T0SemanticCompletionDataset(
        references,
        dataset_root=args.bundle / "datasets",
        distortion_config=args.bundle / "camera_intric.yaml",
        rgb_cache_root=rgb_cache,
        token_cache_root=token_cache,
        audit_root=args.bundle / "runs/t0_highres_source_audit_edge_all",
    )
    device = torch.device(args.device)
    descriptor = FrozenDinoAnyUpDescriptor(
        model_name="dinov2_vitl14", output_channels=64,
        anyup_checkpoint=args.bundle / "outputs/checkpoints/anyup_multi_backbone.pth",
        anyup_root=args.bundle / "third_party/anyup",
    ).to(device).eval()
    written = skipped = 0
    for index, reference in enumerate(references):
        path = dataset._token_cache_path(reference)
        if path.exists():
            skipped += 1
            continue
        rgb = dataset._rectified_rgb(reference)
        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(
            device=device, dtype=torch.float32,
        )[None] / 255.0
        tokens = descriptor.encode_tokens(image)[0].cpu().numpy().astype(np.float16)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(temporary, tokens)
        os.replace(temporary, path)
        written += 1
        if written % 10 == 0:
            print(json.dumps({
                "processed": index + 1, "total": len(references),
                "written": written, "skipped": skipped, "token_shape": list(tokens.shape),
            }), flush=True)
    report = {
        "split": args.split, "cases": len(references), "written": written, "skipped": skipped,
        "rgb_cache": str(rgb_cache), "token_cache": str(token_cache),
        "descriptor": "frozen dinov2_vitl14 final normalized patch tokens projected to 64 channels",
    }
    token_cache.mkdir(parents=True, exist_ok=True)
    (token_cache / f"cache_{args.split}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
