"""Compare semantic DeepFill checkpoints on sequence-disjoint validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from gcr_nvs.datasets.t0_completion import (
    T0SemanticCompletionDataset,
    completion_references,
    evenly_subsample_references,
)
from gcr_nvs.models.deepfill_v2_reference import SemanticDeepFillV2
from gcr_nvs.models.dino_anyup_descriptor import FrozenDinoAnyUpDescriptor


def _u8(tensor: torch.Tensor) -> np.ndarray:
    return np.rint(np.clip(tensor.detach().cpu().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)


def _panel(path: Path, entries: list[tuple[str, np.ndarray]]) -> None:
    width, height = 512, 288
    title = 34
    canvas = Image.new("RGB", (width * len(entries), height + title), "white")
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(entries):
        draw.text((index * width + 8, 5), label, fill="black", font=font)
        canvas.paste(Image.fromarray(array).resize((width, height)), (index * width, title))
    canvas.save(path, quality=94)


def main() -> None:
    bundle = Path("/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823")
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--bundle", type=Path, default=bundle)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--output", type=Path, default=bundle / "runs/t0_semantic_deepfill_v2_pilot_eval")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    references = evenly_subsample_references(
        completion_references(args.bundle / "datasets", args.bundle / "datasets/split_manifest.yaml", "val"),
        args.cases,
    )
    dataset = T0SemanticCompletionDataset(
        references, dataset_root=args.bundle / "datasets",
        distortion_config=args.bundle / "camera_intric.yaml",
        rgb_cache_root=args.bundle / "data/t0_completion_rgb_rectified_512x288",
        token_cache_root=args.bundle / "data/t0_completion_dinov2_l_tokens_64",
        audit_root=args.bundle / "runs/t0_highres_source_audit_edge_all", seed=20260825,
    )
    device = torch.device(args.device)
    descriptor = FrozenDinoAnyUpDescriptor(
        model_name="dinov2_vitl14", output_channels=64,
        anyup_checkpoint=args.bundle / "outputs/checkpoints/anyup_multi_backbone.pth",
        anyup_root=args.bundle / "third_party/anyup",
    ).to(device).eval()
    descriptor.dino = torch.nn.Identity()
    models = []
    for checkpoint in args.checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        base = int(state["config"]["base_channels"])
        model = SemanticDeepFillV2(64, 5, base).to(device).eval()
        model.load_state_dict(state["model"])
        models.append((checkpoint.stem, model))
    args.output.mkdir(parents=True, exist_ok=True)
    aggregates = {name: {"hole_sum": 0.0, "boundary_sum": 0.0, "count": 0} for name, _ in models}
    for index in range(len(dataset)):
        batch = dataset[index]
        target = batch["target_rgb"][None].to(device)
        masked = batch["masked_rgb"][None].to(device)
        hole = batch["hole_mask"][None].to(device)
        retrieved = batch["retrieved_rgb"][None].to(device)
        retrieval_confidence = batch["retrieval_confidence"][None].to(device)
        geometry = batch["geometry"][None].to(device)
        tokens = batch["tokens"][None].to(device)
        with torch.inference_mode():
            semantic = descriptor.upsample_tokens(target, tokens, target.shape[-2:])
        entries = [("真实目标", _u8(target[0])), ("源图找回后的 residual", _u8(masked[0])), ("源图语义参考", _u8(retrieved[0]))]
        boundary = F.max_pool2d(hole, 15, stride=1, padding=7)
        for name, model in models:
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(masked, hole, retrieved, retrieval_confidence, semantic, geometry)
            difference = (output["rgb"] - target).abs()
            hole_l1 = float((difference * hole).sum() / (hole.sum() * 3).clamp_min(1.0))
            boundary_l1 = float((difference * boundary).sum() / (boundary.sum() * 3).clamp_min(1.0))
            observed = ~hole.bool().expand_as(masked)
            changed = int((output["rgb"][observed] != masked[observed]).sum())
            if changed:
                raise RuntimeError(f"{name} modified {changed} observed scalar pixels")
            aggregates[name]["hole_sum"] += hole_l1
            aggregates[name]["boundary_sum"] += boundary_l1
            aggregates[name]["count"] += 1
            entries.append((f"{name}  洞L1={hole_l1:.3f}", _u8(output["rgb"][0])))
        _panel(args.output / f"val_{index:03d}_{batch['camera']}.jpg", entries)
    report = {
        name: {
            "hole_l1": values["hole_sum"] / values["count"],
            "boundary_l1": values["boundary_sum"] / values["count"],
            "cases": values["count"], "observed_modified": 0,
        }
        for name, values in aggregates.items()
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
