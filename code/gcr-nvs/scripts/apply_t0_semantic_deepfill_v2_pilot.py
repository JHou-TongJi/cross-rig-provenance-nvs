"""Apply a semantic DeepFill pilot to one latest T0 residual audit sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from gcr_nvs.models.deepfill_v2_reference import SemanticDeepFillV2
from gcr_nvs.models.dino_anyup_descriptor import FrozenDinoAnyUpDescriptor


def _tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(
        device=device, dtype=torch.float32,
    )[None] / 255.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    bundle = Path("/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=bundle)
    parser.add_argument("--pose", default="mixed_20_50cm")
    parser.add_argument("--camera", default="CAM_FRONT_NARROW")
    parser.add_argument("--checkpoint", type=Path, default=bundle / "checkpoints/t0_semantic_deepfill_v2_residual_pilot_000750.pt")
    parser.add_argument("--output", type=Path, default=bundle / "runs/t0_semantic_deepfill_v2_latest_t0_pilot")
    parser.add_argument("--retrieval-root", type=Path, default=bundle / "runs/t0_semantic_reference_diagnostic_v1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    audit = args.bundle / "runs/t0_highres_source_audit_edge_all" / args.pose / args.camera
    retrieval = args.retrieval_root / args.pose / args.camera
    output = args.output / args.pose / args.camera
    output.mkdir(parents=True, exist_ok=True)
    original = np.asarray(Image.open(audit / "t0_highres_blue.png").convert("RGB"))
    prefilled = np.asarray(Image.open(retrieval / "t0_semantic_recovered_blue_residual.png").convert("RGB"))
    retrieved = np.asarray(Image.open(retrieval / "retrieved_source_rgb.png").convert("RGB"))
    residual = np.asarray(Image.open(retrieval / "residual_true_hole_mask.png").convert("L")) > 127
    target_depth = np.load(retrieval / "target_depth_condition.npy", mmap_mode="r").astype(np.float32)
    height, width = residual.shape
    low_size = (512, 288)
    low_prefilled = cv2.resize(prefilled, low_size, interpolation=cv2.INTER_AREA)
    low_retrieved = cv2.resize(retrieved, low_size, interpolation=cv2.INTER_AREA)
    low_residual = cv2.resize(residual.astype(np.uint8), low_size, interpolation=cv2.INTER_NEAREST) > 0
    low_prefilled[low_residual] = 0
    semantic_query = low_prefilled.copy()
    semantic_query[low_residual] = low_retrieved[low_residual]
    device = torch.device(args.device)
    descriptor = FrozenDinoAnyUpDescriptor(
        model_name="dinov2_vitl14", output_channels=64,
        anyup_checkpoint=args.bundle / "outputs/checkpoints/anyup_multi_backbone.pth",
        anyup_root=args.bundle / "third_party/anyup",
    ).to(device).eval()
    query_tensor = _tensor(semantic_query, device)
    with torch.inference_mode():
        semantic = descriptor(query_tensor, (low_size[1], low_size[0]))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SemanticDeepFillV2(64, 5, int(state["config"]["base_channels"])).to(device).eval()
    model.load_state_dict(state["model"])
    hole = torch.from_numpy(low_residual[None, None].astype(np.float32)).to(device)
    yy, xx = np.indices(low_residual.shape, dtype=np.float32)
    low_depth = cv2.resize(target_depth, low_size, interpolation=cv2.INTER_AREA)
    normalized_depth = np.log1p(np.clip(low_depth, 0.0, 160.0)) / np.log(161.0)
    observed_distance = cv2.distanceTransform(low_residual.astype(np.uint8), cv2.DIST_L2, 5)
    depth_confidence = np.exp(-observed_distance / 24.0).astype(np.float32)
    geometry = np.stack([
        normalized_depth.astype(np.float32), depth_confidence,
        1.0 - low_residual.astype(np.float32),
        xx / (low_size[0] - 1) * 2.0 - 1.0, yy / (low_size[1] - 1) * 2.0 - 1.0,
    ]).astype(np.float32)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        result = model(
            _tensor(low_prefilled, device), hole, _tensor(low_retrieved, device),
            torch.ones_like(hole) * 0.75, semantic,
            torch.from_numpy(geometry[None]).to(device),
        )
    proposal = np.clip(result["proposal"][0].float().cpu().numpy().transpose(1, 2, 0), 0, 1)
    proposal_hd = cv2.resize(proposal, (width, height), interpolation=cv2.INTER_LANCZOS4)
    final = prefilled.copy()
    final[residual] = np.rint(proposal_hd[residual] * 255).astype(np.uint8)
    original_validity = np.load(audit / "validity.npy").astype(bool)
    observed_difference = np.abs(final[original_validity].astype(np.int16) - original[original_validity].astype(np.int16))
    recovered = (~original_validity) & ~residual
    recovered_difference = np.abs(final[recovered].astype(np.int16) - prefilled[recovered].astype(np.int16))
    Image.fromarray(final).save(output / "t0_semantic_deepfill_v2_pilot.png")
    report = {
        "checkpoint": str(args.checkpoint), "pose": args.pose, "camera": args.camera,
        "resolution": [width, height], "generator_resolution": list(low_size),
        "original_observed_max_abs_diff": int(observed_difference.max(initial=0)),
        "semantic_recovered_max_abs_diff": int(recovered_difference.max(initial=0)),
        "generated_fraction": float(residual.mean()),
        "conditioning": {
            "rgb": "latest T0 plus depth-gated semantic source retrieval",
            "depth": "T0 projected target depth with nearest-surface residual extrapolation",
            "semantic": "DINOv2-L + AnyUP from available/retrieved RGB only",
            "geometry_was_zero": False,
        },
        "input_provenance": {
            "t0_rgb": str(audit / "t0_highres_blue.png"),
            "t0_rgb_sha256": _sha256(audit / "t0_highres_blue.png"),
            "t0_validity": str(audit / "validity.npy"),
            "t0_validity_sha256": _sha256(audit / "validity.npy"),
            "t0_report": str(audit / "report.json"),
            "t0_report_sha256": _sha256(audit / "report.json"),
            "audit_summary": str(args.bundle / "runs/t0_highres_source_audit_edge_all/summary.json"),
            "audit_summary_sha256": _sha256(args.bundle / "runs/t0_highres_source_audit_edge_all/summary.json"),
            "structure_checkpoint": str(args.bundle / "checkpoints/t0_structure_rectified_v2_dinob14_320x180_c48_best.pt"),
            "structure_checkpoint_sha256": _sha256(args.bundle / "checkpoints/t0_structure_rectified_v2_dinob14_320x180_c48_best.pt"),
        },
        "status": "pilot only; not promoted to the T0 release",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
