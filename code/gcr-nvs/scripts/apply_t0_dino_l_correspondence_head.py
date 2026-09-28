"""Apply the trained correspondence confidence head to one audit sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from gcr_nvs.models.semantic_correspondence_head import SemanticCorrespondenceHead


BLUE = np.asarray([25, 90, 235], dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-case", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-rgb", type=Path, required=True)
    parser.add_argument("--input-validity", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    case = args.audit_case
    features = np.load(case / "correspondence_features.npy").astype(np.float32)
    oracle = np.load(case / "correspondence_oracle.npy").astype(bool)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    model = SemanticCorrespondenceHead(input_dim=features.shape[-1], hidden_dim=32).to(args.device).eval()
    model.load_state_dict(state["model"])
    with torch.inference_mode():
        probabilities = model(torch.from_numpy(features.reshape(-1, features.shape[-1])).to(args.device)).sigmoid()
    probabilities = probabilities.cpu().numpy().reshape(features.shape[:2])
    hard = (
        (features[..., 0] >= 0.68)
        & (features[..., 1] <= 2.5 / 6.0)
        & (features[..., 2] <= 0.30)
        & (features[..., 3] <= 1.0)
        & (features[..., 4] > 0.5)
        & (features[..., 5] > 0.5)
    )
    accepted_low = hard & (probabilities >= float(args.threshold))
    validity = np.load(args.input_validity).astype(bool)
    original = np.asarray(Image.open(args.input_rgb).convert("RGB"))
    retrieved = np.asarray(Image.open(case / "retrieved_source_rgb.png").convert("RGB"))
    height, width = validity.shape
    accepted = cv2.resize(accepted_low.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    accepted &= ~validity
    final = original.copy()
    final[accepted] = retrieved[accepted]
    final[~(validity | accepted)] = BLUE
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(args.output / "t0_dino_l_head_recovered_blue_residual.png")
    Image.fromarray((accepted.astype(np.uint8) * 255), mode="L").save(args.output / "head_recoverable_mask.png")
    Image.fromarray(np.uint8(np.clip(probabilities, 0, 1) * 255), mode="L").resize((width, height), Image.Resampling.NEAREST).save(args.output / "head_probability.png")
    observed_modified = int(np.any(final[validity] != original[validity], axis=1).sum())
    tp = int((accepted_low & oracle).sum())
    fp = int((accepted_low & ~oracle).sum())
    fn = int((~accepted_low & oracle).sum())
    report = {
        "checkpoint": str(args.checkpoint), "threshold": args.threshold,
        "input_case": str(case), "resolution": [width, height],
        "low_resolution": list(features.shape[:2]),
        "hard_candidate_fraction_of_low_hole": float(hard.sum() / max(int((features[..., 5] > 0.5).sum()), 1)),
        "head_recovered_fraction_of_low_hole": float(accepted_low.sum() / max(int((features[..., 5] > 0.5).sum()), 1)),
        "oracle_precision": tp / max(tp + fp, 1),
        "oracle_recall": tp / max(tp + fn, 1),
        "observed_pixels_modified": observed_modified,
        "status": "pilot; oracle is target RGB supervision from the same audit frame",
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
