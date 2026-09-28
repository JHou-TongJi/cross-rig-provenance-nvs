"""Apply a trained view weighter to a frozen appearance cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gcr_nvs.models.lidar_appearance import LearnedViewWeighter


def _psnr(prediction, target, valid) -> float:
    mse = F.mse_loss(prediction[valid], target[valid]).clamp(min=1e-12)
    return float(-10.0 * torch.log10(mse))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("appearance", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LearnedViewWeighter(feature_dim=int(state["feature_dim"])).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    with np.load(args.appearance) as payload:
        arrays = {key: payload[key] for key in payload.files}
    features = torch.from_numpy(arrays["per_source_features"]).float().to(device)
    rgb = torch.from_numpy(arrays["per_source_rgb"]).float().to(device)
    valid = torch.from_numpy(arrays["per_source_validity"]).bool().to(device)
    scores = torch.from_numpy(arrays["per_source_scores"]).float().to(device)
    target = torch.from_numpy(arrays["target_point_rgb"]).float().to(device)
    baseline = torch.from_numpy(arrays["point_rgb"]).float().to(device)
    with torch.inference_mode():
        learned, weights = model(
            features[None], rgb[None], valid[None], scores[None],
        )
    learned = learned[0]
    observed = valid.any(dim=-1)
    report = {
        "checkpoint": str(args.checkpoint),
        "appearance": str(args.appearance),
        "surface_count": int(observed.sum()),
        "baseline_point_psnr_db": _psnr(baseline, target, observed),
        "learned_point_psnr_db": _psnr(learned, target, observed),
    }
    arrays["point_rgb"] = learned.cpu().numpy()
    arrays["learned_view_weights"] = weights[0].cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
