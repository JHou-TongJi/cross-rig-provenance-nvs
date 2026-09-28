"""Optional COCO instance-mask provider for dynamic-object pseudo labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


class DynamicMaskProvider:
    """Run a pretrained instance model and retain moving-object categories."""

    dynamic_labels = {1, 2, 3, 4, 6, 8}  # person, bicycle, car, motorcycle, bus, truck

    def __init__(self, device: str | None = None, score_threshold: float = 0.5):
        import torchvision

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).to(self.device)
        output = self.model([tensor])[0]
        keep = (output["scores"] >= self.score_threshold).cpu().numpy()
        labels = output["labels"].cpu().numpy()
        masks = output["masks"].squeeze(1).cpu().numpy() > 0.5
        dynamic = np.zeros(image.shape[:2], dtype=bool)
        instance = np.zeros(image.shape[:2], dtype=np.int32)
        for instance_id, (mask, label, selected) in enumerate(zip(masks, labels, keep), start=1):
            if selected and int(label) in self.dynamic_labels:
                dynamic |= mask
                instance[mask] = instance_id
        return dynamic, instance, labels[keep]


def mask_image_file(input_path: Path, output_path: Path, device: str | None = None) -> None:
    image = np.asarray(Image.open(input_path).convert("RGB"))
    dynamic, instance, _ = DynamicMaskProvider(device=device)(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path.with_suffix(".dynamic.npy"), dynamic)
    np.save(output_path.with_suffix(".instance.npy"), instance)
