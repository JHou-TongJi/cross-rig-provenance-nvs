"""Semantic dynamic masks for safe temporal LiDAR fusion."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gcr_nvs.geometry.calibration import CameraCalibration, project_world_points


DYNAMIC_CITYSCAPES_LABELS = frozenset({
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
})


class SegFormerDynamicMasker:
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        device: str = "cuda",
        dilation_pixels: int = 5,
        local_files_only: bool = True,
    ) -> None:
        import torch
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        self.torch = torch
        self.device = torch.device(device)
        self.processor = SegformerImageProcessor.from_pretrained(
            model_name, local_files_only=local_files_only,
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, local_files_only=local_files_only,
        ).to(self.device).eval()
        self.dynamic_ids = {
            int(index)
            for index, label in self.model.config.id2label.items()
            if label.lower() in DYNAMIC_CITYSCAPES_LABELS
        }
        self.label_to_id = {
            str(label).lower(): int(index)
            for index, label in self.model.config.id2label.items()
        }
        self.dilation_pixels = int(dilation_pixels)

    def predict_labels(
        self,
        image: Image.Image | np.ndarray | Path,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        if isinstance(image, Path):
            with Image.open(image) as source:
                rgb = source.convert("RGB")
        elif isinstance(image, np.ndarray):
            rgb = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        else:
            rgb = image.convert("RGB")
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        target_height, target_width = (
            (rgb.height, rgb.width)
            if output_size is None
            else (int(output_size[1]), int(output_size[0]))
        )
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        return self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(target_height, target_width)],
        )[0].cpu().numpy()

    def predict_label_mask(
        self,
        image: Image.Image | np.ndarray | Path,
        labels: set[str] | frozenset[str],
        output_size: tuple[int, int] | None = None,
        dilation_pixels: int = 0,
    ) -> np.ndarray:
        segmentation = self.predict_labels(image, output_size=output_size)
        label_ids = [self.label_to_id[label.lower()] for label in labels]
        mask = np.isin(segmentation, label_ids)
        if dilation_pixels > 0:
            size = 2 * dilation_pixels + 1
            kernel = np.ones((size, size), dtype=np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
        return mask

    def predict(
        self,
        image: Image.Image | np.ndarray | Path,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        segmentation = self.predict_labels(image, output_size=output_size)
        mask = np.isin(segmentation, list(self.dynamic_ids))
        if self.dilation_pixels > 0:
            size = 2 * self.dilation_pixels + 1
            kernel = np.ones((size, size), dtype=np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
        return mask


def dynamic_points_from_camera_masks(
    points: np.ndarray,
    calibrations: dict[str, CameraCalibration],
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    """Mark a LiDAR point dynamic if any calibrated camera says it is dynamic."""
    dynamic = np.zeros(len(points), dtype=bool)
    for name, mask in masks.items():
        if name not in calibrations:
            raise KeyError(f"missing calibration for {name}")
        mask = np.asarray(mask, dtype=bool)
        calibration = calibrations[name]
        pixels, valid = project_world_points(points[:, :3], calibration)
        indices = np.flatnonzero(valid)
        if not len(indices):
            continue
        xy = np.rint(pixels[indices]).astype(np.int64)
        scale_x = mask.shape[1] / calibration.width
        scale_y = mask.shape[0] / calibration.height
        xy[:, 0] = np.rint(xy[:, 0] * scale_x).astype(np.int64)
        xy[:, 1] = np.rint(xy[:, 1] * scale_y).astype(np.int64)
        inside = (
            (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1])
            & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
        )
        visible_indices = indices[inside]
        visible_xy = xy[inside]
        dynamic[visible_indices] |= mask[visible_xy[:, 1], visible_xy[:, 0]]
    return dynamic
