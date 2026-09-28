"""Frozen monocular depth teacher used to densify calibrated RGB geometry.

The teacher is deliberately kept outside the trainable RGB generator.  It
produces a dense candidate surface, while LiDAR remains the metric anchor and
the only hard source of measured depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


@dataclass(frozen=True)
class MonocularDepthPrediction:
    depth: np.ndarray
    confidence: np.ndarray
    is_metric: bool


class DepthAnything3Estimator:
    """Lazy local DA3 inference wrapper.

    Importing this module does not require the optional DA3 package.  This is
    important for the geometry and unit-test paths, which should remain light.
    """

    def __init__(
        self,
        model_dir: Path,
        device: str = "cuda",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        source_root: Path | None = Path("/home/heqing/Depth-Anything-3"),
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.process_res = int(process_res)
        self.process_res_method = process_res_method
        self.source_root = None if source_root is None else Path(source_root)
        self._model = None

    def _load(self):
        if self._model is None:
            # DA3's optional visualization/export stack depends on einops.
            # Reuse local pure-Python packages without importing incompatible
            # binary extensions from those environments.
            fallback_sites = (
                Path("/home/heqing/venvs/l4se-inpaint/lib/python3.10/site-packages"),
                Path("/home/heqing/anaconda3/envs/comfyui/lib/python3.10/site-packages"),
                Path("/home/heqing/anaconda3/envs/xiaohe2_cu128/lib/python3.10/site-packages"),
            )
            for fallback_site in fallback_sites:
                if fallback_site.exists() and str(fallback_site) not in sys.path:
                    sys.path.append(str(fallback_site))
            sys.modules.setdefault("xformers", None)
            if self.source_root is not None:
                source_path = self.source_root / "src"
                if source_path.exists() and str(source_path) not in sys.path:
                    sys.path.insert(0, str(source_path))
            try:
                from depth_anything_3.api import DepthAnything3
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise RuntimeError(
                    "Depth Anything 3 is not importable. Set source_root to the "
                    "local repository (for example /home/heqing/Depth-Anything-3)"
                ) from exc
            self._model = DepthAnything3.from_pretrained(str(self.model_dir))
            self._model = self._model.to(self.device).eval()
        return self._model

    @staticmethod
    def _resize_prediction(value: np.ndarray | None, size: tuple[int, int], default: float) -> np.ndarray:
        width, height = map(int, size)
        if value is None:
            return np.full((height, width), default, dtype=np.float32)
        import cv2

        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"DA3 prediction must be [H,W], got {array.shape}")
        if array.shape[::-1] == (width, height):
            return array
        return cv2.resize(array, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    def predict(
        self,
        image_path: Path,
        intrinsics: np.ndarray,
        output_size: tuple[int, int] | None = None,
    ) -> MonocularDepthPrediction:
        """Predict depth for an already rectified RGB image.

        DA3 may internally resize the image.  Predictions are explicitly
        returned to the caller's rectified image grid and never silently mixed
        with the raw Brown-distorted calibration.
        """
        import torch

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        K = np.asarray(intrinsics, dtype=np.float32)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError("intrinsics must be finite [3,3]")
        from PIL import Image

        with Image.open(image_path) as image:
            native_size = (int(image.width), int(image.height))
        output_size = output_size or native_size
        model = self._load()
        with torch.inference_mode():
            prediction = model.inference(
                [str(image_path)],
                intrinsics=np.asarray([K], dtype=np.float32),
                process_res=self.process_res,
                process_res_method=self.process_res_method,
            )
        raw_depth = np.asarray(prediction.depth[0], dtype=np.float32)
        raw_conf = None if prediction.conf is None else np.asarray(prediction.conf[0], dtype=np.float32)
        depth = self._resize_prediction(raw_depth, output_size, 0.0)
        confidence_was_missing = raw_conf is None
        confidence = self._resize_prediction(raw_conf, output_size, 1.0)
        confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
        valid_conf = confidence[np.isfinite(confidence)]
        if valid_conf.size and not confidence_was_missing:
            lo, hi = np.percentile(valid_conf, [5.0, 95.0])
            confidence = np.clip((confidence - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth <= 0.0] = 0.0
        metric_flag = prediction.is_metric
        if isinstance(metric_flag, dict) or metric_flag.__class__.__name__ == "Dict":
            # Older DA3 output processors wrap the scalar in an empty addict
            # Dict.  The checkpoint name is authoritative in that case.
            metric_flag = "metric" in self.model_dir.name.lower()
        elif hasattr(metric_flag, "detach"):
            metric_flag = metric_flag.detach().cpu().reshape(-1)[0].item()
        elif isinstance(metric_flag, (list, tuple, np.ndarray)):
            metric_flag = np.asarray(metric_flag, dtype=bool).reshape(-1)[0]
        is_metric = bool(metric_flag)
        return MonocularDepthPrediction(depth, confidence.astype(np.float32), is_metric)
