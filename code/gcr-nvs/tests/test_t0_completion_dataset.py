import numpy as np
import torch

from gcr_nvs.datasets.t0_completion import T0SemanticCompletionDataset


def test_retrieval_proxy_and_geometry_inputs_remain_float32() -> None:
    rgb = np.full((24, 32, 3), 127, dtype=np.uint8)
    proxy, confidence = T0SemanticCompletionDataset._retrieval_proxy(
        rgb, np.random.default_rng(3),
    )
    geometry = np.stack([
        np.ones((24, 32), np.float32), confidence,
        np.ones((24, 32), np.float32), np.zeros((24, 32), np.float32),
        np.zeros((24, 32), np.float32),
    ]).astype(np.float32)
    assert proxy.dtype == np.uint8
    assert confidence.dtype == np.float32
    assert torch.from_numpy(geometry).dtype == torch.float32
