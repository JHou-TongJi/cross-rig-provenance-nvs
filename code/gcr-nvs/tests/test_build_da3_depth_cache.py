from pathlib import Path

import numpy as np

from gcr_nvs.evaluation.build_da3_depth_cache import build_frame


def test_cache_builder_rejects_unknown_camera(tmp_path: Path):
    try:
        build_frame(
            root=tmp_path,
            sequence="missing",
            frame_id=1,
            cameras=("CAM_UNKNOWN",),
            output_root=tmp_path / "out",
            distortion=tmp_path / "dist.yaml",
            model_dir=tmp_path / "model",
            source_root=tmp_path / "source",
            device="cpu",
            output_size=(32, 24),
        )
    except ValueError as error:
        assert "unknown cameras" in str(error)
    else:
        raise AssertionError("unknown camera must be rejected before filesystem access")

