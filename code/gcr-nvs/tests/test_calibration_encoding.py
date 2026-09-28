from pathlib import Path

import numpy as np

from gcr_nvs.geometry.calibration import load_distortion


def test_load_distortion_reads_utf8_comments(tmp_path: Path) -> None:
    config = tmp_path / "camera_intric.yaml"
    config.write_text(
        "# 相机去畸变参数\n"
        "camera_image_0:\n"
        "  D_manual: [0.1, -0.2, 0.003, -0.004, 0.01]\n",
        encoding="utf-8",
    )

    distortion = load_distortion(config)

    np.testing.assert_allclose(
        distortion[0],
        np.asarray([0.1, -0.2, 0.003, -0.004, 0.01], dtype=np.float64),
    )
