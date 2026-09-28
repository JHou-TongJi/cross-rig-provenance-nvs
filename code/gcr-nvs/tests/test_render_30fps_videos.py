import numpy as np

from scripts.render_30fps_videos import _flow_interpolator, _motion_interpolate


def test_motion_interpolate_preserves_shape_and_range():
    left = np.zeros((48, 64, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    left[16:32, 12:28] = (20, 120, 220)
    right[16:32, 20:36] = (20, 120, 220)
    engine = _flow_interpolator("ultrafast")
    output, report = _motion_interpolate(left, right, 0.5, engine)
    assert output.shape == left.shape
    assert output.dtype == np.uint8
    assert int(output.min()) >= 0 and int(output.max()) <= 255
    assert report["fallback"] is False
