from scripts.select_video_keyframes import keyframe_score, select_keyframes


def test_real_frames_are_retained_and_virtual_rows_are_marked_non_metric():
    candidates = [
        {
            "frame_id": 1,
            "timestamp_s": 0.0,
            "synthetic": False,
            "pose_novelty": 1.0,
            "visibility_novelty": 1.0,
            "semantic_novelty": 1.0,
            "lidar_coverage": 1.0,
            "quality": 1.0,
        },
        {
            "frame_id": 2,
            "timestamp_s": 0.5,
            "synthetic": False,
            "pose_novelty": 0.0,
            "visibility_novelty": 0.0,
            "semantic_novelty": 0.0,
            "lidar_coverage": 1.0,
            "quality": 1.0,
        },
        {
            "frame_id": "v1",
            "timestamp_s": 0.25,
            "synthetic": True,
            "pose_novelty": 1.0,
            "visibility_novelty": 1.0,
            "semantic_novelty": 1.0,
            "lidar_coverage": 0.0,
            "quality": 1.0,
        },
    ]
    selected = select_keyframes(candidates, total=3)
    assert {row["frame_id"] for row in selected} == {1, 2, "v1"}
    assert all(row["geometry_supervision"] == "metric_lidar" for row in selected if not row["synthetic"])
    assert next(row for row in selected if row["synthetic"])["geometry_supervision"] == "none_synthetic_appearance_only"


def test_dynamic_penalty_is_applied():
    clean = {"pose_novelty": 0.8, "visibility_novelty": 0.8, "semantic_novelty": 0.8, "lidar_coverage": 0.8, "quality": 0.8}
    dynamic = dict(clean, dynamic_ratio=1.0)
    assert keyframe_score(clean) > keyframe_score(dynamic)
