import pytest

from gcr_nvs.datasets.manifest import appearance_source_cameras


@pytest.mark.parametrize(
    ("target", "expected"),
    (
        ("CAM_BACK", ("CAM_BACK_LEFT", "CAM_BACK_RIGHT")),
        ("CAM_BACK_LEFT", ("CAM_BACK",)),
        ("CAM_BACK_RIGHT", ("CAM_BACK",)),
        ("CAM_FRONT_LEFT", ("CAM_FRONT_WIDE",)),
        ("CAM_FRONT_NARROW", ("CAM_FRONT_WIDE",)),
        ("CAM_FRONT_RIGHT", ("CAM_FRONT_WIDE",)),
        (
            "CAM_FRONT_WIDE",
            ("CAM_FRONT_NARROW", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"),
        ),
    ),
)
def test_strict_loo_uses_only_topology_neighbors(target, expected):
    assert appearance_source_cameras(target, include_target=False) == expected


def test_shifted_view_puts_current_same_camera_first():
    sources = appearance_source_cameras(
        "CAM_FRONT_LEFT", include_target=True,
    )
    assert sources == ("CAM_FRONT_LEFT", "CAM_FRONT_WIDE")
    assert not any("BACK" in source for source in sources)


def test_front_and_rear_appearance_families_never_mix():
    for target in ("CAM_FRONT_LEFT", "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE"):
        assert not any(
            "BACK" in source
            for source in appearance_source_cameras(target, include_target=True)
        )
    for target in ("CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"):
        assert not any(
            "FRONT" in source
            for source in appearance_source_cameras(target, include_target=True)
        )
