import cv2
import numpy as np
import torch
from pathlib import Path

from gcr_nvs.datasets.unified_3d_field_dataset import temporal_camera_in_center_frame
from gcr_nvs.geometry.calibration import (
    CameraCalibration,
    project_world_points,
    rectified_calibration,
)
from gcr_nvs.models.multiview_voxel_fusion import (
    MultiViewVoxelFusion,
    brown_distort,
    project_voxels,
)
from gcr_nvs.models.feature_upsampler import FeatureUpsampler
from gcr_nvs.models.unified_3d_field import Unified3DField


def test_rectified_projection_matches_opencv_undistort_points():
    calibration = CameraCalibration(
        name="test",
        intrinsic=np.asarray([
            [820.0, 0.0, 640.0],
            [0.0, 815.0, 360.0],
            [0.0, 0.0, 1.0],
        ]),
        distortion=np.asarray([-0.28, 0.08, 0.001, -0.0005, 0.0]),
        external=np.eye(4),
        width=1280,
        height=720,
    )
    rectified = rectified_calibration(calibration, (960, 540), alpha=0.0)
    points = np.asarray([
        [-2.0, -1.0, 8.0],
        [0.0, 0.0, 10.0],
        [2.0, 0.8, 12.0],
    ])
    raw_uv, raw_valid = project_world_points(points, calibration)
    expected = cv2.undistortPoints(
        raw_uv[:, None].astype(np.float64),
        calibration.intrinsic,
        calibration.distortion,
        P=rectified.intrinsic,
    )[:, 0]
    actual, actual_valid = project_world_points(points, rectified)
    assert raw_valid.all() and actual_valid.all()
    assert np.allclose(actual, expected, atol=1e-4)
    assert np.count_nonzero(rectified.distortion) == 0


def test_brown_projection_is_not_applied_after_rectification():
    xy = torch.tensor([[[[0.3, -0.2]]]], dtype=torch.float32)
    zero = torch.zeros(1, 1, 1, 5)
    assert torch.equal(brown_distort(xy, zero), xy)

    xyz = torch.tensor([[[1.0, 0.0, 5.0]]])
    intrinsic = torch.tensor([[[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]]])
    extrinsic = torch.eye(4)[None, None]
    uv, depth, valid = project_voxels(
        xyz, intrinsic, extrinsic, torch.zeros(1, 1, 5), (100, 120),
    )
    assert torch.allclose(uv[0, 0, 0], torch.tensor([70.0, 40.0]))
    assert depth.item() == 5.0
    assert valid.item()


def test_temporal_camera_motion_is_applied_exactly_once():
    calibration = CameraCalibration(
        name="neighbor",
        intrinsic=np.asarray([
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 40.0],
            [0.0, 0.0, 1.0],
        ]),
        distortion=np.zeros(5),
        external=np.eye(4),
        width=120,
        height=100,
    )
    center_from_neighbor = np.eye(4)
    center_from_neighbor[0, 3] = 1.0
    center_point = np.asarray([[0.0, 0.0, 5.0]])
    neighbor_from_center = np.linalg.inv(center_from_neighbor)
    neighbor_point = (
        neighbor_from_center @ np.c_[center_point, np.ones(1)].T
    ).T[:, :3]

    temporal = temporal_camera_in_center_frame(
        calibration, center_from_neighbor,
    )
    projected_from_center, valid_from_center = project_world_points(
        center_point, temporal,
    )
    projected_in_neighbor, valid_in_neighbor = project_world_points(
        neighbor_point, calibration,
    )
    double_transformed, _ = project_world_points(neighbor_point, temporal)

    assert valid_from_center.all() and valid_in_neighbor.all()
    assert np.allclose(projected_from_center, projected_in_neighbor)
    assert not np.allclose(projected_from_center, double_transformed)


def test_depth_consistency_gate_rejects_an_occluding_view():
    fusion = MultiViewVoxelFusion(
        geometry_dim=4,
        appearance_dim=4,
        hidden_dim=8,
        heads=2,
        unknown_depth_prior=0.35,
    )
    for parameter in fusion.attention.parameters():
        torch.nn.init.zeros_(parameter)
    geometry = torch.zeros(1, 1, 4)
    xyz = torch.tensor([[[0.0, 0.0, 5.0]]])
    features = torch.zeros(1, 2, 4, 4, 4)
    rgb = torch.zeros(1, 2, 3, 4, 4)
    rgb[:, 0, 0] = 1.0
    rgb[:, 1, 2] = 1.0
    intrinsics = torch.tensor([[
        [[1.0, 0.0, 1.5], [0.0, 1.0, 1.5], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 1.5], [0.0, 1.0, 1.5], [0.0, 0.0, 1.0]],
    ]])
    extrinsics = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
    lidar_depth = torch.zeros(1, 2, 1, 4, 4)
    lidar_depth[:, 0] = 5.0
    lidar_depth[:, 1] = 1.0
    result = fusion(
        geometry,
        xyz,
        features,
        rgb,
        intrinsics,
        extrinsics,
        torch.zeros(1, 2, 5),
        source_lidar_depth=lidar_depth,
    )
    assert result.validity.item()
    assert result.view_weights[0, 0, 0] > 0.999
    assert result.rgb[0, 0, 0] > 0.999
    assert result.rgb[0, 0, 2] < 1e-4


def test_bilinear_feature_upsampler_contract():
    image = torch.randn(1, 3, 8, 12)
    features = torch.randn(1, 5, 2, 3)
    output = FeatureUpsampler("bilinear")(image, features, (8, 12))
    assert output.shape == (1, 5, 8, 12)
    assert torch.isfinite(output).all()


def test_anyup_feature_upsampler_checkpoint_contract():
    checkpoint = Path("outputs/checkpoints/anyup_multi_backbone.pth")
    source = Path("third_party/anyup")
    if not checkpoint.exists() or not source.exists():
        return
    upsampler = FeatureUpsampler(
        "anyup",
        checkpoint=checkpoint,
        anyup_root=source,
        q_chunk_size=32,
    ).eval()
    image = torch.randn(1, 3, 8, 12)
    features = torch.randn(1, 6, 2, 3)
    with torch.no_grad():
        output = upsampler(image, features, (8, 12))
    assert output.shape == (1, 6, 8, 12)
    assert torch.isfinite(output).all()


def test_camera_conditioned_residual_starts_as_no_op():
    class _Geometry(torch.nn.Module):
        pass

    class _Encoder(torch.nn.Module):
        def forward(self, images):
            return images.new_zeros(images.shape[0], images.shape[1], 4, *images.shape[-2:])

    model = Unified3DField(
        geometry_channels=(4, 8, 16, 32),
        appearance_dim=4,
        geometry_model=_Geometry(),
        image_encoder=_Encoder(),
    )
    assert torch.count_nonzero(model.camera_rgb_residual[-2].weight) == 0
    assert torch.count_nonzero(model.camera_rgb_residual[-2].bias) == 0
    assert torch.count_nonzero(model.camera_color_affine.weight) == 0
