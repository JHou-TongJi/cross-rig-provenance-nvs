import numpy as np
import torch

from gcr_nvs.datasets.sparse_geometry_dataset import SparseGridSpec
from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.geometry.camera import TargetCamera
from gcr_nvs.models.multiview_voxel_fusion import VoxelAppearance
from gcr_nvs.rendering.environment import EnvironmentRenderResult
from gcr_nvs.rendering.sparse_geometry_raymarch import SparseRaymarchResult
from gcr_nvs.rendering import unified_field_raymarch as renderer


class _FakeField:
    geometry = object()

    @staticmethod
    def encode_sources(images):
        return torch.zeros(images.shape[0], images.shape[1], 4, *images.shape[-2:])

    @staticmethod
    def fuse_at_points(geometry_features, voxel_xyz, source_features, source_images,
                       source_intrinsics, source_extrinsics, source_distortions, **kwargs):
        batch, points, _ = voxel_xyz.shape
        valid = torch.tensor([[True, False]], device=voxel_xyz.device)[:, :points]
        weights = voxel_xyz.new_zeros(batch, points, source_images.shape[1])
        weights[..., 0] = valid
        rgb = voxel_xyz.new_zeros(batch, points, 3)
        rgb[..., 0] = 1.0
        appearance = VoxelAppearance(
            features=voxel_xyz.new_zeros(batch, points, 4),
            rgb=rgb,
            validity=valid,
            confidence=valid[..., None].to(voxel_xyz.dtype),
            view_weights=weights,
            projected_uv=voxel_xyz.new_zeros(batch, source_images.shape[1], points, 2),
            projected_depth=voxel_xyz.new_ones(batch, source_images.shape[1], points),
            projection_validity=torch.ones(
                batch, source_images.shape[1], points,
                dtype=torch.bool, device=voxel_xyz.device,
            ),
            depth_consistency=voxel_xyz.new_ones(batch, source_images.shape[1], points),
        )
        return appearance, torch.where(valid[..., None], rgb, torch.zeros_like(rgb))


class _FakeGenerativeField(_FakeField):
    seen_source_validity = None

    @classmethod
    def generate_target_view(cls, **kwargs):
        cls.seen_source_validity = kwargs["source_validity"].detach().clone()
        source_rgb = kwargs["source_rgb"]
        return {
            "rgb": source_rgb,
            "alpha": torch.zeros_like(kwargs["source_validity"]),
            "logvar": torch.zeros_like(kwargs["source_validity"]),
        }


def _geometry_result():
    structure = torch.tensor([[True, False], [True, False]])
    return SparseRaymarchResult(
        depth=torch.tensor([[5.0, 0.0], [7.0, 0.0]]),
        surface_points=torch.tensor([
            [[0.0, 0.0, 5.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 7.0], [0.0, 0.0, 0.0]],
        ]),
        occupancy=structure.float(),
        confidence=structure.float(),
        normal=torch.zeros(2, 2, 3),
        surface_features=torch.zeros(2, 2, 4),
        structure_validity=structure,
        query_coverage=structure.float(),
        occupied_support=structure.float(),
        unknown_structure_mask=~structure,
    )


def test_environment_fills_only_unknown_structure(monkeypatch):
    monkeypatch.setattr(renderer, "raymarch_sparse_geometry", lambda *args, **kwargs: _geometry_result())
    calibration = CameraCalibration(
        "target", np.eye(3), np.zeros(5), np.eye(4), 2, 2,
    )
    environment = EnvironmentRenderResult(
        rgb=np.full((2, 2, 3), [0.0, 0.0, 1.0], dtype=np.float32),
        validity=np.ones((2, 2), dtype=bool),
        source_count=np.ones((2, 2), dtype=np.uint8),
        source_provenance=np.full((2, 2), 3, dtype=np.int16),
    )
    result = renderer.raymarch_unified_field(
        _FakeField(),
        torch.zeros(1, 7),
        torch.zeros(1, 4, dtype=torch.int32),
        (1, 1, 1),
        TargetCamera(calibration),
        SparseGridSpec(),
        source_images=torch.zeros(1, 1, 3, 2, 2),
        source_intrinsics=torch.eye(3)[None, None],
        source_extrinsics=torch.eye(4)[None, None],
        source_distortions=torch.zeros(1, 1, 5),
        width=2,
        height=2,
        environment=environment,
    )
    assert torch.equal(result.surface_validity, torch.tensor([[True, False], [False, False]]))
    assert torch.equal(result.true_disocclusion, torch.tensor([[False, False], [True, False]]))
    assert torch.equal(result.environment_validity, torch.tensor([[False, True], [False, True]]))
    assert torch.allclose(result.rgb[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(result.rgb[0, 1], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(result.rgb[1, 0], torch.zeros(3))
    assert not result.output_validity[1, 0]


def test_fallback_rgb_is_an_appearance_anchor_for_generator(monkeypatch):
    monkeypatch.setattr(renderer, "raymarch_sparse_geometry", lambda *args, **kwargs: _geometry_result())
    calibration = CameraCalibration(
        "target", np.eye(3), np.zeros(5), np.eye(4), 2, 2,
    )
    fallback = EnvironmentRenderResult(
        rgb=np.full((2, 2, 3), [0.2, 0.4, 0.6], dtype=np.float32),
        validity=np.ones((2, 2), dtype=bool),
        source_count=np.ones((2, 2), dtype=np.uint8),
        source_provenance=np.full((2, 2), 1, dtype=np.int16),
    )
    result = renderer.raymarch_unified_field(
        _FakeGenerativeField(),
        torch.zeros(1, 7),
        torch.zeros(1, 4, dtype=torch.int32),
        (1, 1, 1),
        TargetCamera(calibration),
        SparseGridSpec(),
        source_images=torch.zeros(1, 1, 3, 2, 2),
        source_intrinsics=torch.eye(3)[None, None],
        source_extrinsics=torch.eye(4)[None, None],
        source_distortions=torch.zeros(1, 1, 5),
        width=2,
        height=2,
        fallback_environment=fallback,
    )
    # The geometry hit remains a surface anchor and the unknown pixel uses
    # the directional fallback instead of being treated as free generation.
    # The geometry hit keeps the fake field's measured surface color; the
    # fallback anchor is used for the unknown pixel.
    assert torch.equal(_FakeGenerativeField.seen_source_validity[0, 0], torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
    assert torch.allclose(result.rgb[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(result.rgb[0, 1], torch.tensor([0.2, 0.4, 0.6]))
    assert torch.all(result.output_validity)
