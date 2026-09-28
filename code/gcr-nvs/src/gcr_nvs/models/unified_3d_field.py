"""Unified LiDAR-geometry and RGB-appearance field for novel-view rendering."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gcr_nvs.models.multiview_voxel_fusion import MultiViewVoxelFusion, VoxelAppearance
from gcr_nvs.models.source_encoder import SourceImageEncoder
from gcr_nvs.models.sparse_geometry import SparseGeometryStudent, hard_replace_exact
from gcr_nvs.models.target_view_generator import TargetViewGenerator


@dataclass(frozen=True)
class UnifiedFieldOutput:
    geometry: dict[str, torch.Tensor]
    appearance: VoxelAppearance
    field_rgb: torch.Tensor
    field_confidence: torch.Tensor
    structure_validity: torch.Tensor
    appearance_validity: torch.Tensor
    unknown_structure: torch.Tensor
    true_disocclusion: torch.Tensor


class Unified3DField(nn.Module):
    """Build one persistent 3D field shared by every requested target camera.

    Geometry heads consume LiDAR sparse voxels only. RGB features are queried
    at the resulting 3D locations and therefore cannot modify occupancy, SDF,
    normals, or measured free space.
    """

    def __init__(
        self,
        lidar_input_dim: int = 7,
        geometry_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        appearance_dim: int = 64,
        fusion_hidden_dim: int = 96,
        decoder_dim: int = 48,
        residual_hidden_dim: int = 96,
        use_dino: bool = True,
        dino_model_name: str = "dinov2_vitb14",
        dino_pretrained: bool = True,
        semantic_upsampling: str = "bilinear",
        anyup_checkpoint=None,
        anyup_root=None,
        anyup_q_chunk_size: int | None = 8192,
        max_observed_rgb_residual: float = 0.05,
        max_camera_rgb_residual: float = 0.03,
        camera_count: int = 7,
        camera_embedding_dim: int = 16,
        query_position_encoding: bool = True,
        target_camera_conditioning: bool = False,
        geometry_model: nn.Module | None = None,
        image_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry_model or SparseGeometryStudent(
            input_dim=lidar_input_dim,
            channels=geometry_channels,
            query_position_encoding=query_position_encoding,
        )
        self.image_encoder = image_encoder or SourceImageEncoder(
            feature_dim=appearance_dim,
            use_dino=use_dino,
            dino_model_name=dino_model_name,
            dino_pretrained=dino_pretrained,
            semantic_upsampling=semantic_upsampling,
            anyup_checkpoint=anyup_checkpoint,
            anyup_root=anyup_root,
            anyup_q_chunk_size=anyup_q_chunk_size,
        )
        geometry_dim = geometry_channels[0]
        self.voxel_fusion = MultiViewVoxelFusion(
            geometry_dim=geometry_dim,
            appearance_dim=appearance_dim,
            hidden_dim=fusion_hidden_dim,
        )
        self.target_view_generator = TargetViewGenerator(
            geometry_dim=geometry_dim,
            appearance_dim=appearance_dim,
            decoder_dim=decoder_dim,
            camera_count=camera_count,
            camera_conditioning=target_camera_conditioning,
        )
        self.rgb_residual = nn.Sequential(
            nn.Linear(geometry_dim + appearance_dim + 3, residual_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(residual_hidden_dim, 3),
            nn.Tanh(),
        )
        # Keep the original residual head checkpoint-compatible.  This second
        # branch learns camera/focal-role color response without allowing RGB
        # features to alter the LiDAR geometry field.
        self.unknown_camera_index = int(camera_count)
        self.target_camera_embedding = nn.Embedding(
            camera_count + 1, camera_embedding_dim,
        )
        conditioned_dim = geometry_dim + appearance_dim + 3 + camera_embedding_dim + 2
        self.camera_rgb_residual = nn.Sequential(
            nn.Linear(conditioned_dim, residual_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(residual_hidden_dim, 3),
            nn.Tanh(),
        )
        self.camera_rgb_gate = nn.Sequential(
            nn.Linear(conditioned_dim, max(16, residual_hidden_dim // 2)),
            nn.SiLU(inplace=True),
            nn.Linear(max(16, residual_hidden_dim // 2), 1),
            nn.Sigmoid(),
        )
        self.camera_color_affine = nn.Embedding(camera_count + 1, 6)
        nn.init.zeros_(self.camera_rgb_residual[-2].weight)
        nn.init.zeros_(self.camera_rgb_residual[-2].bias)
        nn.init.zeros_(self.camera_color_affine.weight)
        self.max_observed_rgb_residual = float(max_observed_rgb_residual)
        self.max_camera_rgb_residual = float(max_camera_rgb_residual)

    def encode_sources(self, source_images: torch.Tensor) -> torch.Tensor:
        """Encode source views without constructing target-view depth."""
        return self.image_encoder(source_images)

    def generate_target_view(self, **condition: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode a target image from rasterized 3D geometry and appearance."""
        return self.target_view_generator(**condition)

    @staticmethod
    def lock_measured_geometry(
        geometry: dict[str, torch.Tensor],
        exact_occupancy: torch.Tensor | None,
        exact_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if exact_occupancy is None or exact_mask is None:
            return geometry
        locked = dict(geometry)
        locked["occupancy"] = hard_replace_exact(
            geometry["occupancy"], exact_occupancy, exact_mask,
        )
        locked["sdf"] = hard_replace_exact(
            geometry["sdf"], torch.zeros_like(geometry["sdf"]), exact_mask,
        )
        return locked

    def fuse_at_points(
        self,
        geometry_features: torch.Tensor,
        voxel_xyz: torch.Tensor,
        source_features: torch.Tensor,
        source_images: torch.Tensor,
        source_intrinsics: torch.Tensor,
        source_extrinsics: torch.Tensor,
        source_distortions: torch.Tensor,
        source_rgb_images: torch.Tensor | None = None,
        source_rgb_intrinsics: torch.Tensor | None = None,
        source_rgb_distortions: torch.Tensor | None = None,
        source_lidar_depth: torch.Tensor | None = None,
        source_validity: torch.Tensor | None = None,
        source_time_offsets: torch.Tensor | None = None,
        source_alignment_confidence: torch.Tensor | None = None,
        focal_role: torch.Tensor | None = None,
        source_camera_indices: torch.Tensor | None = None,
        target_view_direction: torch.Tensor | None = None,
        target_camera_index: torch.Tensor | int | None = None,
        target_focal_role: torch.Tensor | None = None,
    ) -> tuple[VoxelAppearance, torch.Tensor]:
        appearance = self.voxel_fusion(
            geometry_features,
            voxel_xyz,
            source_features,
            source_images if source_rgb_images is None else source_rgb_images,
            source_intrinsics,
            source_extrinsics,
            source_distortions,
            source_rgb_intrinsics=source_rgb_intrinsics,
            source_rgb_distortions=source_rgb_distortions,
            source_lidar_depth=source_lidar_depth,
            source_validity=source_validity,
            source_time_offsets=source_time_offsets,
            source_alignment_confidence=source_alignment_confidence,
            focal_role=focal_role,
            source_camera_indices=source_camera_indices,
            target_camera_index=target_camera_index,
        )
        if target_view_direction is None:
            target_view_direction = torch.zeros_like(voxel_xyz)
        batch, points, _ = voxel_xyz.shape
        if target_camera_index is None:
            camera_index = torch.full(
                (batch,), self.unknown_camera_index,
                dtype=torch.long, device=voxel_xyz.device,
            )
        else:
            camera_index = torch.as_tensor(
                target_camera_index, dtype=torch.long, device=voxel_xyz.device,
            ).reshape(-1)
            if camera_index.numel() == 1 and batch > 1:
                camera_index = camera_index.expand(batch)
            if camera_index.numel() != batch:
                raise ValueError("target_camera_index must contain one index per batch")
        camera_index = camera_index.clamp(0, self.unknown_camera_index)
        camera_embedding = self.target_camera_embedding(camera_index)
        camera_embedding = camera_embedding[:, None].expand(-1, points, -1)
        if target_focal_role is None:
            target_focal_role = voxel_xyz.new_zeros(batch, 2)
        else:
            target_focal_role = target_focal_role.to(
                device=voxel_xyz.device, dtype=voxel_xyz.dtype,
            ).reshape(batch, 2)
        focal_condition = target_focal_role[:, None].expand(-1, points, -1)
        residual = self.rgb_residual(torch.cat([
            geometry_features, appearance.features, target_view_direction,
        ], dim=-1))
        condition = torch.cat([
            geometry_features,
            appearance.features,
            target_view_direction,
            camera_embedding,
            focal_condition,
        ], dim=-1)
        affine = self.camera_color_affine(camera_index)
        color_scale = 1.0 + 0.10 * torch.tanh(affine[:, :3])
        color_bias = 0.05 * torch.tanh(affine[:, 3:])
        calibrated_rgb = appearance.rgb * color_scale[:, None] + color_bias[:, None]
        camera_residual = self.camera_rgb_residual(condition)
        camera_gate = self.camera_rgb_gate(condition)
        confidence = appearance.confidence
        field_rgb = calibrated_rgb + confidence * (
            residual * self.max_observed_rgb_residual
            + camera_residual * camera_gate * self.max_camera_rgb_residual
        )
        field_rgb = field_rgb.clamp(0.0, 1.0)
        field_rgb = torch.where(
            appearance.validity[..., None], field_rgb, torch.zeros_like(field_rgb),
        )
        return appearance, field_rgb

    def forward(
        self,
        lidar_features: torch.Tensor,
        lidar_indices: torch.Tensor,
        spatial_shape: tuple[int, int, int] | list[int],
        query_indices: torch.Tensor,
        query_xyz: torch.Tensor,
        source_images: torch.Tensor,
        source_intrinsics: torch.Tensor,
        source_extrinsics: torch.Tensor,
        source_distortions: torch.Tensor,
        source_rgb_images: torch.Tensor | None = None,
        source_rgb_intrinsics: torch.Tensor | None = None,
        source_rgb_distortions: torch.Tensor | None = None,
        source_lidar_depth: torch.Tensor | None = None,
        source_validity: torch.Tensor | None = None,
        source_time_offsets: torch.Tensor | None = None,
        source_alignment_confidence: torch.Tensor | None = None,
        focal_role: torch.Tensor | None = None,
        source_camera_indices: torch.Tensor | None = None,
        exact_occupancy: torch.Tensor | None = None,
        exact_mask: torch.Tensor | None = None,
        target_view_direction: torch.Tensor | None = None,
        target_camera_index: torch.Tensor | int | None = None,
        target_focal_role: torch.Tensor | None = None,
    ) -> UnifiedFieldOutput:
        if source_images.ndim != 5:
            raise ValueError("source_images must be [B,V,3,H,W]")
        batch_size = source_images.shape[0]
        geometry = self.geometry(
            lidar_features,
            lidar_indices,
            spatial_shape,
            batch_size=batch_size,
            query_indices=query_indices,
        )
        geometry = self.lock_measured_geometry(geometry, exact_occupancy, exact_mask)
        if batch_size != 1:
            raise NotImplementedError("variable-count sparse voxel batches require packed attention; train with batch_size=1")
        geometry_features = geometry["surface_features"][None]
        if query_xyz.ndim == 2:
            query_xyz = query_xyz[None]
        source_features = self.encode_sources(source_images)
        appearance, field_rgb = self.fuse_at_points(
            geometry_features,
            query_xyz,
            source_features,
            source_images,
            source_intrinsics,
            source_extrinsics,
            source_distortions,
            source_rgb_images=source_rgb_images,
            source_rgb_intrinsics=source_rgb_intrinsics,
            source_rgb_distortions=source_rgb_distortions,
            source_lidar_depth=source_lidar_depth,
            source_validity=source_validity,
            source_time_offsets=source_time_offsets,
            source_alignment_confidence=source_alignment_confidence,
            focal_role=focal_role,
            source_camera_indices=source_camera_indices,
            target_view_direction=target_view_direction,
            target_camera_index=target_camera_index,
            target_focal_role=target_focal_role,
        )
        structure_validity = (
            geometry["query_validity"][:, 0] > 0.5
        ) & (geometry["occupancy"][:, 0] > 0.5)
        appearance_validity = appearance.validity[0]
        field_confidence = appearance.confidence * geometry["confidence"][None]
        return UnifiedFieldOutput(
            geometry=geometry,
            appearance=appearance,
            field_rgb=field_rgb,
            field_confidence=field_confidence,
            structure_validity=structure_validity[None],
            appearance_validity=appearance_validity[None],
            unknown_structure=(~structure_validity)[None],
            true_disocclusion=(structure_validity & ~appearance_validity)[None],
        )
