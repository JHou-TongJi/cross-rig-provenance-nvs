"""Documented DINOv2-B + NAF + Restormer GCR-NVS Foundation model."""

from __future__ import annotations

import torch
from torch import nn

from gcr_nvs.models.anchor_pipeline import LiDARAnchoredViewFusion
from gcr_nvs.models.restormer import RestormerSmallFeatureDecoder
from gcr_nvs.rendering.plane_sweep import PlaneSweepRenderer


class GCRNVSFoundationDINO(nn.Module):
    """Single-frame Foundation model following the documented B2 contract."""

    def __init__(
        self,
        use_dino: bool = True,
        plane_sweep_depth_layers: int = 32,
        front_narrow_depth_layers: int | None = None,
        dino_input_size: tuple[int, int] = (476, 840),
        freeze_plane_sweep: bool = True,
        freeze_view_fusion: bool = True,
        restormer_blocks: tuple[int, int, int, int] = (4, 6, 6, 8),
        restormer_refinement_blocks: int = 4,
        use_checkpoint: bool = True,
        observed_gate_scale: float = 1.0,
        correctness_aware_composition: bool = False,
        max_observed_residual: float = 0.15,
        lidar_depth_mode: str = "primary",
    ):
        super().__init__()
        self.freeze_plane_sweep = freeze_plane_sweep
        self.observed_gate_scale = observed_gate_scale
        self.correctness_aware_composition = correctness_aware_composition
        self.max_observed_residual = max_observed_residual
        self.plane_sweep = PlaneSweepRenderer(
            feature_channels=32,
            num_depth_layers=plane_sweep_depth_layers,
            depth_min=2.0,
            depth_max=80.0,
            narrow_depth_layers=front_narrow_depth_layers,
            lidar_depth_mode=lidar_depth_mode,
        )
        self.view_fusion = LiDARAnchoredViewFusion(use_dino=use_dino)
        self.view_fusion.encoder.dino_input_size = dino_input_size
        self.target_feature_fusion = nn.Conv2d(64, 32, 1, bias=True)
        with torch.no_grad():
            self.target_feature_fusion.weight.zero_()
            self.target_feature_fusion.bias.zero_()
            identity = torch.eye(32).view(32, 32, 1, 1)
            self.target_feature_fusion.weight[:, :32].copy_(identity)
        self.restormer = RestormerSmallFeatureDecoder(
            input_channels=48,
            output_channels=48,
            blocks=restormer_blocks,
            refinement_blocks=restormer_refinement_blocks,
            use_checkpoint=use_checkpoint,
        )
        self.rgb_head = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        self.narrow_rgb_head = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        self.completion_head = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        self.alpha_head = nn.Sequential(
            nn.Conv2d(48, 16, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        self.narrow_alpha_head = nn.Sequential(
            nn.Conv2d(48, 16, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        self.logvar_head = nn.Sequential(
            nn.Conv2d(48, 16, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        if freeze_plane_sweep:
            self.plane_sweep.requires_grad_(False)
            self.plane_sweep.log_baseline_temperature.requires_grad_(True)
            self.plane_sweep.log_baseline_temperature_narrow.requires_grad_(True)
            self.plane_sweep.eval()
        if freeze_view_fusion:
            self.view_fusion.observations.requires_grad_(False)
            self.view_fusion.fusion.requires_grad_(False)
            self.view_fusion.renderer.requires_grad_(False)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Old checkpoints used rgb_head for both observed repair and hole filling.
        # Copy it into the new completion head so strict loading preserves the
        # legacy initialization while allowing the branches to diverge in training.
        for suffix in ("0.weight", "0.bias", "2.weight", "2.bias"):
            completion_key = prefix + "completion_head." + suffix
            rgb_key = prefix + "rgb_head." + suffix
            if completion_key not in state_dict and rgb_key in state_dict:
                state_dict[completion_key] = state_dict[rgb_key].detach().clone()
            narrow_rgb_key = prefix + "narrow_rgb_head." + suffix
            if narrow_rgb_key not in state_dict and rgb_key in state_dict:
                state_dict[narrow_rgb_key] = state_dict[rgb_key].detach().clone()
            narrow_alpha_key = prefix + "narrow_alpha_head." + suffix
            alpha_key = prefix + "alpha_head." + suffix
            if narrow_alpha_key not in state_dict and alpha_key in state_dict:
                state_dict[narrow_alpha_key] = state_dict[alpha_key].detach().clone()
        for suffix in ("weight", "bias"):
            fusion_key = prefix + "target_feature_fusion." + suffix
            if fusion_key not in state_dict:
                state_dict[fusion_key] = getattr(
                    self.target_feature_fusion, suffix,
                ).detach().clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_plane_sweep:
            self.plane_sweep.eval()
        return self

    @staticmethod
    def geometry_contract(geometry: torch.Tensor) -> dict:
        if geometry.ndim != 4 or geometry.shape[1] != 13:
            raise ValueError(f"geometry must have shape [B, 13, H, W], got {tuple(geometry.shape)}")
        return {
            "shape": list(geometry.shape),
            "dtype": str(geometry.dtype),
            "channels": {
                "0:3": "coarse_rgb", "3": "depth", "4:7": "normal",
                "7": "opacity", "8": "confidence", "9": "view_variance",
                "10": "source_count", "11": "dynamic_mask", "12": "hole",
            },
            "source_provenance": "separate integer tensor; not part of geometry",
        }

    @staticmethod
    def _geometry(
        coarse: dict[str, torch.Tensor],
        sparse_geometry: torch.Tensor,
    ) -> torch.Tensor:
        rgb = coarse["rgb"]
        batch, _, height, width = rgb.shape
        zeros = rgb.new_zeros(batch, 1, height, width)
        opacity = (
            (coarse["valid_mask"] > 0.5)
            & (coarse.get("appearance_validity", coarse["valid_mask"]) > 0.5)
        ).to(rgb.dtype)
        confidence = coarse["confidence"].clamp(0.0, 1.0)
        if sparse_geometry.shape[1] >= 12:
            normals = sparse_geometry[:, 4:7]
            view_variance = sparse_geometry[:, 9:10]
            source_count = sparse_geometry[:, 10:11]
            dynamic = sparse_geometry[:, 11:12]
        else:
            normals = rgb.new_zeros(batch, 3, height, width)
            view_variance = source_count = dynamic = zeros
        return torch.cat([
            rgb,
            coarse["depth"],
            normals,
            opacity,
            confidence,
            view_variance,
            source_count,
            dynamic,
            1.0 - opacity,
        ], dim=1)

    def forward(
        self,
        source_images: torch.Tensor,
        source_calibs: list[dict[str, torch.Tensor]],
        target_calib: dict[str, torch.Tensor],
        ray_map: torch.Tensor,
        sparse_geometry: torch.Tensor,
        anchor_xyz: torch.Tensor,
        source_uv: torch.Tensor,
        target_uv: torch.Tensor,
        target_anchor_valid: torch.Tensor,
        source_valid: torch.Tensor,
        source_lidar_depths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.amp.autocast("cuda", enabled=False):
            coarse = self.plane_sweep(
                source_images.float(),
                source_calibs,
                target_calib,
                sparse_geometry[:, 3:4].float(),
                sparse_geometry[:, 11:12].float(),
                source_lidar_depths.float() if source_lidar_depths is not None else None,
            )
        geometry = self._geometry(coarse, sparse_geometry)
        fusion = self.view_fusion(
            source_images,
            anchor_xyz,
            source_uv,
            target_uv,
            target_anchor_valid,
            source_valid,
            geometry.shape[-2:],
        )
        target_features = self.target_feature_fusion(torch.cat([
            fusion["rendered_feature"], coarse["features"],
        ], dim=1))
        restored = self.restormer(torch.cat([geometry, ray_map, target_features], dim=1))
        target_focal_ratio = float(coarse["target_focal_ratio"])
        narrow_refiner_active = target_focal_ratio >= self.plane_sweep.narrow_focal_ratio_threshold
        residual_head = self.narrow_rgb_head if narrow_refiner_active else self.rgb_head
        alpha_head = self.narrow_alpha_head if narrow_refiner_active else self.alpha_head
        residual = torch.tanh(residual_head(restored))
        completion_logits = self.completion_head(restored)
        completion_rgb = torch.sigmoid(completion_logits)
        alpha = torch.sigmoid(alpha_head(restored))
        log_uncertainty = self.logvar_head(restored).clamp(-6.0, 2.0)
        opacity = geometry[:, 7:8].clamp(0.0, 1.0)
        confidence = geometry[:, 8:9].clamp(0.0, 1.0)
        hole = 1.0 - opacity
        if self.correctness_aware_composition:
            depth_confidence = coarse["depth_confidence"].clamp(0.0, 1.0)
            photometric_confidence = coarse["photometric_confidence"].clamp(0.0, 1.0)
            seam = coarse["seam_mask"].clamp(0.0, 1.0)
            geometry_reliability = (
                opacity
                * torch.sqrt((depth_confidence * photometric_confidence).clamp(min=1e-6))
                * (1.0 - seam)
            ).clamp(0.0, 1.0)
            protected_mask = (
                (opacity > 0.5)
                & (
                    (coarse["lidar_exact_mask"] > 0.5)
                    | (
                        (coarse["lidar_propagated_mask"] > 0)
                        & (coarse["lidar_propagated_mask"] <= 1)
                        & (coarse["lidar_reliability"] >= 0.80)
                    )
                )
                & (depth_confidence >= 0.50)
                & (photometric_confidence >= 0.55)
                & (seam <= 0.20)
            ).to(opacity.dtype)
            uncertain_observed = opacity * (1.0 - protected_mask)
            observed_alpha = alpha * uncertain_observed * (1.0 - geometry_reliability)
            bounded_residual = self.max_observed_residual * residual
            proposal = (geometry[:, :3] + bounded_residual).clamp(0.0, 1.0)
            observed_rgb = (
                (1.0 - observed_alpha) * geometry[:, :3]
                + observed_alpha * proposal
            ).clamp(0.0, 1.0)
            rgb = (opacity * observed_rgb + hole * completion_rgb).clamp(0.0, 1.0)
            alpha_effective = observed_alpha + hole
        else:
            geometry_reliability = confidence
            protected_mask = torch.zeros_like(opacity)
            uncertain_observed = opacity
            modify_gate = (
                hole + self.observed_gate_scale * opacity * (1.0 - confidence)
            ).clamp(0.0, 1.0)
            alpha_effective = alpha * modify_gate
            proposal = (geometry[:, :3] + residual).clamp(0.0, 1.0)
            observed_rgb = (
                (1.0 - alpha_effective) * geometry[:, :3]
                + alpha_effective * proposal
            ).clamp(0.0, 1.0)
            completion_rgb = proposal
            rgb = observed_rgb
        return {
            **fusion,
            "rgb": rgb,
            "final_rgb": rgb,
            "coarse_rgb": geometry[:, :3],
            "projected_rgb": coarse["rgb"],
            "coarse_depth": geometry[:, 3:4],
            "coarse_confidence": confidence,
            "depth_confidence": coarse["depth_confidence"],
            "stereo_depth": coarse["stereo_depth"],
            "stereo_confidence": coarse["stereo_confidence"],
            "lidar_reliability": coarse["lidar_reliability"],
            "lidar_exact_mask": coarse["lidar_exact_mask"],
            "lidar_propagated_mask": coarse["lidar_propagated_mask"],
            "structure_validity": coarse["structure_validity"],
            "photometric_confidence": coarse["photometric_confidence"],
            "photometric_agreement": coarse["photometric_agreement"],
            "view_dominance": coarse["view_dominance"],
            "valid_source_count": coarse["valid_source_count"],
            "appearance_validity": coarse["appearance_validity"],
            "per_view_valid_mask": coarse["per_view_valid_mask"],
            "view_weights": coarse["view_weights"],
            "view_similarity": coarse["view_similarity"],
            "view_baseline": coarse["view_baseline"],
            "baseline_temperature": coarse["baseline_temperature"],
            "depth_layer_count": coarse["depth_layer_count"],
            "target_focal_ratio": coarse["target_focal_ratio"],
            "narrow_refiner_active": narrow_refiner_active,
            "seam_mask": coarse["seam_mask"],
            "target_geometry": geometry,
            "restored_feature": restored,
            "target_features": target_features,
            "rgb_residual": residual,
            "proposal": proposal,
            "observed_rgb": observed_rgb,
            "completion_rgb": completion_rgb,
            "geometry_reliability": geometry_reliability,
            "protected_mask": protected_mask,
            "uncertain_observed_mask": uncertain_observed,
            "disocclusion_mask": hole,
            "correctness_aware_composition": self.correctness_aware_composition,
            "alpha": alpha,
            "alpha_effective": alpha_effective,
            "log_uncertainty": log_uncertainty,
            "depth": geometry[:, 3:4],
        }
