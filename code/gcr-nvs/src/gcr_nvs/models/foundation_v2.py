"""GCR-NVS Foundation V2 with Plane-Sweep dense RGB baseline.

Key improvements:
1. Plane-sweep dense RGB as coarse (not sparse black holes)
2. No 0.25 residual limit
3. Direct RGB prediction capability
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

from gcr_nvs.rendering.plane_sweep import PlaneSweepRenderer
from gcr_nvs.models.refiner import LiteTemporalRefiner
from gcr_nvs.models.restormer import RestormerFeatureAdapter


class FoundationV2SingleFrame(nn.Module):
    """Foundation V2 with plane-sweep dense baseline."""
    
    def __init__(
        self,
        use_plane_sweep: bool = True,
        plane_sweep_depth_layers: int = 48,
        refiner_base_channels: int = 48,
        remove_residual_limit: bool = True,
        freeze_plane_sweep: bool = False,
    ):
        super().__init__()
        
        self.use_plane_sweep = use_plane_sweep
        self.remove_residual_limit = remove_residual_limit
        self.freeze_plane_sweep = freeze_plane_sweep
        
        if use_plane_sweep:
            self.plane_sweep = PlaneSweepRenderer(
                feature_channels=32,
                num_depth_layers=plane_sweep_depth_layers,
                depth_min=2.0,
                depth_max=80.0,
            )
            if freeze_plane_sweep:
                self.plane_sweep.requires_grad_(False)
                self.plane_sweep.eval()
        
        self.refiner = LiteTemporalRefiner(
            input_channels=48,  # 13 geom + 3 ray + 32 features
            base_channels=refiner_base_channels,
            max_rgb_residual=1.0 if remove_residual_limit else 0.25,
        )
        
        # Optional restormer for quality
        self.use_restormer = False  # Can enable later
        if self.use_restormer:
            self.restormer = RestormerFeatureAdapter(
                in_channels=13,
                out_channels=32,
                channels=48,
                blocks=4,  # Start with 4, can increase to 16
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_plane_sweep and self.freeze_plane_sweep:
            self.plane_sweep.eval()
        return self

    def forward(
        self,
        source_images: torch.Tensor,
        source_calibs: list[dict],
        target_calib: dict,
        target_ray_map: torch.Tensor,
        lidar_depth_guide: torch.Tensor | None = None,
        sparse_geometry: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            source_images: [B, num_views, 3, H, W]
            source_calibs: list of calibration dicts
            target_calib: target camera calibration
            target_ray_map: [B, 3, H, W] ray directions
            lidar_depth_guide: [B, 1, H, W] optional LiDAR depth
            sparse_geometry: [B, 12, H, W] optional sparse geometry from fixed splat
        
        Returns:
            dict with rgb, depth, confidence, etc.
        """
        B, num_views, _, H, W = source_images.shape
        
        if self.use_plane_sweep:
            context = torch.no_grad() if self.freeze_plane_sweep else nullcontext()
            with context:
                with torch.amp.autocast('cuda', enabled=False):
                    ps_output = self.plane_sweep(
                        source_images.float(), source_calibs, target_calib,
                        None if lidar_depth_guide is None else lidar_depth_guide.float(),
                    )
            
            projected_rgb = ps_output['rgb']
            coarse_depth = ps_output['depth']  # [B, 1, H, W]
            coarse_confidence = ps_output['confidence']  # [B, 1, H, W]
            coarse_opacity = ps_output['valid_mask']
            coarse_seam = ps_output.get('seam_mask', projected_rgb.new_zeros(B, 1, H, W))
        else:
            # Fallback to sparse geometry
            if sparse_geometry is not None:
                projected_rgb = sparse_geometry[:, :3]
                coarse_depth = sparse_geometry[:, 3:4]
                coarse_confidence = sparse_geometry[:, 8:9]
                coarse_opacity = sparse_geometry[:, 7:8]
                coarse_seam = sparse_geometry[:, 11:12]
            else:
                raise ValueError('Need either plane_sweep or sparse_geometry')
        
        zeros = projected_rgb.new_zeros(B, 1, H, W)
        normals = projected_rgb.new_zeros(B, 3, H, W)
        opacity = (coarse_opacity > 0.5).to(projected_rgb.dtype)
        coarse_rgb = projected_rgb * opacity
        feathered_opacity = F.avg_pool2d(
            opacity,
            61,
            stride=1,
            padding=30,
            count_include_pad=False,
        )
        hole_boundary = 4.0 * feathered_opacity * (1.0 - feathered_opacity)
        coarse_seam = torch.maximum(coarse_seam, hole_boundary)
        hole = 1.0 - opacity
        geom_input = torch.cat([
            coarse_rgb, coarse_depth, normals, opacity, coarse_confidence,
            coarse_seam, feathered_opacity, coarse_seam, hole,
        ], dim=1)
        
        # Optional restormer
        if self.use_restormer:
            refiner_input_features = self.restormer(geom_input)
        elif self.use_plane_sweep:
            refiner_input_features = ps_output['features']
        else:
            refiner_input_features = geom_input.new_zeros(B, 32, H, W)
        
        refined = self.refiner(geom_input, target_ray_map, refiner_input_features)
        
        # Prepare output
        output = {
            'coarse_rgb': coarse_rgb,
            'projected_rgb': projected_rgb,
            'coarse_depth': coarse_depth,
            'coarse_confidence': coarse_confidence,
            'seam_mask': coarse_seam,
            **refined,
        }
        
        return output


class FoundationV2Temporal(nn.Module):
    """Foundation V2 with 3-frame temporal support."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.single_frame = FoundationV2SingleFrame(**kwargs)
        
        # Add ConvGRU for temporal fusion
        from gcr_nvs.models.temporal import TemporalConvGRU
        self.temporal_fusion = TemporalConvGRU(
            input_channels=3,  # RGB
            hidden_channels=32,
            bidirectional=True,
        )
        self.temporal_rgb = nn.Conv2d(32, 3, 3, padding=1)
    
    def forward(
        self,
        frames: list[dict],  # List of 3 frames (t-1, t, t+1)
        target_ray_map: torch.Tensor,
    ) -> dict:
        """Process 3 frames with temporal fusion."""
        
        # Process each frame independently
        frame_outputs = []
        for frame_data in frames:
            output = self.single_frame(
                source_images=frame_data['source_images'],
                source_calibs=frame_data['source_calibs'],
                target_calib=frame_data['target_calib'],
                target_ray_map=target_ray_map,
                lidar_depth_guide=frame_data.get('lidar_depth_guide'),
                sparse_geometry=frame_data.get('sparse_geometry'),
            )
            frame_outputs.append(output)
        
        # Stack RGB from 3 frames
        rgb_sequence = torch.stack([out['rgb'] for out in frame_outputs], dim=1)  # [B, 3, 3, H, W]
        
        # Temporal fusion
        rgb_fused = torch.sigmoid(self.temporal_rgb(self.temporal_fusion(rgb_sequence)))
        
        # Return center frame output with fused RGB
        center_output = frame_outputs[1]
        center_output['rgb'] = rgb_fused
        center_output['final_rgb'] = rgb_fused
        
        return center_output
