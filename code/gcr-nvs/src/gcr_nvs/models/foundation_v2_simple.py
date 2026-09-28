"""Simplified Foundation V2 - focuses on removing 0.25 limit first.

Step 1: Just remove the 0.25 residual limit
Step 2: Later integrate plane-sweep
"""

from __future__ import annotations

import torch
from torch import nn

from gcr_nvs.models.refiner import LiteTemporalRefiner


class FoundationV2Simple(nn.Module):
    """Simplified version: just remove 0.25 limit, use sparse geometry."""
    
    def __init__(
        self,
        base_channels: int = 48,
        remove_residual_limit: bool = True,
    ):
        super().__init__()
        
        # Use the standard refiner but with no residual limit
        self.refiner = LiteTemporalRefiner(
            input_channels=48,
            base_channels=base_channels,
            max_rgb_residual=1.0 if remove_residual_limit else 0.25,
        )
    
    def forward(self, geometry, ray_map, rendered_feature=None):
        """
        Args:
            geometry: [B, 12, H, W] or [B, 13, H, W]
            ray_map: [B, 3, H, W]
            rendered_feature: [B, 32, H, W] optional
        
        Returns:
            dict with rgb, etc.
        """
        B, C, H, W = geometry.shape
        
        # Ensure geometry is 13 channels
        if C == 12:
            # Add hole mask as 13th channel
            hole_mask = (geometry[:, 7:8] <= 0).float()
            geometry = torch.cat([geometry, hole_mask], dim=1)
        
        # Prepare 48-channel input
        if rendered_feature is None:
            rendered_feature = geometry.new_zeros(B, 32, H, W)
        
        full_input = torch.cat([
            geometry,      # 13 channels
            ray_map,       # 3 channels  
            rendered_feature,  # 32 channels
        ], dim=1)  # Total: 48 channels
        
        # Refine
        refined = self.refiner(full_input, ray_map, rendered_feature)
        
        # Add coarse_rgb for compatibility
        refined['coarse_rgb'] = geometry[:, :3]
        refined['depth'] = geometry[:, 3:4]
        
        return refined
