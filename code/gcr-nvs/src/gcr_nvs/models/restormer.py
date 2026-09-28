"""Compact Restormer-style feature adapter for the Foundation refiner."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(variance + 1e-6) * self.weight + self.bias


class RestormerBlock(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.0, heads: int = 4, bias: bool = True):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        hidden = int(channels * expansion)
        self.norm1 = LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=bias)
        self.project = nn.Conv2d(channels, channels, 1, bias=bias)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden * 2, 1, bias=bias),
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias),
            nn.GLU(dim=1),
            nn.Conv2d(hidden, channels, 1, bias=bias),
        )
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        query, key, value = self.qkv_dw(self.qkv(normalized)).chunk(3, dim=1)
        batch, channels, height, width = query.shape
        head_channels = channels // self.heads
        query = query.reshape(batch, self.heads, head_channels, height * width)
        key = key.reshape(batch, self.heads, head_channels, height * width)
        value = value.reshape(batch, self.heads, head_channels, height * width)
        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)
        attention = torch.softmax((query @ key.transpose(-1, -2)) * self.temperature, dim=-1)
        restored = (attention @ value).reshape(batch, channels, height, width)
        x = x + self.project(restored)
        return x + self.ffn(self.norm2(x))


class RestormerFeatureAdapter(nn.Module):
    def __init__(self, input_channels: int = 48, output_channels: int = 32, channels: int = 48, blocks: int = 2):
        super().__init__()
        self.input = nn.Conv2d(input_channels, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[RestormerBlock(channels) for _ in range(blocks)])
        self.output = nn.Conv2d(channels, output_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(x)))


class RestormerStage(nn.Module):
    def __init__(self, channels: int, blocks: int, heads: int, expansion: float, use_checkpoint: bool):
        super().__init__()
        self.blocks = nn.ModuleList([
            RestormerBlock(channels, expansion=expansion, heads=heads, bias=False)
            for _ in range(blocks)
        ])
        self.use_checkpoint = use_checkpoint

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training and inputs.requires_grad:
                inputs = checkpoint(block, inputs, use_reentrant=False)
            else:
                inputs = block(inputs)
        return inputs


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(inputs)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(inputs)


class RestormerSmallFeatureDecoder(nn.Module):
    """Documented Restormer-Small hierarchy returning 48-channel features."""

    def __init__(
        self,
        input_channels: int = 48,
        output_channels: int = 48,
        dim: int = 48,
        blocks: tuple[int, int, int, int] = (4, 6, 6, 8),
        refinement_blocks: int = 4,
        heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        expansion: float = 2.66,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(input_channels, dim, 3, padding=1, bias=False)
        self.encoder1 = RestormerStage(dim, blocks[0], heads[0], expansion, use_checkpoint)
        self.down1 = Downsample(dim)
        self.encoder2 = RestormerStage(dim * 2, blocks[1], heads[1], expansion, use_checkpoint)
        self.down2 = Downsample(dim * 2)
        self.encoder3 = RestormerStage(dim * 4, blocks[2], heads[2], expansion, use_checkpoint)
        self.down3 = Downsample(dim * 4)
        self.latent = RestormerStage(dim * 8, blocks[3], heads[3], expansion, use_checkpoint)
        self.up3 = Upsample(dim * 8)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=False)
        self.decoder3 = RestormerStage(dim * 4, blocks[2], heads[2], expansion, use_checkpoint)
        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.decoder2 = RestormerStage(dim * 2, blocks[1], heads[1], expansion, use_checkpoint)
        self.up1 = Upsample(dim * 2)
        self.decoder1 = RestormerStage(dim * 2, blocks[0], heads[0], expansion, use_checkpoint)
        self.refinement = RestormerStage(dim * 2, refinement_blocks, heads[0], expansion, use_checkpoint)
        self.output = nn.Conv2d(dim * 2, output_channels, 3, padding=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2] % 8 or inputs.shape[-1] % 8:
            raise ValueError("Restormer input height and width must be divisible by 8")
        level1 = self.encoder1(self.patch_embed(inputs))
        level2 = self.encoder2(self.down1(level1))
        level3 = self.encoder3(self.down2(level2))
        latent = self.latent(self.down3(level3))
        decoded3 = self.reduce3(torch.cat([self.up3(latent), level3], dim=1))
        decoded3 = self.decoder3(decoded3)
        decoded2 = self.reduce2(torch.cat([self.up2(decoded3), level2], dim=1))
        decoded2 = self.decoder2(decoded2)
        decoded1 = torch.cat([self.up1(decoded2), level1], dim=1)
        decoded1 = self.refinement(self.decoder1(decoded1))
        return self.output(decoded1)
