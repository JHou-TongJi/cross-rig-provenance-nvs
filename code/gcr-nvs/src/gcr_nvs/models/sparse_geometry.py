"""SparseConv U-Net geometry student with LiDAR-only inputs."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

try:
    import spconv.pytorch as spconv
except ImportError:  # pragma: no cover - exercised only in minimal CPU installs.
    spconv = None


def _replace(sparse_tensor, features: torch.Tensor):
    return sparse_tensor.replace_feature(features)


class SparseResidualBlock(nn.Module):
    def __init__(self, channels: int, indice_key: str):
        super().__init__()
        if spconv is None:
            raise RuntimeError("spconv is required for SparseGeometryStudent")
        self.conv1 = spconv.SubMConv3d(
            channels, channels, 3, padding=1, bias=False,
            indice_key=f"{indice_key}_a",
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = spconv.SubMConv3d(
            channels, channels, 3, padding=1, bias=False,
            indice_key=f"{indice_key}_b",
        )
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, inputs):
        residual = inputs.features
        outputs = self.conv1(inputs)
        outputs = _replace(outputs, F.silu(self.norm1(outputs.features)))
        outputs = self.conv2(outputs)
        outputs = _replace(
            outputs,
            F.silu(self.norm2(outputs.features) + residual),
        )
        return outputs


class SparseGeometryStudent(nn.Module):
    """Predict occupancy, SDF, normal, visibility, and geometry confidence."""

    def __init__(
        self,
        input_dim: int = 6,
        channels=(32, 64, 128, 256),
        query_position_encoding: bool = False,
    ):
        super().__init__()
        if spconv is None:
            raise RuntimeError(
                "spconv is not installed; install the wheel matching the PyTorch CUDA ABI"
            )
        c1, c2, c3, c4 = channels
        self.query_position_encoding = query_position_encoding
        self.stem = spconv.SubMConv3d(
            input_dim, c1, 3, padding=1, bias=False, indice_key="stem",
        )
        self.stem_norm = nn.BatchNorm1d(c1)
        self.enc1 = SparseResidualBlock(c1, "enc1")
        self.down2 = spconv.SparseConv3d(
            c1, c2, 3, stride=2, padding=1, bias=False, indice_key="down2",
        )
        self.down2_norm = nn.BatchNorm1d(c2)
        self.enc2 = SparseResidualBlock(c2, "enc2")
        self.down3 = spconv.SparseConv3d(
            c2, c3, 3, stride=2, padding=1, bias=False, indice_key="down3",
        )
        self.down3_norm = nn.BatchNorm1d(c3)
        self.enc3 = SparseResidualBlock(c3, "enc3")
        self.down4 = spconv.SparseConv3d(
            c3, c4, 3, stride=2, padding=1, bias=False, indice_key="down4",
        )
        self.down4_norm = nn.BatchNorm1d(c4)
        self.latent = SparseResidualBlock(c4, "latent")
        self.up3 = spconv.SparseInverseConv3d(
            c4, c3, 3, bias=False, indice_key="down4",
        )
        self.dec3 = SparseResidualBlock(c3, "dec3")
        self.up2 = spconv.SparseInverseConv3d(
            c3, c2, 3, bias=False, indice_key="down3",
        )
        self.dec2 = SparseResidualBlock(c2, "dec2")
        self.up1 = spconv.SparseInverseConv3d(
            c2, c1, 3, bias=False, indice_key="down2",
        )
        self.dec1 = SparseResidualBlock(c1, "dec1")
        query_geometry_dim = 19 if query_position_encoding else 7
        self.query_fusion = nn.Sequential(
            nn.Linear(c1 + c2 + c3 + c4 + query_geometry_dim, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, c1),
            nn.SiLU(inplace=True),
        )
        self.occupancy_head = nn.Linear(c1, 1)
        self.sdf_head = nn.Linear(c1, 1)
        self.normal_head = nn.Linear(c1, 3)
        self.visibility_head = nn.Linear(c1, 1)
        self.confidence_head = nn.Linear(c1, 1)

    @staticmethod
    def _skip(outputs, skip):
        if not torch.equal(outputs.indices, skip.indices):
            raise RuntimeError("spconv inverse-convolution indices do not match the skip")
        return _replace(outputs, outputs.features + skip.features)

    @staticmethod
    def _lookup_sparse(query_indices: torch.Tensor, sparse_tensor):
        source_indices = sparse_tensor.indices.to(dtype=torch.int64)
        query_indices = query_indices.to(dtype=torch.int64)
        extent = torch.maximum(
            source_indices.amax(dim=0), query_indices.amax(dim=0),
        ) + 2
        multipliers = torch.stack([
            extent[1] * extent[2] * extent[3],
            extent[2] * extent[3],
            extent[3],
            extent.new_tensor(1),
        ])
        source_keys = (source_indices * multipliers).sum(dim=1)
        query_keys = (query_indices * multipliers).sum(dim=1)
        sorted_keys, order = source_keys.sort()
        positions = torch.searchsorted(sorted_keys, query_keys)
        clamped = positions.clamp(max=max(len(sorted_keys) - 1, 0))
        valid = (positions < len(sorted_keys)) & (
            sorted_keys[clamped] == query_keys
        )
        features = sparse_tensor.features.new_zeros(
            len(query_indices), sparse_tensor.features.shape[1],
        )
        if valid.any():
            features[valid] = sparse_tensor.features[order[clamped[valid]]]
        return features, valid[:, None].to(features.dtype)

    def _query_features(
        self,
        query_indices: torch.Tensor,
        spatial_shape: tuple[int, int, int] | list[int],
        pyramids,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gathered = []
        validity = []
        local_coordinates = []
        for level, sparse_tensor in enumerate(pyramids):
            scaled = query_indices.clone().to(dtype=torch.int64)
            stride = 2**level
            if self.query_position_encoding:
                local_coordinates.append(
                    torch.remainder(scaled[:, 1:], stride).to(torch.float32)
                    / max(stride, 1)
                )
            scaled[:, 1:] = torch.div(scaled[:, 1:], stride, rounding_mode="floor")
            features, valid = self._lookup_sparse(scaled, sparse_tensor)
            gathered.append(features)
            validity.append(valid)
        spatial = query_indices.new_tensor(
            list(spatial_shape), dtype=torch.float32,
        ).clamp(min=1.0)
        normalized_xyz = query_indices[:, 1:].to(torch.float32) / spatial
        geometry_features = [*validity, normalized_xyz]
        if self.query_position_encoding:
            geometry_features.extend(local_coordinates)
        query_inputs = torch.cat([*gathered, *geometry_features], dim=1)
        return self.query_fusion(query_inputs), torch.cat(validity, dim=1)

    def encode_sparse(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: tuple[int, int, int] | list[int],
        batch_size: int,
    ) -> dict[str, object]:
        if features.ndim != 2 or features.shape[1] != self.stem.in_channels:
            raise ValueError(f"features must have shape [N,{self.stem.in_channels}]")
        sparse = spconv.SparseConvTensor(
            features,
            indices.to(dtype=torch.int32),
            list(spatial_shape),
            batch_size,
        )
        enc1 = self.stem(sparse)
        enc1 = _replace(enc1, F.silu(self.stem_norm(enc1.features)))
        enc1 = self.enc1(enc1)
        enc2 = self.down2(enc1)
        enc2 = _replace(enc2, F.silu(self.down2_norm(enc2.features)))
        enc2 = self.enc2(enc2)
        enc3 = self.down3(enc2)
        enc3 = _replace(enc3, F.silu(self.down3_norm(enc3.features)))
        enc3 = self.enc3(enc3)
        latent = self.down4(enc3)
        latent = _replace(latent, F.silu(self.down4_norm(latent.features)))
        latent = self.latent(latent)
        decoded = self._skip(self.up3(latent), enc3)
        decoded = self.dec3(decoded)
        decoded = self._skip(self.up2(decoded), enc2)
        decoded = self.dec2(decoded)
        decoded = self._skip(self.up1(decoded), enc1)
        decoded = self.dec1(decoded)
        return {
            "decoded": decoded,
            "pyramids": (enc1, enc2, enc3, latent),
            "spatial_shape": tuple(int(value) for value in spatial_shape),
        }

    def query_encoded(
        self,
        encoded: dict[str, object],
        query_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        decoded = encoded["decoded"]
        output_indices = decoded.indices
        output_features = decoded.features
        query_validity = output_features.new_ones(len(output_features), 1)
        if query_indices is not None:
            output_indices = query_indices.to(dtype=torch.int32)
            output_features, per_level_validity = self._query_features(
                output_indices,
                encoded["spatial_shape"],
                encoded["pyramids"],
            )
            query_validity = per_level_validity.amax(dim=1, keepdim=True)
        return {
            "indices": output_indices,
            "surface_features": output_features,
            "query_validity": query_validity,
            "occupancy": torch.sigmoid(self.occupancy_head(output_features)),
            "sdf": self.sdf_head(output_features),
            "normal": F.normalize(self.normal_head(output_features), dim=-1, eps=1e-6),
            "visibility": torch.sigmoid(self.visibility_head(output_features)),
            "confidence": torch.sigmoid(self.confidence_head(output_features)),
        }

    def forward(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: tuple[int, int, int] | list[int],
        batch_size: int,
        query_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode_sparse(
            features, indices, spatial_shape, batch_size,
        )
        return self.query_encoded(encoded, query_indices)


def hard_replace_exact(
    predicted: torch.Tensor,
    exact_values: torch.Tensor,
    exact_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep measured LiDAR values immutable at the student output boundary."""
    if exact_mask.dtype != torch.bool:
        exact_mask = exact_mask > 0
    return torch.where(exact_mask, exact_values, predicted)
