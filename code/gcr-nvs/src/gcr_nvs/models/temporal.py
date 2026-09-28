"""Temporal ConvGRU used by the three-frame refinement stage."""

from __future__ import annotations

import torch
from torch import nn


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        combined = torch.cat([x, hidden], dim=1)
        reset, update = self.gates(combined).chunk(2, dim=1)
        reset, update = torch.sigmoid(reset), torch.sigmoid(update)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * hidden], dim=1)))
        return (1 - update) * hidden + update * candidate


class TemporalConvGRU(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 48, bidirectional: bool = True):
        super().__init__()
        self.forward_cell = ConvGRUCell(input_channels, hidden_channels)
        self.bidirectional = bidirectional
        self.backward_cell = ConvGRUCell(input_channels, hidden_channels) if bidirectional else None
        self.output = nn.Conv2d(hidden_channels * (2 if bidirectional else 1), hidden_channels, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        outputs = []
        hidden = None
        for frame in sequence.unbind(dim=1):
            hidden = self.forward_cell(frame, hidden)
            outputs.append(hidden)
        if self.bidirectional:
            reverse = []
            hidden = None
            for frame in reversed(sequence.unbind(dim=1)):
                hidden = self.backward_cell(frame, hidden)
                reverse.append(hidden)
            outputs = [torch.cat([forward, backward], dim=1) for forward, backward in zip(outputs, reversed(reverse))]
        return self.output(outputs[-1])
