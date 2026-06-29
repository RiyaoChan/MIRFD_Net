from __future__ import annotations

import math

import torch
from torch import nn


def _logit(value: float) -> float:
    value = min(max(value, 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class DirectionalSelectiveScan2D(nn.Module):
    """Lightweight 2D selective state propagation.

    This is a self-contained Mamba-style fallback. It performs learned
    exponential scans in four spatial directions and intentionally behaves like
    an adaptive low-pass context propagator. Replace this module with a VMamba
    SS2D block when an external implementation is available.
    """

    def __init__(self, dim: int, decay_init: float = 0.75) -> None:
        super().__init__()
        self.decay_logit = nn.Parameter(torch.full((4, dim), _logit(decay_init)))

    @staticmethod
    def _scan_sequence(x: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
        # x: N, C, L. decay: C.
        d = decay.view(1, -1)
        scale = 1.0 - d
        state = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        outputs = []
        for idx in range(x.shape[-1]):
            state = d * state + scale * x[:, :, idx]
            outputs.append(state)
        return torch.stack(outputs, dim=-1)

    def _scan_width(self, x: torch.Tensor, decay: torch.Tensor, reverse: bool) -> torch.Tensor:
        if reverse:
            x = torch.flip(x, dims=(-1,))
        b, c, h, w = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * h, c, w)
        out = self._scan_sequence(seq, decay)
        out = out.reshape(b, h, c, w).permute(0, 2, 1, 3).contiguous()
        if reverse:
            out = torch.flip(out, dims=(-1,))
        return out

    def _scan_height(self, x: torch.Tensor, decay: torch.Tensor, reverse: bool) -> torch.Tensor:
        out = self._scan_width(x.transpose(-1, -2), decay, reverse)
        return out.transpose(-1, -2).contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        decays = torch.sigmoid(self.decay_logit).clamp(0.02, 0.98)
        left_to_right = self._scan_width(x, decays[0], reverse=False)
        right_to_left = self._scan_width(x, decays[1], reverse=True)
        top_to_bottom = self._scan_height(x, decays[2], reverse=False)
        bottom_to_top = self._scan_height(x, decays[3], reverse=True)
        return left_to_right, right_to_left, top_to_bottom, bottom_to_top


class Mamba2D(nn.Module):
    """A compact Mamba-style 2D block for MIRFD-Net.

    Input and output are BCHW tensors with the same channel count.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 2.0,
        conv_kernel: int = 3,
        dropout: float = 0.0,
        decay_init: float = 0.75,
    ) -> None:
        super().__init__()
        inner_dim = int(dim * expansion)
        padding = conv_kernel // 2

        self.in_proj = nn.Conv2d(dim, inner_dim * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(
            inner_dim,
            inner_dim,
            kernel_size=conv_kernel,
            padding=padding,
            groups=inner_dim,
        )
        self.act = nn.SiLU(inplace=False)
        self.scan = DirectionalSelectiveScan2D(inner_dim, decay_init=decay_init)
        self.mix = nn.Conv2d(inner_dim * 4, inner_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(inner_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, gate = self.in_proj(x).chunk(2, dim=1)
        u = self.act(self.dwconv(u))
        streams = self.scan(u)
        y = self.mix(torch.cat(streams, dim=1))
        y = y * self.act(gate)
        y = self.out_proj(y)
        return self.dropout(y)
