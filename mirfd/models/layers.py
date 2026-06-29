from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for BCHW tensors."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.weight.numel(),), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


def _group_count(dim: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, dim), 0, -1):
        if dim % groups == 0:
            return groups
    return 1


def make_norm(norm: Optional[str], dim: int) -> nn.Module:
    if norm is None or norm == "none" or norm == "identity":
        return nn.Identity()
    norm = norm.lower()
    if norm == "batch":
        return nn.BatchNorm2d(dim)
    if norm == "layer":
        return LayerNorm2d(dim)
    if norm == "group":
        return nn.GroupNorm(_group_count(dim), dim)
    raise ValueError(f"Unsupported norm: {norm}")


def make_act(act: Optional[str]) -> nn.Module:
    if act is None or act == "none" or act == "identity":
        return nn.Identity()
    act = act.lower()
    if act == "gelu":
        return nn.GELU()
    if act == "silu" or act == "swish":
        return nn.SiLU(inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation: {act}")


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        norm: str = "batch",
        act: str = "gelu",
        groups: int = 1,
        bias: Optional[bool] = None,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        if bias is None:
            bias = norm in (None, "none", "identity")
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias),
            make_norm(norm, out_ch),
            make_act(act),
        )
