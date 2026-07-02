from __future__ import annotations

import torch
from torch import nn

from .layers import make_norm


class ContextGuidedResidualSelector(nn.Module):
    """Use Mamba-induced context to select target-related residual responses."""

    def __init__(
        self,
        dim: int,
        use_reference: bool = False,
        gamma_init: float = 0.1,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.use_reference = bool(use_reference)
        in_dim = dim * (3 if self.use_reference else 2)
        self.selector = nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size=1, bias=False),
            make_norm(norm, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            make_norm(norm, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(
        self,
        low: torch.Tensor,
        residual: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_reference:
            if reference is None:
                raise ValueError(
                    "ContextGuidedResidualSelector was built with use_reference=True, "
                    "but no reference tensor was provided."
                )
            x = torch.cat([low, residual, reference], dim=1)
        else:
            x = torch.cat([low, residual], dim=1)

        selector = self.selector(x)
        gamma = torch.clamp(self.gamma, 0.0, 1.0)
        selected = residual + gamma * (selector - 0.5) * residual
        return selected, selector
