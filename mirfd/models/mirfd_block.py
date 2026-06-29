from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .layers import ConvNormAct, make_norm
from .ss2d import build_mamba_block


class HighFrequencyEnhancer(nn.Module):
    def __init__(self, dim: int, kernels: Iterable[int] = (3, 5), norm: str = "batch") -> None:
        super().__init__()
        kernels = tuple(kernels)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim)
                for k in kernels
            ]
        )
        self.fuse = ConvNormAct(dim * len(kernels), dim, kernel_size=1, padding=0, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class TargetAwareGate(nn.Module):
    def __init__(self, dim: int, norm: str = "batch") -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            make_norm(norm, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, low: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.gate(torch.cat([low, residual], dim=1))


class MIRFDBlock(nn.Module):
    """Mamba-induced residual frequency decoupling block."""

    SUPPORTED_RESIDUALS = {"mamba_residual", "avgpool", "laplace", "sobel"}
    SUPPORTED_FUSIONS = {"concat", "residual_compensation"}

    def __init__(
        self,
        dim: int,
        mamba_block: type[nn.Module] | None = None,
        mamba_kwargs: dict | None = None,
        residual_type: str = "mamba_residual",
        fusion: str = "concat",
        use_gate: bool = True,
        use_learnable_gamma: bool = False,
        hfe_kernels: Iterable[int] = (3, 5),
        norm: str = "batch",
        pre_norm: str = "layer",
    ) -> None:
        super().__init__()
        if residual_type not in self.SUPPORTED_RESIDUALS:
            raise ValueError(f"Unsupported residual_type: {residual_type}")
        if fusion not in self.SUPPORTED_FUSIONS:
            raise ValueError(f"Unsupported fusion: {fusion}")

        mamba_block = mamba_block or build_mamba_block
        mamba_kwargs = mamba_kwargs or {}

        self.residual_type = residual_type
        self.fusion = fusion
        self.use_gate = use_gate
        self.norm = make_norm(pre_norm, dim)
        self.mamba = mamba_block(dim, **mamba_kwargs)
        self.align = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            make_norm(norm, dim),
        )
        self.hfe = HighFrequencyEnhancer(dim, kernels=hfe_kernels, norm=norm)
        self.gate = TargetAwareGate(dim, norm=norm) if use_gate else None

        if fusion == "concat":
            self.fuse = ConvNormAct(dim * 2, dim, kernel_size=1, padding=0, norm=norm)
        else:
            self.gamma = nn.Parameter(torch.tensor(0.1), requires_grad=use_learnable_gamma)

        self.register_buffer(
            "laplace_kernel",
            torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "sobel_x_kernel",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0,
            persistent=False,
        )
        self.register_buffer(
            "sobel_y_kernel",
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0,
            persistent=False,
        )

    @staticmethod
    def _depthwise_fixed(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        weight = kernel.to(device=x.device, dtype=x.dtype).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, weight, padding=1, groups=x.shape[1])

    def _low_and_residual(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.residual_type == "mamba_residual":
            fm = self.mamba(self.norm(x))
            low = self.align(fm)
            residual = x - low
            return low, residual

        if self.residual_type == "avgpool":
            low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
            residual = x - low
            return low, residual

        if self.residual_type == "laplace":
            residual = self._depthwise_fixed(x, self.laplace_kernel)
            low = x - residual
            return low, residual

        grad_x = self._depthwise_fixed(x, self.sobel_x_kernel)
        grad_y = self._depthwise_fixed(x, self.sobel_y_kernel)
        residual = 0.5 * (grad_x.abs() + grad_y.abs())
        low = x - residual
        return low, residual

    def forward(self, x: torch.Tensor, return_branches: bool = False):
        low, residual = self._low_and_residual(x)
        high = self.hfe(residual)
        gate = self.gate(low, residual) if self.gate is not None else torch.ones_like(high)
        high_hat = gate * high

        if self.fusion == "concat":
            out = self.fuse(torch.cat([low, high_hat], dim=1)) + x
        else:
            out = x + low + self.gamma * high_hat

        if return_branches:
            return out, {
                "low": low,
                "high": high_hat,
                "residual": residual,
                "gate": gate,
            }
        return out
