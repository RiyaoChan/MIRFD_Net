from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .layers import ConvNormAct, make_norm
from .context_residual import ContextGuidedResidualSelector
from .frequency_enhancer import FFCFrequencyResidualEnhancer, FrequencySelectiveResidualEnhancer
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


class FixedDepthwiseBlur(nn.Module):
    """Fixed 3x3 binomial low-pass filter applied channel-wise."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ) / 16.0
        self.register_buffer("weight", kernel.view(1, 1, 3, 3).repeat(dim, 1, 1, 1), persistent=False)
        self.groups = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight.to(dtype=x.dtype), padding=1, groups=self.groups)


class LowSmooth(nn.Module):
    """Lightweight calibration on the Mamba-induced low representation."""

    def __init__(self, dim: int, beta_init: float = 0.3) -> None:
        super().__init__()
        self.blur = FixedDepthwiseBlur(dim)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = torch.clamp(self.beta, 0.0, 1.0)
        return x + beta * (self.blur(x) - x)


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


def build_high_enhancer(
    enhancer_type: str,
    dim: int,
    hfe_kernels: Iterable[int] = (3, 5),
    norm: str = "batch",
    fsre_num_bands: int = 4,
    fsre_window_size: int = 8,
    fsre_gamma_init: float = 0.1,
    ffc_gamma_init: float = 0.1,
    ffc_use_highfreq_gate: bool = True,
    ffc_highfreq_threshold: float = 0.5,
    ffc_gate_reduction: int = 4,
    ffc_local_kernel: int = 3,
    ffc_fft_norm: str = "ortho",
) -> nn.Module:
    if enhancer_type == "identity":
        return nn.Identity()
    if enhancer_type == "conv_hfe":
        return HighFrequencyEnhancer(dim, kernels=hfe_kernels, norm=norm)
    if enhancer_type == "freq_window":
        return FrequencySelectiveResidualEnhancer(
            dim=dim,
            num_bands=fsre_num_bands,
            window_size=fsre_window_size,
            gamma_init=fsre_gamma_init,
            norm=norm,
        )
    if enhancer_type == "ffc":
        return FFCFrequencyResidualEnhancer(
            dim=dim,
            gamma_init=ffc_gamma_init,
            norm=norm,
            fft_norm=ffc_fft_norm,
            use_highfreq_gate=ffc_use_highfreq_gate,
            highfreq_threshold=ffc_highfreq_threshold,
            gate_reduction=ffc_gate_reduction,
            local_kernel=ffc_local_kernel,
        )
    raise ValueError(f"Unsupported high_enhancer_type: {enhancer_type}")


class MIRFDBlock(nn.Module):
    """Mamba-induced residual frequency decoupling block."""

    SUPPORTED_RESIDUALS = {"mamba_residual", "avgpool", "laplace", "sobel", "pyramid_avgpool"}
    SUPPORTED_FUSIONS = {"concat", "residual_compensation"}
    SUPPORTED_HIGH_RESIDUAL_MODES = {"hfe", "concat_proj", "add", "add_scaled"}
    SUPPORTED_GATE_MODES = {"none", "suppress", "enhance", "half_enhance", "centered"}
    SUPPORTED_HIGH_ENHANCERS = {"identity", "conv_hfe", "freq_window", "ffc"}
    SUPPORTED_BLOCK_FUSION_HIGH_SOURCES = {"high_hat", "high_raw", "residual", "selected_residual"}

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
        use_low_smooth: bool = False,
        low_smooth_beta_init: float = 0.3,
        high_residual_mode: str = "hfe",
        hfe_scale_init: float = 0.1,
        high_enhancer_type: str = "conv_hfe",
        fsre_num_bands: int = 4,
        fsre_window_size: int = 8,
        fsre_gamma_init: float = 0.1,
        ffc_gamma_init: float = 0.1,
        ffc_use_highfreq_gate: bool = True,
        ffc_highfreq_threshold: float = 0.5,
        ffc_gate_reduction: int = 4,
        ffc_local_kernel: int = 3,
        ffc_fft_norm: str = "ortho",
        block_fusion_high_source: str = "high_hat",
        gate_mode: str = "suppress",
        gate_alpha_init: float = 1.0,
        gate_scale_min: float = 0.25,
        gate_scale_max: float = 1.75,
        use_context_residual_selector: bool = False,
        selector_gamma_init: float = 0.1,
        selector_use_reference: bool = False,
    ) -> None:
        super().__init__()
        if residual_type not in self.SUPPORTED_RESIDUALS:
            raise ValueError(f"Unsupported residual_type: {residual_type}")
        if fusion not in self.SUPPORTED_FUSIONS:
            raise ValueError(f"Unsupported fusion: {fusion}")
        if high_residual_mode not in self.SUPPORTED_HIGH_RESIDUAL_MODES:
            raise ValueError(f"Unsupported high_residual_mode: {high_residual_mode}")
        if high_enhancer_type not in self.SUPPORTED_HIGH_ENHANCERS:
            raise ValueError(f"Unsupported high_enhancer_type: {high_enhancer_type}")
        if block_fusion_high_source not in self.SUPPORTED_BLOCK_FUSION_HIGH_SOURCES:
            raise ValueError(
                f"Unsupported block_fusion_high_source: {block_fusion_high_source}. "
                f"Expected one of {sorted(self.SUPPORTED_BLOCK_FUSION_HIGH_SOURCES)}."
            )
        if gate_mode not in self.SUPPORTED_GATE_MODES:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        if gate_scale_max <= gate_scale_min:
            raise ValueError("gate_scale_max must be greater than gate_scale_min")

        mamba_block = mamba_block or build_mamba_block
        mamba_kwargs = mamba_kwargs or {}

        self.residual_type = residual_type
        self.fusion = fusion
        self.use_gate = use_gate
        self.high_residual_mode = high_residual_mode
        self.high_enhancer_type = high_enhancer_type
        self.block_fusion_high_source = block_fusion_high_source
        self.use_context_residual_selector = bool(use_context_residual_selector)
        self.gate_mode = gate_mode
        self.gate_scale_min = float(gate_scale_min)
        self.gate_scale_max = float(gate_scale_max)
        self.norm = make_norm(pre_norm, dim)
        self.mamba = mamba_block(dim, **mamba_kwargs)
        self.align = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            make_norm(norm, dim),
        )
        self.low_smooth = LowSmooth(dim, beta_init=low_smooth_beta_init) if use_low_smooth else nn.Identity()
        self.hfe = build_high_enhancer(
            high_enhancer_type,
            dim,
            hfe_kernels=hfe_kernels,
            norm=norm,
            fsre_num_bands=fsre_num_bands,
            fsre_window_size=fsre_window_size,
            fsre_gamma_init=fsre_gamma_init,
            ffc_gamma_init=ffc_gamma_init,
            ffc_use_highfreq_gate=ffc_use_highfreq_gate,
            ffc_highfreq_threshold=ffc_highfreq_threshold,
            ffc_gate_reduction=ffc_gate_reduction,
            ffc_local_kernel=ffc_local_kernel,
            ffc_fft_norm=ffc_fft_norm,
        )
        self.high_proj = (
            ConvNormAct(dim * 2, dim, kernel_size=1, padding=0, norm=norm)
            if high_residual_mode == "concat_proj"
            else nn.Identity()
        )
        if high_residual_mode == "add_scaled":
            self.hfe_scale = nn.Parameter(torch.tensor(float(hfe_scale_init)))
        self.gate = TargetAwareGate(dim, norm=norm) if use_gate and gate_mode != "none" else None
        self.gate_alpha = nn.Parameter(torch.tensor(float(gate_alpha_init)))
        self.residual_selector = (
            ContextGuidedResidualSelector(
                dim,
                use_reference=selector_use_reference,
                gamma_init=selector_gamma_init,
                norm=norm,
            )
            if use_context_residual_selector
            else None
        )

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

    def _low_and_residual(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.residual_type == "mamba_residual":
            fm = self.mamba(self.norm(x))
            low0 = self.align(fm)
            low = self.low_smooth(low0)
            residual = x - low
            return low0, low, residual

        if self.residual_type == "avgpool":
            low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
            residual = x - low
            return low, low, residual

        if self.residual_type == "laplace":
            residual = self._depthwise_fixed(x, self.laplace_kernel)
            low = x - residual
            return low, low, residual

        if self.residual_type == "sobel":
            grad_x = self._depthwise_fixed(x, self.sobel_x_kernel)
            grad_y = self._depthwise_fixed(x, self.sobel_y_kernel)
            residual = 0.5 * (grad_x.abs() + grad_y.abs())
            low = x - residual
            return low, low, residual

        pooled = F.avg_pool2d(x, kernel_size=2, stride=2)
        low = F.interpolate(pooled, size=x.shape[-2:], mode="nearest")
        residual = x - low
        return low, low, residual

    def _high_branch(self, residual: torch.Tensor) -> torch.Tensor:
        enhanced = self.hfe(residual)
        if self.high_residual_mode == "concat_proj":
            return self.high_proj(torch.cat([residual, enhanced], dim=1))
        if self.high_residual_mode == "add":
            return residual + enhanced
        if self.high_residual_mode == "add_scaled":
            scale = torch.clamp(self.hfe_scale, 0.0, 1.0)
            return residual + scale * enhanced
        return enhanced

    def _apply_gate(self, high_raw: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "none":
            return high_raw
        if self.gate_mode == "suppress":
            return gate * high_raw
        if self.gate_mode == "half_enhance":
            return (0.5 + gate) * high_raw
        if self.gate_mode == "centered":
            alpha = torch.clamp(self.gate_alpha, 0.0, 2.0)
            scale = 1.0 + alpha * (gate - 0.5)
            scale = torch.clamp(scale, self.gate_scale_min, self.gate_scale_max)
            return scale * high_raw
        alpha = torch.clamp(self.gate_alpha, 0.0, 2.0)
        return (1.0 + alpha * gate) * high_raw

    def _select_high_for_fusion(
        self,
        residual: torch.Tensor,
        high_raw: torch.Tensor,
        high_hat: torch.Tensor,
        selected_residual: torch.Tensor,
    ) -> torch.Tensor:
        if self.block_fusion_high_source == "high_hat":
            return high_hat
        if self.block_fusion_high_source == "high_raw":
            return high_raw
        if self.block_fusion_high_source == "residual":
            return residual
        if self.block_fusion_high_source == "selected_residual":
            return selected_residual
        raise RuntimeError(f"Invalid block_fusion_high_source: {self.block_fusion_high_source}")

    def forward(
        self,
        x: torch.Tensor,
        return_branches: bool = False,
        selector_reference: torch.Tensor | None = None,
    ):
        low0, low, residual = self._low_and_residual(x)
        high_raw = self._high_branch(residual)
        selector_enabled = self.residual_selector is not None
        selector_reference_used = bool(
            selector_enabled
            and self.residual_selector is not None
            and self.residual_selector.use_reference
            and selector_reference is not None
        )
        if self.residual_selector is None:
            selected_residual = residual
            selector = torch.ones_like(residual)
        else:
            selected_residual, selector = self.residual_selector(low, residual, reference=selector_reference)
        if self.gate is None:
            gate = torch.ones_like(high_raw)
            high_hat = high_raw
        else:
            gate = self.gate(low, residual)
            high_hat = self._apply_gate(high_raw, gate)
        high_for_fusion = self._select_high_for_fusion(
            residual=residual,
            high_raw=high_raw,
            high_hat=high_hat,
            selected_residual=selected_residual,
        )

        if self.fusion == "concat":
            out = self.fuse(torch.cat([low, high_for_fusion], dim=1)) + x
        else:
            out = x + low + self.gamma * high_for_fusion

        if return_branches:
            return out, {
                "low0": low0,
                "low": low,
                "high": high_for_fusion,
                "high_for_fusion": high_for_fusion,
                "high_raw": high_raw,
                "high_hat": high_hat,
                "residual": residual,
                "selected_residual": selected_residual,
                "selector": selector,
                "selector_enabled": selector_enabled,
                "selector_use_reference": bool(
                    self.residual_selector.use_reference if self.residual_selector is not None else False
                ),
                "selector_reference_used": selector_reference_used,
                "gate": gate,
                "block_fusion_high_source": self.block_fusion_high_source,
            }
        return out
