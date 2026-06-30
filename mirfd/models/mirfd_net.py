from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .layers import ConvNormAct
from .mirfd_block import FixedDepthwiseBlur, HighFrequencyEnhancer, MIRFDBlock


class ConvStage(nn.Module):
    def __init__(self, dim: int, depth: int, norm: str = "batch") -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[ConvNormAct(dim, dim, kernel_size=3, norm=norm) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MIRFDStage(nn.Module):
    def __init__(self, dim: int, depth: int, block_kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([MIRFDBlock(dim, **block_kwargs) for _ in range(depth)])

    def forward(self, x: torch.Tensor, return_branches: bool = False):
        last = None
        for block in self.blocks:
            if return_branches:
                x, last = block(x, return_branches=True)
            else:
                x = block(x)
        if return_branches:
            if last is None:
                last = {
                    "low0": x,
                    "low": x,
                    "high": torch.zeros_like(x),
                    "high_raw": torch.zeros_like(x),
                    "high_hat": torch.zeros_like(x),
                    "residual": torch.zeros_like(x),
                    "gate": torch.ones_like(x),
                }
            return x, last
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        high_dim: int,
        skip_dim: int,
        out_dim: int,
        use_high_residual_skip: bool = True,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        in_dim = high_dim + skip_dim + (skip_dim if use_high_residual_skip else 0)
        self.use_high_residual_skip = use_high_residual_skip
        self.fuse = nn.Sequential(
            ConvNormAct(in_dim, out_dim, kernel_size=3, norm=norm),
            ConvNormAct(out_dim, out_dim, kernel_size=3, norm=norm),
        )

    def forward(
        self,
        high: torch.Tensor,
        skip: torch.Tensor,
        high_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        high = F.interpolate(high, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        parts = [high, skip]
        if self.use_high_residual_skip:
            if high_residual is None:
                high_residual = torch.zeros_like(skip)
            parts.append(high_residual)
        return self.fuse(torch.cat(parts, dim=1))


class SegHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 1, norm: str = "batch") -> None:
        super().__init__()
        hidden = max(in_dim // 2, num_classes)
        self.head = nn.Sequential(
            ConvNormAct(in_dim, hidden, kernel_size=3, norm=norm),
            nn.Conv2d(hidden, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MIRFDNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_dim: int = 32,
        dims: tuple[int, int, int, int] | None = None,
        depths: tuple[int, int, int, int] = (1, 1, 2, 2),
        block_kwargs: dict[str, Any] | None = None,
        use_high_residual_skip: bool = True,
        use_stage1_high_skip: bool = False,
        high_skip_stages: Iterable[int] | None = None,
        use_aux_heads: bool = True,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        dims = dims or (base_dim, base_dim * 2, base_dim * 4, base_dim * 8)
        block_kwargs = block_kwargs or {}
        self.use_aux_heads = use_aux_heads
        self.use_high_residual_skip = use_high_residual_skip
        if high_skip_stages is None:
            stages = {2, 3} if use_high_residual_skip else set()
            if use_stage1_high_skip:
                stages.add(1)
        else:
            stages = {int(stage) for stage in high_skip_stages}
            invalid_stages = stages - {1, 2, 3, 4}
            if invalid_stages:
                raise ValueError(f"Unsupported high_skip_stages: {sorted(invalid_stages)}")
        self.high_skip_stages = stages
        self.use_stage1_high_skip = 1 in self.high_skip_stages

        self.stem = ConvNormAct(in_channels, dims[0], kernel_size=3, stride=2, norm=norm)
        self.stage1 = ConvStage(dims[0], depths[0], norm=norm)
        if self.use_stage1_high_skip:
            self.stage1_blur = FixedDepthwiseBlur(dims[0])
            self.stage1_hfe = HighFrequencyEnhancer(dims[0], norm=norm)
        else:
            self.stage1_blur = nn.Identity()
            self.stage1_hfe = nn.Identity()

        self.down12 = ConvNormAct(dims[0], dims[1], kernel_size=3, stride=2, norm=norm)
        self.stage2 = MIRFDStage(dims[1], depths[1], block_kwargs)

        self.down23 = ConvNormAct(dims[1], dims[2], kernel_size=3, stride=2, norm=norm)
        self.stage3 = MIRFDStage(dims[2], depths[2], block_kwargs)

        self.down34 = ConvNormAct(dims[2], dims[3], kernel_size=3, stride=2, norm=norm)
        self.stage4 = MIRFDStage(dims[3], depths[3], block_kwargs)

        self.dec3 = DecoderBlock(dims[3], dims[2], dims[2], use_high_residual_skip, norm=norm)
        self.dec2 = DecoderBlock(dims[2], dims[1], dims[1], use_high_residual_skip, norm=norm)
        self.dec1 = DecoderBlock(dims[1], dims[0], dims[0], use_high_residual_skip, norm=norm)
        self.head = SegHead(dims[0], num_classes=num_classes, norm=norm)

        if use_aux_heads:
            self.aux_heads = nn.ModuleList(
                [nn.Conv2d(dims[1], num_classes, 1), nn.Conv2d(dims[2], num_classes, 1), nn.Conv2d(dims[3], num_classes, 1)]
            )
        else:
            self.aux_heads = nn.ModuleList()

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_dict: bool | None = None,
    ):
        input_size = x.shape[-2:]
        collect = return_features or self.use_aux_heads or bool(self.high_skip_stages & {2, 3, 4})

        e1 = self.stage1(self.stem(x))
        e2_in = self.down12(e1)
        e2, b2 = self.stage2(e2_in, return_branches=True) if collect else (self.stage2(e2_in), None)
        e3_in = self.down23(e2)
        e3, b3 = self.stage3(e3_in, return_branches=True) if collect else (self.stage3(e3_in), None)
        e4_in = self.down34(e3)
        e4, b4 = self.stage4(e4_in, return_branches=True) if collect else (self.stage4(e4_in), None)

        h3 = b3["high_hat"] if (b3 is not None and 3 in self.high_skip_stages) else None
        h2 = b2["high_hat"] if (b2 is not None and 2 in self.high_skip_stages) else None

        d3 = self.dec3(e4, e3, h3)
        d2 = self.dec2(d3, e2, h2)
        if 1 in self.high_skip_stages:
            e1_low = self.stage1_blur(e1)
            e1_residual = e1 - e1_low
            e1_high = self.stage1_hfe(e1_residual)
        else:
            e1_low = e1
            e1_residual = torch.zeros_like(e1)
            e1_high = torch.zeros_like(e1)
        d1 = self.dec1(d2, e1, e1_high)
        logits = self.head(d1)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        output = {"logits": logits}
        if self.use_aux_heads:
            output["aux_logits"] = [
                head(feat)
                for head, feat in zip(self.aux_heads, [b2["high_hat"], b3["high_hat"], b4["high_hat"]])
            ]
        if return_features:
            output["features"] = {
                "low0": [b2["low0"], b3["low0"], b4["low0"]],
                "low": [b2["low"], b3["low"], b4["low"]],
                "high": [b2["high_hat"], b3["high_hat"], b4["high_hat"]],
                "high_raw": [b2["high_raw"], b3["high_raw"], b4["high_raw"]],
                "high_hat": [b2["high_hat"], b3["high_hat"], b4["high_hat"]],
                "residual": [b2["residual"], b3["residual"], b4["residual"]],
                "gate": [b2["gate"], b3["gate"], b4["gate"]],
                "stage1_low": [e1_low],
                "stage1_residual": [e1_residual],
                "stage1_high": [e1_high],
            }

        if return_dict is None:
            return_dict = self.use_aux_heads or return_features
        return output if return_dict else logits


def build_model(config: dict[str, Any] | None = None) -> MIRFDNet:
    config = config or {}
    model_cfg = config.get("model", config)
    mirfd_cfg = model_cfg.get("mirfd", {})
    mamba_cfg = model_cfg.get("mamba", {})
    decoder_cfg = model_cfg.get("decoder", {})

    block_kwargs = {
        "residual_type": mirfd_cfg.get("residual_type", "mamba_residual"),
        "fusion": mirfd_cfg.get("fusion", "concat"),
        "use_gate": mirfd_cfg.get("use_gate", True),
        "use_learnable_gamma": mirfd_cfg.get("use_learnable_gamma", False),
        "hfe_kernels": tuple(mirfd_cfg.get("hfe_kernels", (3, 5))),
        "norm": model_cfg.get("norm", "batch"),
        "pre_norm": mirfd_cfg.get("pre_norm", "layer"),
        "use_low_smooth": mirfd_cfg.get("use_low_smooth", False),
        "low_smooth_beta_init": mirfd_cfg.get("low_smooth_beta_init", 0.3),
        "high_residual_mode": mirfd_cfg.get("high_residual_mode", "hfe"),
        "hfe_scale_init": mirfd_cfg.get("hfe_scale_init", 0.1),
        "gate_mode": mirfd_cfg.get("gate_mode", "suppress"),
        "gate_alpha_init": mirfd_cfg.get("gate_alpha_init", 1.0),
        "gate_scale_min": mirfd_cfg.get("gate_scale_min", 0.25),
        "gate_scale_max": mirfd_cfg.get("gate_scale_max", 1.75),
        "mamba_kwargs": {
            "variant": mamba_cfg.get("variant", "fallback"),
            "expansion": mamba_cfg.get("expansion", 2.0),
            "conv_kernel": mamba_cfg.get("conv_kernel", 3),
            "dropout": mamba_cfg.get("dropout", 0.0),
            "scan_backend": mamba_cfg.get("scan_backend", "auto"),
            "decay_init": mamba_cfg.get("decay_init", 0.75),
            "d_state": mamba_cfg.get("d_state", 16),
            "dt_rank": mamba_cfg.get("dt_rank", "auto"),
            "dt_min": mamba_cfg.get("dt_min", 0.001),
            "dt_max": mamba_cfg.get("dt_max", 0.1),
            "dt_init": mamba_cfg.get("dt_init", "random"),
            "dt_scale": mamba_cfg.get("dt_scale", 1.0),
            "dt_init_floor": mamba_cfg.get("dt_init_floor", 1e-4),
            "bias": mamba_cfg.get("bias", False),
            "conv_bias": mamba_cfg.get("conv_bias", True),
            "external_import_path": mamba_cfg.get("external_import_path"),
            "external_layout": mamba_cfg.get("external_layout", "auto"),
            "external_kwargs": mamba_cfg.get("external_kwargs", {}),
            "parallel_real_variant": mamba_cfg.get("parallel_real_variant", "ss2d"),
            "parallel_fusion": mamba_cfg.get("parallel_fusion", "concat"),
        },
    }

    dims = model_cfg.get("dims")
    depths = model_cfg.get("depths", (1, 1, 2, 2))
    return MIRFDNet(
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 1),
        base_dim=model_cfg.get("base_dim", 32),
        dims=tuple(dims) if dims is not None else None,
        depths=tuple(depths),
        block_kwargs=block_kwargs,
        use_high_residual_skip=decoder_cfg.get("use_high_residual_skip", True),
        use_stage1_high_skip=model_cfg.get("use_stage1_high_skip", decoder_cfg.get("use_stage1_high_skip", False)),
        high_skip_stages=model_cfg.get("high_skip_stages", decoder_cfg.get("high_skip_stages")),
        use_aux_heads=model_cfg.get("use_aux_heads", True),
        norm=model_cfg.get("norm", "batch"),
    )
