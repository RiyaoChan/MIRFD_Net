from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .layers import ConvNormAct
from .mirfd_block import FixedDepthwiseBlur, MIRFDBlock, build_high_enhancer


class ConvStage(nn.Module):
    def __init__(self, dim: int, depth: int, norm: str = "batch") -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[ConvNormAct(dim, dim, kernel_size=3, norm=norm) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MIRFDStage(nn.Module):
    def __init__(self, dim: int, depth: int, block_kwargs: dict[str, Any], stage: int) -> None:
        super().__init__()
        self.stage = int(stage)
        kwargs = dict(block_kwargs)
        selector_stages = kwargs.pop("selector_stages", None)
        if selector_stages is not None:
            selector_enabled = int(stage) in {int(item) for item in selector_stages}
            kwargs["use_context_residual_selector"] = bool(
                kwargs.get("use_context_residual_selector", False) and selector_enabled
            )
        self.blocks = nn.ModuleList([MIRFDBlock(dim, **kwargs) for _ in range(depth)])

    def forward(
        self,
        x: torch.Tensor,
        return_branches: bool = False,
        selector_reference: torch.Tensor | None = None,
    ):
        last = None
        for block in self.blocks:
            if return_branches:
                x, last = block(x, return_branches=True, selector_reference=selector_reference)
            else:
                x = block(x, selector_reference=selector_reference)
        if return_branches:
            if last is None:
                last = {
                    "low0": x,
                    "low": x,
                    "high": torch.zeros_like(x),
                    "high_for_fusion": torch.zeros_like(x),
                    "high_raw": torch.zeros_like(x),
                    "high_hat": torch.zeros_like(x),
                    "selected_residual": torch.zeros_like(x),
                    "selector": torch.ones_like(x),
                    "selector_enabled": False,
                    "selector_use_reference": False,
                    "selector_reference_used": False,
                    "residual": torch.zeros_like(x),
                    "gate": torch.ones_like(x),
                    "block_fusion_high_source": "high_hat",
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
    SUPPORTED_HIGH_SKIP_STAGES = {1, 2, 3}
    SUPPORTED_DECODER_HIGH_SOURCES = {"high_raw", "high_hat", "residual"}
    SUPPORTED_SELECTOR_REFERENCE_SOURCES = {"stage1", "stage1_residual", "stage1_high"}
    SUPPORTED_SELECTOR_REFERENCE_HFMS = {"none", "avgpool"}

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
        decoder_high_source: str = "high_hat",
        stage1_high_enhancer_type: str = "conv_hfe",
        stage1_high_enhancer_kwargs: dict[str, Any] | None = None,
        selector_reference_source: str = "stage1",
        selector_reference_hfm: str = "avgpool",
        use_aux_heads: bool = True,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        if decoder_high_source not in self.SUPPORTED_DECODER_HIGH_SOURCES:
            raise ValueError(f"Unsupported decoder_high_source: {decoder_high_source}")
        if selector_reference_source not in self.SUPPORTED_SELECTOR_REFERENCE_SOURCES:
            raise ValueError(f"Unsupported selector_reference_source: {selector_reference_source}")
        if selector_reference_hfm not in self.SUPPORTED_SELECTOR_REFERENCE_HFMS:
            raise ValueError(f"Unsupported selector_reference_hfm: {selector_reference_hfm}")
        dims = dims or (base_dim, base_dim * 2, base_dim * 4, base_dim * 8)
        block_kwargs = block_kwargs or {}
        stage1_high_enhancer_kwargs = stage1_high_enhancer_kwargs or {}
        self.use_aux_heads = use_aux_heads
        self.use_high_residual_skip = use_high_residual_skip
        self.decoder_high_source = decoder_high_source
        self.selector_reference_source = selector_reference_source
        self.selector_reference_hfm = selector_reference_hfm
        selector_stage_cfg = block_kwargs.get("selector_stages")
        if selector_stage_cfg is None:
            selector_stages = {2, 3, 4}
        else:
            selector_stages = {int(stage) for stage in selector_stage_cfg}
        invalid_selector_stages = selector_stages - {2, 3, 4}
        if invalid_selector_stages:
            raise ValueError(
                f"Unsupported selector_stages: {sorted(invalid_selector_stages)}. "
                "CGRS currently applies only to MIRFD stages {2, 3, 4}."
            )
        self.selector_stages = selector_stages
        self.selector_use_reference = bool(
            block_kwargs.get("use_context_residual_selector", False)
            and block_kwargs.get("selector_use_reference", False)
            and selector_stages
        )
        if high_skip_stages is None:
            stages = {2, 3} if use_high_residual_skip else set()
            if use_stage1_high_skip:
                stages.add(1)
        else:
            stages = {int(stage) for stage in high_skip_stages}
            invalid_stages = stages - self.SUPPORTED_HIGH_SKIP_STAGES
            if invalid_stages:
                raise ValueError(
                    f"Unsupported high_skip_stages: {sorted(invalid_stages)}. "
                    "Only decoder high skip stages {1, 2, 3} are supported. "
                    "Stage 4 high_hat is available for diagnostics and auxiliary heads, "
                    "but it is not injected into the decoder unless bottleneck high injection is implemented."
                )
        if stages and not use_high_residual_skip:
            raise ValueError(
                "high_skip_stages requires decoder.use_high_residual_skip=True; "
                f"got use_high_residual_skip=False with high_skip_stages={sorted(stages)}."
            )
        self.high_skip_stages = stages
        self.use_stage1_high_skip = 1 in self.high_skip_stages

        self.stem = ConvNormAct(in_channels, dims[0], kernel_size=3, stride=2, norm=norm)
        self.stage1 = ConvStage(dims[0], depths[0], norm=norm)
        needs_stage1_residual = self.use_stage1_high_skip or (
            self.selector_use_reference and selector_reference_source in {"stage1_residual", "stage1_high"}
        )
        needs_stage1_high = self.use_stage1_high_skip or (
            self.selector_use_reference and selector_reference_source == "stage1_high"
        )
        if needs_stage1_residual:
            self.stage1_blur = FixedDepthwiseBlur(dims[0])
        else:
            self.stage1_blur = nn.Identity()
        if needs_stage1_high:
            self.stage1_hfe = build_high_enhancer(
                stage1_high_enhancer_type,
                dims[0],
                norm=norm,
                **stage1_high_enhancer_kwargs,
            )
        else:
            self.stage1_hfe = nn.Identity()
        self.selector_reference_projs = nn.ModuleDict()
        if self.selector_use_reference:
            stage_dims = {2: dims[1], 3: dims[2], 4: dims[3]}
            self.selector_reference_projs = nn.ModuleDict(
                {str(stage): nn.Conv2d(dims[0], stage_dims[stage], kernel_size=1) for stage in sorted(selector_stages)}
            )

        self.down12 = ConvNormAct(dims[0], dims[1], kernel_size=3, stride=2, norm=norm)
        self.stage2 = MIRFDStage(dims[1], depths[1], block_kwargs, stage=2)

        self.down23 = ConvNormAct(dims[1], dims[2], kernel_size=3, stride=2, norm=norm)
        self.stage3 = MIRFDStage(dims[2], depths[2], block_kwargs, stage=3)

        self.down34 = ConvNormAct(dims[2], dims[3], kernel_size=3, stride=2, norm=norm)
        self.stage4 = MIRFDStage(dims[3], depths[3], block_kwargs, stage=4)

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

    def _decoder_branch(self, branches: dict[str, torch.Tensor] | None) -> torch.Tensor | None:
        if branches is None:
            return None
        return branches[self.decoder_high_source]

    @staticmethod
    def _avgpool_high_pass(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] < 2 or x.shape[-1] < 2:
            return torch.zeros_like(x)
        low = F.avg_pool2d(x, kernel_size=2, stride=2)
        low = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x - low

    def _stage1_reference(
        self,
        e1: torch.Tensor,
        e1_residual: torch.Tensor,
        e1_high: torch.Tensor,
    ) -> torch.Tensor:
        if self.selector_reference_source == "stage1":
            reference = e1
        elif self.selector_reference_source == "stage1_residual":
            reference = e1_residual
        elif self.selector_reference_source == "stage1_high":
            reference = e1_high
        else:  # pragma: no cover - guarded in __init__
            raise RuntimeError(f"Invalid selector_reference_source: {self.selector_reference_source}")
        if self.selector_reference_hfm == "avgpool":
            reference = self._avgpool_high_pass(reference)
        return reference

    def _selector_reference_for_stage(
        self,
        stage1_reference: torch.Tensor | None,
        stage: int,
        target: torch.Tensor,
    ) -> torch.Tensor | None:
        if stage1_reference is None or not self.selector_use_reference or stage not in self.selector_stages:
            return None
        stage_key = str(stage)
        if stage_key not in self.selector_reference_projs:
            return None
        proj = self.selector_reference_projs[stage_key]
        reference = F.interpolate(stage1_reference, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return proj(reference)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_dict: bool | None = None,
    ):
        input_size = x.shape[-2:]
        collect = return_features or self.use_aux_heads or bool(self.high_skip_stages & {2, 3})

        e1 = self.stage1(self.stem(x))
        need_stage1_branches = return_features or self.use_stage1_high_skip or self.selector_use_reference
        if need_stage1_branches:
            e1_low = self.stage1_blur(e1)
            e1_residual = e1 - e1_low
            e1_high = self.stage1_hfe(e1_residual)
        else:
            e1_low = e1
            e1_residual = torch.zeros_like(e1)
            e1_high = torch.zeros_like(e1)
        stage1_reference = (
            self._stage1_reference(e1, e1_residual, e1_high)
            if self.selector_use_reference
            else None
        )
        e2_in = self.down12(e1)
        ref2 = self._selector_reference_for_stage(stage1_reference, 2, e2_in)
        e2, b2 = (
            self.stage2(e2_in, return_branches=True, selector_reference=ref2)
            if collect
            else (self.stage2(e2_in, selector_reference=ref2), None)
        )
        e3_in = self.down23(e2)
        ref3 = self._selector_reference_for_stage(stage1_reference, 3, e3_in)
        e3, b3 = (
            self.stage3(e3_in, return_branches=True, selector_reference=ref3)
            if collect
            else (self.stage3(e3_in, selector_reference=ref3), None)
        )
        e4_in = self.down34(e3)
        ref4 = self._selector_reference_for_stage(stage1_reference, 4, e4_in)
        e4, b4 = (
            self.stage4(e4_in, return_branches=True, selector_reference=ref4)
            if collect
            else (self.stage4(e4_in, selector_reference=ref4), None)
        )

        h3 = self._decoder_branch(b3) if (b3 is not None and 3 in self.high_skip_stages) else None
        h2 = self._decoder_branch(b2) if (b2 is not None and 2 in self.high_skip_stages) else None

        d3 = self.dec3(e4, e3, h3)
        d2 = self.dec2(d3, e2, h2)
        if 1 in self.high_skip_stages:
            e1_decoder_high = e1_residual if self.decoder_high_source == "residual" else e1_high
        else:
            e1_decoder_high = e1_high
        d1 = self.dec1(d2, e1, e1_decoder_high)
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
                "high": [b2["high"], b3["high"], b4["high"]],
                "high_for_fusion": [b2["high_for_fusion"], b3["high_for_fusion"], b4["high_for_fusion"]],
                "high_raw": [b2["high_raw"], b3["high_raw"], b4["high_raw"]],
                "high_hat": [b2["high_hat"], b3["high_hat"], b4["high_hat"]],
                "residual": [b2["residual"], b3["residual"], b4["residual"]],
                "selected_residual": [b2["selected_residual"], b3["selected_residual"], b4["selected_residual"]],
                "selector": [b2["selector"], b3["selector"], b4["selector"]],
                "selector_enabled": [
                    b2["selector_enabled"],
                    b3["selector_enabled"],
                    b4["selector_enabled"],
                ],
                "selector_use_reference": [
                    b2["selector_use_reference"],
                    b3["selector_use_reference"],
                    b4["selector_use_reference"],
                ],
                "selector_reference_used": [
                    b2["selector_reference_used"],
                    b3["selector_reference_used"],
                    b4["selector_reference_used"],
                ],
                "gate": [b2["gate"], b3["gate"], b4["gate"]],
                "block_fusion_high_source": [
                    b2["block_fusion_high_source"],
                    b3["block_fusion_high_source"],
                    b4["block_fusion_high_source"],
                ],
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
        "high_enhancer_type": mirfd_cfg.get("high_enhancer_type", "conv_hfe"),
        "fsre_num_bands": mirfd_cfg.get("fsre_num_bands", 4),
        "fsre_window_size": mirfd_cfg.get("fsre_window_size", 8),
        "fsre_gamma_init": mirfd_cfg.get("fsre_gamma_init", 0.1),
        "ffc_gamma_init": mirfd_cfg.get("ffc_gamma_init", 0.1),
        "ffc_use_highfreq_gate": mirfd_cfg.get("ffc_use_highfreq_gate", True),
        "ffc_highfreq_threshold": mirfd_cfg.get("ffc_highfreq_threshold", 0.5),
        "ffc_gate_reduction": mirfd_cfg.get("ffc_gate_reduction", 4),
        "ffc_local_kernel": mirfd_cfg.get("ffc_local_kernel", 3),
        "ffc_fft_norm": mirfd_cfg.get("ffc_fft_norm", "ortho"),
        "block_fusion_high_source": mirfd_cfg.get("block_fusion_high_source", "high_hat"),
        "gate_mode": mirfd_cfg.get("gate_mode", "suppress"),
        "gate_alpha_init": mirfd_cfg.get("gate_alpha_init", 1.0),
        "gate_scale_min": mirfd_cfg.get("gate_scale_min", 0.25),
        "gate_scale_max": mirfd_cfg.get("gate_scale_max", 1.75),
        "use_context_residual_selector": mirfd_cfg.get("use_context_residual_selector", False),
        "selector_stages": mirfd_cfg.get("selector_stages"),
        "selector_gamma_init": mirfd_cfg.get("selector_gamma_init", 0.1),
        "selector_use_reference": mirfd_cfg.get("selector_use_reference", False),
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
    stage1_high_enhancer_kwargs = {
        "hfe_kernels": tuple(mirfd_cfg.get("hfe_kernels", (3, 5))),
        "fsre_num_bands": mirfd_cfg.get("stage1_fsre_num_bands", mirfd_cfg.get("fsre_num_bands", 4)),
        "fsre_window_size": mirfd_cfg.get("stage1_fsre_window_size", mirfd_cfg.get("fsre_window_size", 8)),
        "fsre_gamma_init": mirfd_cfg.get("stage1_fsre_gamma_init", mirfd_cfg.get("fsre_gamma_init", 0.1)),
        "ffc_gamma_init": mirfd_cfg.get("stage1_ffc_gamma_init", mirfd_cfg.get("ffc_gamma_init", 0.1)),
        "ffc_use_highfreq_gate": mirfd_cfg.get(
            "stage1_ffc_use_highfreq_gate",
            mirfd_cfg.get("ffc_use_highfreq_gate", True),
        ),
        "ffc_highfreq_threshold": mirfd_cfg.get(
            "stage1_ffc_highfreq_threshold",
            mirfd_cfg.get("ffc_highfreq_threshold", 0.5),
        ),
        "ffc_gate_reduction": mirfd_cfg.get("stage1_ffc_gate_reduction", mirfd_cfg.get("ffc_gate_reduction", 4)),
        "ffc_local_kernel": mirfd_cfg.get("stage1_ffc_local_kernel", mirfd_cfg.get("ffc_local_kernel", 3)),
        "ffc_fft_norm": mirfd_cfg.get("stage1_ffc_fft_norm", mirfd_cfg.get("ffc_fft_norm", "ortho")),
    }
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
        decoder_high_source=model_cfg.get("decoder_high_source", decoder_cfg.get("decoder_high_source", "high_hat")),
        stage1_high_enhancer_type=model_cfg.get("stage1_high_enhancer_type", "conv_hfe"),
        stage1_high_enhancer_kwargs=stage1_high_enhancer_kwargs,
        selector_reference_source=mirfd_cfg.get(
            "selector_reference_source",
            mirfd_cfg.get("reference_source", "stage1"),
        ),
        selector_reference_hfm=mirfd_cfg.get(
            "selector_reference_hfm",
            mirfd_cfg.get("reference_hfm", "avgpool"),
        ),
        use_aux_heads=model_cfg.get("use_aux_heads", True),
        norm=model_cfg.get("norm", "batch"),
    )
