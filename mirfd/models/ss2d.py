from __future__ import annotations

from collections.abc import Callable
import importlib
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .mamba2d import Mamba2D


def _import_object(import_path: str) -> type[nn.Module]:
    module_name, _, object_name = import_path.rpartition(".")
    if not module_name or not object_name:
        raise ValueError(f"Invalid import path: {import_path}")
    module = importlib.import_module(module_name)
    obj = getattr(module, object_name)
    if not issubclass(obj, nn.Module):
        raise TypeError(f"{import_path} is not an nn.Module")
    return obj


def _selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    a = a.float()
    b = b.float()
    c = c.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float().view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, seqlen = u.shape
    d_state = a.shape[1]
    if b.dim() == 4:
        b = b.repeat_interleave(dim // b.shape[1], dim=1)
    if c.dim() == 4:
        c = c.repeat_interleave(dim // c.shape[1], dim=1)

    state = a.new_zeros((batch, dim, d_state))
    outputs = []
    last_state = None
    for idx in range(seqlen):
        delta_i = delta[:, :, idx]
        u_i = u[:, :, idx]
        delta_a = torch.exp(delta_i.unsqueeze(-1) * a.unsqueeze(0))
        if b.dim() == 2:
            delta_b_u = delta_i.unsqueeze(-1) * b.unsqueeze(0) * u_i.unsqueeze(-1)
        elif b.dim() == 3:
            delta_b_u = delta_i.unsqueeze(-1) * b[:, :, idx].unsqueeze(1) * u_i.unsqueeze(-1)
        else:
            delta_b_u = delta_i.unsqueeze(-1) * b[:, :, :, idx] * u_i.unsqueeze(-1)

        state = delta_a * state + delta_b_u
        if c.dim() == 2:
            y = torch.einsum("bdn,dn->bd", state, c)
        elif c.dim() == 3:
            y = torch.einsum("bdn,bn->bd", state, c[:, :, idx])
        else:
            y = torch.einsum("bdn,bdn->bd", state, c[:, :, :, idx])
        if idx == seqlen - 1:
            last_state = state
        outputs.append(y)

    out = torch.stack(outputs, dim=2)
    if d is not None:
        out = out + u * d.float().view(1, -1, 1)
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def _resolve_selective_scan(scan_backend: str = "auto") -> Callable:
    scan_backend = scan_backend.lower()
    if scan_backend == "ref":
        return _selective_scan_ref

    candidates = [
        "mamba_ssm.ops.selective_scan_interface.selective_scan_fn",
        "mamba_ssm.ops.selective_scan_interface.selective_scan_ref",
    ]
    errors = []
    for path in candidates:
        module_name, _, object_name = path.rpartition(".")
        try:
            module = importlib.import_module(module_name)
            return getattr(module, object_name)
        except Exception as exc:  # pragma: no cover - optional dependency path
            errors.append(f"{path}: {exc}")
    if scan_backend == "auto":
        return _selective_scan_ref
    joined = "\n".join(errors)
    raise ImportError(
        "SS2D requires mamba-ssm with selective_scan_fn/selective_scan_ref. "
        "Install a VMamba-compatible mamba_ssm/selective_scan environment or "
        "set model.mamba.scan_backend=ref or model.mamba.variant=fallback.\n"
        f"Tried:\n{joined}"
    )


def _dt_init(
    dt_rank: int,
    dim_inner: int,
    dt_scale: float,
    dt_init: str,
    dt_min: float,
    dt_max: float,
    dt_init_floor: float,
) -> nn.Linear:
    dt_proj = nn.Linear(dt_rank, dim_inner, bias=True)
    dt_init_std = dt_rank**-0.5 * dt_scale
    if dt_init == "constant":
        nn.init.constant_(dt_proj.weight, dt_init_std)
    elif dt_init == "random":
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
    else:
        raise ValueError(f"Unsupported dt_init: {dt_init}")

    dt = torch.exp(
        torch.rand(dim_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
        dt_proj.bias.copy_(inv_dt)
    return dt_proj


def _a_log_init(d_state: int, dim_inner: int, copies: int) -> nn.Parameter:
    a = torch.arange(1, d_state + 1, dtype=torch.float32)
    a = a.view(1, -1).repeat(dim_inner, 1).contiguous()
    a_log = torch.log(a)
    a_log = a_log.repeat(copies, 1)
    return nn.Parameter(a_log)


class SS2D(nn.Module):
    """VMamba-style SS2D block with cross selective scan.

    This module keeps the MIRFD BCHW contract while following the SS2D layout:
    input projection, depth-wise local mixing, four-direction cross scan,
    selective scan, cross merge, gating, and output projection.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        expansion: float = 2.0,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_kernel: int = 3,
        dropout: float = 0.0,
        scan_backend: str = "auto",
        bias: bool = False,
        conv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.selective_scan = _resolve_selective_scan(scan_backend)
        self.scan_backend = scan_backend
        self.dim = dim
        self.d_state = d_state
        self.dim_inner = int(expansion * dim)
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else int(dt_rank)
        self.k_group = 4

        self.in_proj = nn.Linear(dim, self.dim_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            self.dim_inner,
            self.dim_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=self.dim_inner,
            bias=conv_bias,
        )
        self.act = nn.SiLU(inplace=False)

        x_proj = [
            nn.Linear(self.dim_inner, self.dt_rank + d_state * 2, bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in x_proj], dim=0))

        dt_projs = [
            _dt_init(self.dt_rank, self.dim_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(self.k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([layer.weight for layer in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([layer.bias for layer in dt_projs], dim=0))

        self.a_logs = _a_log_init(d_state, self.dim_inner, copies=self.k_group)
        self.ds = nn.Parameter(torch.ones(self.k_group * self.dim_inner))
        self.out_norm = nn.LayerNorm(self.dim_inner)
        self.out_proj = nn.Linear(self.dim_inner, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @staticmethod
    def _cross_scan(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        l = h * w
        xs = torch.stack(
            [
                x.reshape(b, c, l),
                x.transpose(2, 3).contiguous().reshape(b, c, l),
            ],
            dim=1,
        )
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)
        return xs

    @staticmethod
    def _cross_merge(ys: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, k, d, l = ys.shape
        inv_y = torch.flip(ys[:, 2:4], dims=[-1])
        wh_y = ys[:, 1].view(b, d, w, h).transpose(2, 3).contiguous().view(b, d, l)
        inv_wh_y = inv_y[:, 1].view(b, d, w, h).transpose(2, 3).contiguous().view(b, d, l)
        return ys[:, 0] + inv_y[:, 0] + wh_y + inv_wh_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        xz = self.in_proj(x_nhwc)
        x_part, z = xz.chunk(2, dim=-1)

        x_part = x_part.permute(0, 3, 1, 2).contiguous()
        x_part = self.act(self.conv2d(x_part))
        xs = self._cross_scan(x_part)

        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight)
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)

        xs_flat = xs.float().view(b, -1, h * w)
        dts_flat = dts.contiguous().float().view(b, -1, h * w)
        bs = bs.float().contiguous()
        cs = cs.float().contiguous()
        a_s = -torch.exp(self.a_logs.float())
        d_s = self.ds.float()
        dt_bias = self.dt_projs_bias.float().view(-1)

        y = self.selective_scan(
            xs_flat,
            dts_flat,
            a_s,
            bs,
            cs,
            d_s,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        if isinstance(y, tuple):
            y = y[0]
        y = y.view(b, self.k_group, self.dim_inner, h * w)
        y = self._cross_merge(y, h, w)
        y = y.transpose(1, 2).contiguous().view(b, h, w, self.dim_inner)
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y.permute(0, 3, 1, 2).contiguous()


class ExternalVMambaBlock(nn.Module):
    """Adapter for an externally installed VMamba/SS2D module."""

    def __init__(
        self,
        dim: int,
        import_path: str,
        layout: str = "auto",
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cls = _import_object(import_path)
        self.layout = layout
        kwargs = dict(kwargs or {})
        self.block = cls(dim, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layout == "bchw":
            return self.block(x)
        if self.layout == "bhwc":
            y = self.block(x.permute(0, 2, 3, 1).contiguous())
            return y.permute(0, 3, 1, 2).contiguous()

        try:
            y = self.block(x)
            if y.shape == x.shape:
                return y
        except Exception:
            pass
        y = self.block(x.permute(0, 2, 3, 1).contiguous())
        if y.ndim != 4 or y.shape[-1] != x.shape[1]:
            raise RuntimeError(
                "External VMamba block did not return BCHW or BHWC output compatible with input."
            )
        return y.permute(0, 3, 1, 2).contiguous()


class ParallelMamba2D(nn.Module):
    """Run the fallback and real SS2D/VMamba branch in parallel."""

    def __init__(
        self,
        dim: int,
        real_branch: nn.Module,
        fallback_kwargs: dict[str, Any] | None = None,
        fusion: str = "concat",
    ) -> None:
        super().__init__()
        if fusion not in {"concat", "sum"}:
            raise ValueError(f"Unsupported parallel fusion: {fusion}")
        self.fallback = Mamba2D(dim, **(fallback_kwargs or {}))
        self.real_branch = real_branch
        self.fusion = fusion
        if fusion == "concat":
            self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)
        else:
            self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fallback = self.fallback(x)
        real = self.real_branch(x)
        if self.fusion == "concat":
            return self.fuse(torch.cat([fallback, real], dim=1))
        return fallback + self.gamma * real


def build_mamba_block(
    dim: int,
    variant: str = "fallback",
    external_import_path: str | None = None,
    external_layout: str = "auto",
    external_kwargs: dict[str, Any] | None = None,
    parallel_fusion: str = "concat",
    **kwargs: Any,
) -> nn.Module:
    variant = variant.lower()
    fallback_keys = {"expansion", "conv_kernel", "dropout", "decay_init"}
    ss2d_keys = {
        "d_state",
        "expansion",
        "dt_rank",
        "dt_min",
        "dt_max",
        "dt_init",
        "dt_scale",
        "dt_init_floor",
        "conv_kernel",
        "dropout",
        "scan_backend",
        "bias",
        "conv_bias",
    }
    fallback_kwargs = {key: kwargs[key] for key in fallback_keys if key in kwargs}

    if variant == "fallback":
        return Mamba2D(dim, **fallback_kwargs)

    if variant == "ss2d":
        ss2d_kwargs = {key: kwargs[key] for key in ss2d_keys if key in kwargs}
        return SS2D(dim, **ss2d_kwargs)

    if variant == "external":
        if not external_import_path:
            raise ValueError("model.mamba.external_import_path is required for variant=external")
        return ExternalVMambaBlock(
            dim,
            import_path=external_import_path,
            layout=external_layout,
            kwargs=external_kwargs,
        )

    if variant == "parallel":
        real_variant = kwargs.pop("parallel_real_variant", "ss2d")
        if real_variant == "external":
            if not external_import_path:
                raise ValueError("model.mamba.external_import_path is required for parallel external branch")
            real_branch = ExternalVMambaBlock(
                dim,
                import_path=external_import_path,
                layout=external_layout,
                kwargs=external_kwargs,
            )
        elif real_variant == "ss2d":
            ss2d_kwargs = {key: kwargs[key] for key in ss2d_keys if key in kwargs}
            real_branch = SS2D(dim, **ss2d_kwargs)
        else:
            raise ValueError(f"Unsupported parallel_real_variant: {real_variant}")
        return ParallelMamba2D(
            dim,
            real_branch=real_branch,
            fallback_kwargs=fallback_kwargs,
            fusion=parallel_fusion,
        )

    raise ValueError(f"Unsupported model.mamba.variant: {variant}")
