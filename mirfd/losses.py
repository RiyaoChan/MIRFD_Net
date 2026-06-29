from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


def _target_like(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = target.float()
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
    return target


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = _target_like(logits, target)
    prob = torch.sigmoid(logits)
    reduce_dims = tuple(range(1, prob.ndim))
    intersection = (prob * target).sum(dim=reduce_dims)
    denominator = prob.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    target = _target_like(logits, target)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_from_logits(logits, target)
    return bce_weight * bce + dice_weight * dice


def _frequency_mask(
    h: int,
    w: int,
    radius_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype).view(h, 1)
    xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype).view(1, w)
    radius = torch.sqrt(xx * xx + yy * yy)
    return (radius <= radius_ratio).to(dtype).view(1, 1, h, w)


def _spectral_ratio(
    features: torch.Tensor,
    penalize: str,
    radius_ratio: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    spectrum = torch.fft.fftshift(torch.fft.fft2(features, norm="ortho"), dim=(-2, -1)).abs()
    low_mask = _frequency_mask(
        features.shape[-2],
        features.shape[-1],
        radius_ratio,
        features.device,
        spectrum.dtype,
    )
    mask = low_mask if penalize == "low" else (1.0 - low_mask)
    return (spectrum * mask).sum() / (spectrum.sum() + eps)


def spectral_regularization(
    low_features: Iterable[torch.Tensor] | None = None,
    high_features: Iterable[torch.Tensor] | None = None,
    radius_ratio: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    lows = [feat for feat in (low_features or []) if feat is not None]
    highs = [feat for feat in (high_features or []) if feat is not None]
    if lows:
        low_loss = torch.stack([_spectral_ratio(feat, "high", radius_ratio) for feat in lows]).mean()
    elif highs:
        low_loss = highs[0].new_tensor(0.0)
    else:
        low_loss = torch.tensor(0.0)

    if highs:
        high_loss = torch.stack([_spectral_ratio(feat, "low", radius_ratio) for feat in highs]).mean()
    else:
        high_loss = low_loss.new_tensor(0.0)
    return low_loss, high_loss


class MIRFDLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        aux_weight: float = 0.2,
        spectral_low_weight: float = 0.0,
        spectral_high_weight: float = 0.0,
        spectral_low_radius_ratio: float = 0.25,
        spectral_high_target: str = "high_raw",
    ) -> None:
        super().__init__()
        if spectral_high_target not in {"high", "high_hat", "high_raw", "residual"}:
            raise ValueError(f"Unsupported spectral_high_target: {spectral_high_target}")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.spectral_low_weight = spectral_low_weight
        self.spectral_high_weight = spectral_high_weight
        self.spectral_low_radius_ratio = spectral_low_radius_ratio
        self.spectral_high_target = spectral_high_target

    def forward(self, outputs, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        seg = bce_dice_loss(logits, target, self.bce_weight, self.dice_weight)
        loss = seg
        details = {"seg": float(seg.detach().cpu())}

        if isinstance(outputs, dict) and self.aux_weight > 0.0 and outputs.get("aux_logits"):
            aux_losses = [
                bce_dice_loss(aux, target, self.bce_weight, self.dice_weight)
                for aux in outputs["aux_logits"]
            ]
            aux = torch.stack(aux_losses).mean()
            loss = loss + self.aux_weight * aux
            details["aux"] = float(aux.detach().cpu())

        need_spectral = self.spectral_low_weight > 0.0 or self.spectral_high_weight > 0.0
        if isinstance(outputs, dict) and need_spectral and "features" in outputs:
            features = outputs["features"]
            high_features = features.get(self.spectral_high_target)
            if high_features is None and self.spectral_high_target == "high":
                high_features = features.get("high_hat")
            low_reg, high_reg = spectral_regularization(
                features.get("low"),
                high_features,
                radius_ratio=self.spectral_low_radius_ratio,
            )
            loss = loss + self.spectral_low_weight * low_reg + self.spectral_high_weight * high_reg
            details["spectral_low"] = float(low_reg.detach().cpu())
            details["spectral_high"] = float(high_reg.detach().cpu())

        details["total"] = float(loss.detach().cpu())
        return loss, details
