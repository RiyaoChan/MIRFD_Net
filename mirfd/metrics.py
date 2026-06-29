from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    target = target > 0.5
    pred = torch.sigmoid(logits) >= threshold

    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()
    pred_sum = pred.sum().float()
    target_sum = target.sum().float()

    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)

    batch_inter = (pred & target).flatten(1).sum(dim=1).float()
    batch_union = (pred | target).flatten(1).sum(dim=1).float()
    niou = ((batch_inter + eps) / (batch_union + eps)).mean()

    return {
        "iou": float(iou.cpu()),
        "niou": float(niou.cpu()),
        "dice": float(dice.cpu()),
        "precision": float(precision.cpu()),
        "recall": float(recall.cpu()),
    }


class MetricAverager:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count = 0

    def update(self, values: dict[str, float], n: int = 1) -> None:
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * n
        self.count += n

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.totals.items()}
