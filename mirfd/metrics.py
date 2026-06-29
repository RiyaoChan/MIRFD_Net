from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is available in the bundled runtime
    cv2 = None


def _prepare_binary(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    target_bin = target > 0.5
    pred_bin = torch.sigmoid(logits) >= threshold
    return pred_bin, target_bin


def _connected_components(mask: np.ndarray) -> tuple[int, np.ndarray]:
    if cv2 is not None:
        return cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)

    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or labels[y, x] != 0:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                for ny in range(max(cy - 1, 0), min(cy + 2, h)):
                    for nx in range(max(cx - 1, 0), min(cx + 2, w)):
                        if mask[ny, nx] != 0 and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return current + 1, labels


@torch.no_grad()
def segmentation_metric_counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    pred, target = _prepare_binary(logits, target, threshold)

    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()
    pred_sum = pred.sum().float()
    target_sum = target.sum().float()
    false_positive = (pred & ~target).sum().float()
    total_pixels = torch.tensor(target.numel(), device=target.device, dtype=torch.float32)

    batch_inter = (pred & target).flatten(1).sum(dim=1).float()
    batch_union = (pred | target).flatten(1).sum(dim=1).float()
    batch_iou_sum = ((batch_inter + 1e-6) / (batch_union + 1e-6)).sum()

    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    target_np = target.detach().cpu().numpy().astype(np.uint8)
    detected_targets = 0
    target_instances = 0
    for idx in range(target_np.shape[0]):
        labels_count, labels = _connected_components(target_np[idx, 0])
        target_instances += max(labels_count - 1, 0)
        for label_id in range(1, labels_count):
            if np.any(pred_np[idx, 0][labels == label_id]):
                detected_targets += 1

    return {
        "intersection": float(intersection.cpu()),
        "union": float(union.cpu()),
        "pred_sum": float(pred_sum.cpu()),
        "target_sum": float(target_sum.cpu()),
        "batch_iou_sum": float(batch_iou_sum.cpu()),
        "images": float(pred.shape[0]),
        "detected_targets": float(detected_targets),
        "target_instances": float(target_instances),
        "false_positive_pixels": float(false_positive.cpu()),
        "total_pixels": float(total_pixels.cpu()),
    }


def metrics_from_counts(counts: dict[str, float], eps: float = 1e-6) -> dict[str, float]:
    intersection = counts.get("intersection", 0.0)
    union = counts.get("union", 0.0)
    pred_sum = counts.get("pred_sum", 0.0)
    target_sum = counts.get("target_sum", 0.0)
    target_instances = counts.get("target_instances", 0.0)
    total_pixels = counts.get("total_pixels", 0.0)

    return {
        "iou": (intersection + eps) / (union + eps),
        "niou": counts.get("batch_iou_sum", 0.0) / max(counts.get("images", 0.0), eps),
        "dice": (2.0 * intersection + eps) / (pred_sum + target_sum + eps),
        "precision": (intersection + eps) / (pred_sum + eps),
        "recall": (intersection + eps) / (target_sum + eps),
        "pd": counts.get("detected_targets", 0.0) / max(target_instances, eps),
        "fa": counts.get("false_positive_pixels", 0.0) / max(total_pixels, eps),
    }


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    return metrics_from_counts(segmentation_metric_counts(logits, target, threshold), eps=eps)


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


class SegmentationMetricAccumulator:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.counts: dict[str, float] = {}

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        batch_counts = segmentation_metric_counts(logits, target, threshold=self.threshold)
        for key, value in batch_counts.items():
            self.counts[key] = self.counts.get(key, 0.0) + value

    def compute(self) -> dict[str, float]:
        if not self.counts:
            return {}
        return metrics_from_counts(self.counts)
