from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mirfd.datasets import build_dataset
from mirfd.losses import bce_dice_loss
from mirfd.metrics import SegmentationMetricAccumulator
from mirfd.models import build_model
from mirfd.utils import AverageMeter, ensure_dir, load_config, set_seed


BRANCHES = (
    "low",
    "residual",
    "high_raw",
    "low_residual",
    "low_high_raw",
    "low_residual_high_raw",
)
STAGE_NAMES = (2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight branch probes on frozen MIRFD features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--branch", default=",".join(BRANCHES), help="Comma-separated branch names.")
    parser.add_argument("--stage", default="2", help="Comma-separated stages, e.g. 1,2,3.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--pd-fa-mode", default=None)
    parser.add_argument("--pd-fa-distance", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    return parser.parse_args()


class ProbeHead(nn.Module):
    def __init__(self, in_channels: int, norm: str = "batch") -> None:
        super().__init__()
        hidden = max(in_channels // 2, 4)
        if norm == "batch":
            norm_layer: nn.Module = nn.BatchNorm2d(hidden)
        elif norm == "instance":
            norm_layer = nn.InstanceNorm2d(hidden, affine=True)
        else:
            norm_layer = nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            norm_layer,
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_stage_list(value: str) -> list[int]:
    stages = [int(item) for item in _parse_csv_list(value)]
    invalid = [stage for stage in stages if stage not in {1, 2, 3, 4}]
    if invalid:
        raise ValueError(f"Unsupported probe stages: {invalid}")
    return stages


def _limit_dataset(dataset, max_samples: int):
    if max_samples and max_samples > 0:
        return torch.utils.data.Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def make_loader(cfg: dict[str, Any], split: str, batch_size: int, shuffle: bool, max_samples: int, num_workers: int):
    dataset = build_dataset(cfg["data"], split=split, augment=(split == "train"))
    dataset = _limit_dataset(dataset, max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def load_model_checkpoint(checkpoint_path: str | Path, model: torch.nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    ignorable_missing = {key for key in missing if key.endswith("gate_alpha")}
    relevant_missing = sorted(set(missing) - ignorable_missing)
    if relevant_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is not compatible. "
            f"Missing: {relevant_missing}; unexpected: {list(unexpected)}"
        )


def feature_at(features: dict[str, Any], key: str, stage: int) -> torch.Tensor:
    if stage == 1:
        stage1_key = {
            "low": "stage1_low",
            "residual": "stage1_residual",
            "high_raw": "stage1_high",
        }[key]
        values = features[stage1_key]
        return values[0] if isinstance(values, (list, tuple)) else values

    stage_index = STAGE_NAMES.index(stage)
    values = features[key]
    return values[stage_index] if isinstance(values, (list, tuple)) else values


def select_branch_feature(features: dict[str, Any], stage: int, branch: str) -> torch.Tensor:
    if branch == "low":
        return feature_at(features, "low", stage)
    if branch == "residual":
        return feature_at(features, "residual", stage)
    if branch == "high_raw":
        return feature_at(features, "high_raw", stage)
    if branch == "low_residual":
        return torch.cat([feature_at(features, "low", stage), feature_at(features, "residual", stage)], dim=1)
    if branch == "low_high_raw":
        return torch.cat([feature_at(features, "low", stage), feature_at(features, "high_raw", stage)], dim=1)
    if branch == "low_residual_high_raw":
        return torch.cat(
            [
                feature_at(features, "low", stage),
                feature_at(features, "residual", stage),
                feature_at(features, "high_raw", stage),
            ],
            dim=1,
        )
    raise ValueError(f"Unsupported branch: {branch}. Expected one of {BRANCHES}.")


def has_false_alarm(logits: torch.Tensor, masks: torch.Tensor, threshold: float, min_area: int = 3) -> tuple[int, int]:
    try:
        import cv2  # type: ignore
    except Exception:  # pragma: no cover
        cv2 = None

    pred = (torch.sigmoid(logits) >= threshold).detach().cpu().numpy().astype("uint8")
    gt = (masks > 0.5).detach().cpu().numpy().astype("uint8")
    false_alarm_images = 0
    for idx in range(pred.shape[0]):
        if cv2 is not None:
            num_labels, labels = cv2.connectedComponents(pred[idx, 0], connectivity=8)
        else:
            num_labels, labels = connected_components_numpy(pred[idx, 0])
        has_fa = 0
        for label_id in range(1, num_labels):
            comp = labels == label_id
            if int(comp.sum()) >= min_area and int((comp & (gt[idx, 0] > 0)).sum()) == 0:
                has_fa = 1
                break
        false_alarm_images += has_fa
    return false_alarm_images, pred.shape[0]


def connected_components_numpy(binary: np.ndarray) -> tuple[int, np.ndarray]:
    binary = binary.astype(bool)
    labels = np.zeros(binary.shape, dtype=np.int32)
    current_label = 0
    height, width = binary.shape
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            current_label += 1
            stack = [(y, x)]
            labels[y, x] = current_label
            while stack:
                cy, cx = stack.pop()
                for ny in range(max(cy - 1, 0), min(cy + 2, height)):
                    for nx in range(max(cx - 1, 0), min(cx + 2, width)):
                        if binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current_label
                            stack.append((ny, nx))
    return current_label + 1, labels


@torch.no_grad()
def evaluate_probe(
    model: torch.nn.Module,
    probe: torch.nn.Module,
    loader: DataLoader,
    stage: int,
    branch: str,
    device: torch.device,
    metrics_cfg: dict[str, Any],
) -> tuple[dict[str, float], float]:
    model.eval()
    probe.eval()
    threshold = float(metrics_cfg.get("threshold", 0.5))
    meter = SegmentationMetricAccumulator(
        threshold=threshold,
        pd_fa_mode=metrics_cfg.get("pd_fa_mode", "overlap"),
        pd_fa_distance=metrics_cfg.get("pd_fa_distance", 3.0),
    )
    fa_count = 0
    image_count = 0
    for batch in tqdm(loader, desc=f"eval s{stage} {branch}", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        outputs = model(images, return_features=True, return_dict=True)
        feat = select_branch_feature(outputs["features"], stage, branch).detach()
        logits = probe(feat)
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        meter.update(logits, masks)
        batch_fa, batch_n = has_false_alarm(logits, masks, threshold)
        fa_count += batch_fa
        image_count += batch_n
    metrics = meter.compute()
    false_alarm_rate = fa_count / max(image_count, 1)
    return metrics, false_alarm_rate


def train_one_probe(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    stage: int,
    branch: str,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    probe: ProbeHead | None = None
    optimizer: torch.optim.Optimizer | None = None
    best: dict[str, Any] = {"probe_iou": -1.0}

    for epoch in range(1, args.epochs + 1):
        model.eval()
        if probe is not None:
            probe.train()
        loss_meter = AverageMeter()
        for batch in tqdm(train_loader, desc=f"probe s{stage} {branch} {epoch}/{args.epochs}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.no_grad():
                outputs = model(images, return_features=True, return_dict=True)
                feat = select_branch_feature(outputs["features"], stage, branch).detach()
            if probe is None:
                probe = ProbeHead(feat.shape[1], norm=cfg.get("model", {}).get("norm", "batch")).to(device)
                optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                probe.train()
            assert optimizer is not None
            logits = probe(feat)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = bce_dice_loss(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_meter.update(float(loss.detach().cpu()), n=images.size(0))

        assert probe is not None
        metrics, fa_rate = evaluate_probe(model, probe, eval_loader, stage, branch, device, cfg.get("metrics", {}))
        if metrics.get("iou", -1.0) > best.get("probe_iou", -1.0):
            best = {
                "dataset": args.dataset_name,
                "stage": stage,
                "branch": branch,
                "best_epoch": epoch,
                "train_loss": loss_meter.avg,
                "probe_iou": metrics.get("iou", float("nan")),
                "probe_niou": metrics.get("niou", float("nan")),
                "probe_dice": metrics.get("dice", float("nan")),
                "probe_precision": metrics.get("precision", float("nan")),
                "probe_recall": metrics.get("recall", float("nan")),
                "probe_pd": metrics.get("pd", float("nan")),
                "probe_fa": metrics.get("fa", float("nan")),
                "false_alarm_rate": fa_rate,
            }
        print(
            f"stage={stage} branch={branch} epoch={epoch} loss={loss_meter.avg:.4f} "
            f"iou={metrics.get('iou', 0.0):.4f} pd={metrics.get('pd', 0.0):.4f} "
            f"fa={metrics.get('fa', 0.0):.6f} fa_rate={fa_rate:.4f}"
        )

    return best


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.setdefault("data", {})["root"] = args.data_root
    if args.threshold is not None:
        cfg.setdefault("metrics", {})["threshold"] = args.threshold
    if args.pd_fa_mode is not None:
        cfg.setdefault("metrics", {})["pd_fa_mode"] = args.pd_fa_mode
    if args.pd_fa_distance is not None:
        cfg.setdefault("metrics", {})["pd_fa_distance"] = args.pd_fa_distance

    set_seed(cfg.get("seed", 42))
    device = torch.device(args.device)
    train_cfg = cfg.get("train", {})
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 8))
    eval_batch_size = int(args.eval_batch_size or train_cfg.get("eval_batch_size", batch_size))
    num_workers = int(args.num_workers if args.num_workers is not None else cfg["data"].get("num_workers", 4))

    train_loader = make_loader(
        cfg,
        args.train_split,
        batch_size=batch_size,
        shuffle=True,
        max_samples=args.max_train_samples,
        num_workers=num_workers,
    )
    eval_loader = make_loader(
        cfg,
        args.eval_split,
        batch_size=eval_batch_size,
        shuffle=False,
        max_samples=args.max_eval_samples,
        num_workers=num_workers,
    )

    model = build_model(cfg).to(device)
    load_model_checkpoint(args.checkpoint, model, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    branches = _parse_csv_list(args.branch)
    stages = _parse_stage_list(args.stage)
    rows = []
    for stage in stages:
        for branch in branches:
            if branch not in BRANCHES:
                raise ValueError(f"Unsupported branch: {branch}. Expected one of {BRANCHES}.")
            rows.append(train_one_probe(model, train_loader, eval_loader, stage, branch, args, cfg, device))

    output_csv = Path(args.output_csv)
    ensure_dir(output_csv.parent)
    fieldnames = [
        "dataset",
        "stage",
        "branch",
        "best_epoch",
        "train_loss",
        "probe_iou",
        "probe_niou",
        "probe_dice",
        "probe_precision",
        "probe_recall",
        "probe_pd",
        "probe_fa",
        "false_alarm_rate",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote branch probe results: {output_csv}")


if __name__ == "__main__":
    main()
