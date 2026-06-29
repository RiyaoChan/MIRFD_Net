from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mirfd.datasets import build_dataset
from mirfd.losses import MIRFDLoss
from mirfd.metrics import SegmentationMetricAccumulator
from mirfd.models import build_model
from mirfd.utils import AverageMeter, ensure_dir, load_config, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MIRFD-Net")
    parser.add_argument("--config", default="configs/mirfd_default.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="runs/mirfd")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def make_loader(cfg: dict, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = build_dataset(cfg["data"], split=split, augment=(split == "train"))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )


@torch.no_grad()
def evaluate(model, loader, device, need_features: bool) -> dict[str, float]:
    model.eval()
    meter = SegmentationMetricAccumulator()
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        outputs = model(images, return_features=need_features)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        meter.update(logits, masks)
    return meter.compute()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.setdefault("data", {})["root"] = args.data_root
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs

    set_seed(cfg.get("seed", 42))
    out_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device)

    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})
    batch_size = train_cfg.get("batch_size", 8)

    train_loader = make_loader(cfg, "train", batch_size, shuffle=True)
    try:
        val_loader = make_loader(cfg, "val", batch_size, shuffle=False)
    except ValueError:
        val_loader = None

    model = build_model(cfg).to(device)
    criterion = MIRFDLoss(**loss_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-2),
    )
    epochs = train_cfg.get("epochs", 300)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if train_cfg.get("scheduler", "cosine") == "cosine"
        else None
    )
    use_amp = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    need_features = loss_cfg.get("spectral_low_weight", 0.0) > 0.0 or loss_cfg.get("spectral_high_weight", 0.0) > 0.0

    best_iou = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images, return_features=need_features)
                loss, details = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(details["total"], n=images.size(0))
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if scheduler is not None:
            scheduler.step()

        metrics = evaluate(model, val_loader, device, need_features=False) if val_loader is not None else {}
        current_iou = metrics.get("iou", -1.0)
        save_checkpoint(out_dir / "last.pt", model, epoch=epoch, optimizer=optimizer.state_dict(), metrics=metrics, config=cfg)
        if current_iou > best_iou:
            best_iou = current_iou
            save_checkpoint(out_dir / "best.pt", model, epoch=epoch, optimizer=optimizer.state_dict(), metrics=metrics, config=cfg)

        metric_text = " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"epoch={epoch} train_loss={loss_meter.avg:.4f} {metric_text}")


if __name__ == "__main__":
    main()
