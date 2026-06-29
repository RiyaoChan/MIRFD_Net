from __future__ import annotations

import argparse
import math
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
    parser.add_argument("--resume", default=None, help="Resume from a checkpoint saved by this script.")
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


def build_optimizer(model: torch.nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    name = train_cfg.get("optimizer", "AdamW").lower()
    lr = train_cfg.get("lr", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-2)
    betas = tuple(train_cfg.get("betas", (0.9, 0.999)))
    eps = train_cfg.get("eps", 1e-8)
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=train_cfg.get("amsgrad", False),
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    if name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {train_cfg.get('optimizer')}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict,
    epochs: int,
    steps_per_epoch: int,
):
    name = str(train_cfg.get("scheduler", "cosine")).lower()
    if name in {"none", "null", "off"}:
        return None, "epoch"

    step_unit = str(train_cfg.get("scheduler_step", "epoch")).lower()
    if step_unit not in {"epoch", "iter"}:
        raise ValueError(f"Unsupported scheduler_step: {step_unit}")

    eta_min = float(train_cfg.get("eta_min", train_cfg.get("min_lr", 0.0)))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    if name == "cosine" and warmup_epochs <= 0 and step_unit == "epoch":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min), step_unit

    if name not in {"cosine", "cosine_warmup"}:
        raise ValueError(f"Unsupported scheduler: {train_cfg.get('scheduler')}")

    total_steps = max(epochs * steps_per_epoch if step_unit == "iter" else epochs, 1)
    warmup_steps = max(warmup_epochs * steps_per_epoch if step_unit == "iter" else warmup_epochs, 0)
    base_lr = float(train_cfg.get("lr", optimizer.param_groups[0]["lr"]))
    min_factor = eta_min / max(base_lr, 1e-12)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        denom = max(total_steps - warmup_steps, 1)
        progress = min(max(float(step - warmup_steps) / float(denom), 0.0), 1.0)
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), step_unit


def load_resume_state(
    resume_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> int:
    checkpoint = torch.load(resume_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    if isinstance(checkpoint, dict) and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if (
        isinstance(checkpoint, dict)
        and scheduler is not None
        and checkpoint.get("scheduler") is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", 0)) + 1 if isinstance(checkpoint, dict) else 1


def checkpoint_metric(path: Path, metric: str, default: float) -> float:
    if not path.exists():
        return default
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return default
    metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    if metric == "pd_fa":
        return float(metrics.get("pd", 0.0)) - float(metrics.get("fa", 0.0))
    return float(metrics.get(metric, default))


@torch.no_grad()
def evaluate(model, loader, device, need_features: bool, metrics_cfg: dict) -> dict[str, float]:
    model.eval()
    meter = SegmentationMetricAccumulator(
        threshold=metrics_cfg.get("threshold", 0.5),
        pd_fa_mode=metrics_cfg.get("pd_fa_mode", "overlap"),
        pd_fa_distance=metrics_cfg.get("pd_fa_distance", 3.0),
    )
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
    metrics_cfg = cfg.get("metrics", {})
    batch_size = train_cfg.get("batch_size", 8)
    eval_batch_size = train_cfg.get("eval_batch_size", batch_size)

    train_loader = make_loader(cfg, "train", batch_size, shuffle=True)
    try:
        val_loader = make_loader(cfg, "val", eval_batch_size, shuffle=False)
    except ValueError:
        val_loader = None

    model = build_model(cfg).to(device)
    criterion = MIRFDLoss(**loss_cfg)
    optimizer = build_optimizer(model, train_cfg)
    epochs = train_cfg.get("epochs", 300)
    scheduler, scheduler_step = build_scheduler(optimizer, train_cfg, epochs, len(train_loader))
    use_amp = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    need_features = loss_cfg.get("spectral_low_weight", 0.0) > 0.0 or loss_cfg.get("spectral_high_weight", 0.0) > 0.0
    clip_grad_norm = train_cfg.get("clip_grad_norm", train_cfg.get("grad_clip_norm"))

    best_iou = -1.0
    best_niou = -1.0
    best_dice = -1.0
    best_pd_fa = -float("inf")
    start_epoch = 1
    if args.resume is not None:
        start_epoch = load_resume_state(args.resume, model, optimizer, scheduler, device)
        best_iou = checkpoint_metric(out_dir / "best_iou.pt", "iou", best_iou)
        best_niou = checkpoint_metric(out_dir / "best_niou.pt", "niou", best_niou)
        best_dice = checkpoint_metric(out_dir / "best_dice.pt", "dice", best_dice)
        best_pd_fa = checkpoint_metric(out_dir / "best_pd_fa.pt", "pd_fa", best_pd_fa)
        print(f"resumed from {args.resume} at epoch={start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for iter_idx, batch in enumerate(progress, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images, return_features=need_features)
                loss, details = criterion(outputs, masks)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite loss detected at epoch={epoch}, iter={iter_idx}, loss={loss.item()}"
                )
            scaler.scale(loss).backward()
            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scheduler_step == "iter":
                scheduler.step()

            loss_meter.update(details["total"], n=images.size(0))
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if scheduler is not None and scheduler_step == "epoch":
            scheduler.step()

        metrics = evaluate(model, val_loader, device, need_features=False, metrics_cfg=metrics_cfg) if val_loader is not None else {}
        current_iou = metrics.get("iou", -1.0)
        current_niou = metrics.get("niou", -1.0)
        current_dice = metrics.get("dice", -1.0)
        current_pd_fa = metrics.get("pd", 0.0) - metrics.get("fa", 0.0)
        save_checkpoint(
            out_dir / "last.pt",
            model,
            epoch=epoch,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else None,
            metrics=metrics,
            config=cfg,
        )
        save_checkpoint(
            out_dir / "last_finite.pt",
            model,
            epoch=epoch,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else None,
            metrics=metrics,
            config=cfg,
        )
        if current_iou > best_iou:
            best_iou = current_iou
            save_checkpoint(
                out_dir / "best.pt",
                model,
                epoch=epoch,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict() if scheduler is not None else None,
                metrics=metrics,
                config=cfg,
            )
            save_checkpoint(
                out_dir / "best_iou.pt",
                model,
                epoch=epoch,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict() if scheduler is not None else None,
                metrics=metrics,
                config=cfg,
            )
        if current_niou > best_niou:
            best_niou = current_niou
            save_checkpoint(
                out_dir / "best_niou.pt",
                model,
                epoch=epoch,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict() if scheduler is not None else None,
                metrics=metrics,
                config=cfg,
            )
        if current_dice > best_dice:
            best_dice = current_dice
            save_checkpoint(
                out_dir / "best_dice.pt",
                model,
                epoch=epoch,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict() if scheduler is not None else None,
                metrics=metrics,
                config=cfg,
            )
        if current_pd_fa > best_pd_fa:
            best_pd_fa = current_pd_fa
            save_checkpoint(
                out_dir / "best_pd_fa.pt",
                model,
                epoch=epoch,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict() if scheduler is not None else None,
                metrics=metrics,
                config=cfg,
            )

        metric_text = " ".join(
            [f"{k}={v:.6f}" if k == "fa" else f"{k}={v:.4f}" for k, v in metrics.items()]
        )
        print(f"epoch={epoch} train_loss={loss_meter.avg:.4f} lr={optimizer.param_groups[0]['lr']:.8f} {metric_text}")


if __name__ == "__main__":
    main()
