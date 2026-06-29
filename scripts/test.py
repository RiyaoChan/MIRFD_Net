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
from mirfd.metrics import SegmentationMetricAccumulator
from mirfd.models import build_model
from mirfd.utils import load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MIRFD-Net")
    parser.add_argument("--config", default="configs/mirfd_default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.setdefault("data", {})["root"] = args.data_root
    device = torch.device(args.device)

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    dataset = build_dataset(cfg["data"], split=args.split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or cfg.get("train", {}).get("batch_size", 8),
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )

    meter = SegmentationMetricAccumulator(threshold=args.threshold)
    with torch.no_grad():
        for batch in tqdm(loader, desc=args.split):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            outputs = model(images, return_dict=True)
            meter.update(outputs["logits"], masks)

    for key, value in meter.compute().items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
