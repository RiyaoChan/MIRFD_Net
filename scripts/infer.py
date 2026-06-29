from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image
import torch

from mirfd.models import build_model
from mirfd.utils import ensure_dir, load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIRFD-Net inference")
    parser.add_argument("--config", default="configs/mirfd_default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/infer")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def iter_images(path: Path):
    if path.is_file():
        yield path
    else:
        for item in sorted(path.rglob("*")):
            if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                yield item


def preprocess(path: Path, resize) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("L")
    original_size = (image.height, image.width)
    if resize is not None:
        image = image.resize((int(resize[1]), int(resize[0])), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0), original_size


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    output_dir = ensure_dir(args.output_dir)

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    resize = cfg.get("data", {}).get("resize")

    with torch.no_grad():
        for image_path in iter_images(Path(args.input)):
            tensor, original_size = preprocess(image_path, resize)
            tensor = tensor.to(device)
            logits = model(tensor, return_dict=True)["logits"]
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            if prob.shape != original_size:
                prob = cv2.resize(prob, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
            mask = (prob >= args.threshold).astype(np.uint8) * 255
            cv2.imwrite(str(output_dir / f"{image_path.stem}_prob.png"), (prob * 255).clip(0, 255).astype(np.uint8))
            cv2.imwrite(str(output_dir / f"{image_path.stem}_mask.png"), mask)
            print(output_dir / f"{image_path.stem}_mask.png")


if __name__ == "__main__":
    main()
