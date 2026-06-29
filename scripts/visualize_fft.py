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
    parser = argparse.ArgumentParser(description="Visualize MIRFD branch spectra")
    parser.add_argument("--config", default="configs/mirfd_default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="outputs/fft")
    parser.add_argument("--stage", type=int, default=-1, help="0, 1, 2 for MIRFD stages, or -1 for last")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def normalize_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-6)
    return (x * 255).astype(np.uint8)


def spectrum_image(feat: torch.Tensor) -> np.ndarray:
    feat = feat.mean(dim=1, keepdim=True)
    spec = torch.fft.fftshift(torch.fft.fft2(feat, norm="ortho"), dim=(-2, -1)).abs()
    spec = torch.log1p(spec)[0, 0].detach().cpu().numpy()
    return cv2.applyColorMap(normalize_uint8(spec), cv2.COLORMAP_INFERNO)


def response_image(feat: torch.Tensor) -> np.ndarray:
    resp = feat.detach().float().abs().mean(dim=1)[0].cpu().numpy()
    return cv2.applyColorMap(normalize_uint8(resp), cv2.COLORMAP_VIRIDIS)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    output_dir = ensure_dir(args.output_dir)

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    data_cfg = cfg.get("data", {})
    image = Image.open(args.image).convert("F" if data_cfg.get("normalize") is not None else "L")
    resize = data_cfg.get("test_resize", data_cfg.get("val_resize", data_cfg.get("resize")))
    if resize is not None:
        image = image.resize((int(resize[1]), int(resize[0])), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    raw_arr = arr.copy()
    if data_cfg.get("normalize") is not None:
        mean = float(data_cfg["normalize"].get("mean", 0.0))
        std = float(data_cfg["normalize"].get("std", 1.0))
        arr = (arr - mean) / std
    else:
        arr = arr / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor, return_features=True, return_dict=True)
    features = outputs["features"]
    stage = args.stage
    low = features["low"][stage]
    residual = features["residual"][stage]
    high_raw = features.get("high_raw", features["high"])[stage]
    high_hat = features.get("high_hat", features["high"])[stage]
    gate = features["gate"][stage].mean(dim=1)[0].detach().cpu().numpy()

    input_img = cv2.cvtColor(normalize_uint8(raw_arr), cv2.COLOR_GRAY2BGR)
    size = (input_img.shape[1], input_img.shape[0])
    panels = [
        input_img,
        cv2.resize(response_image(low), size),
        cv2.resize(response_image(residual), size),
        cv2.resize(response_image(high_raw), size),
        cv2.resize(response_image(high_hat), size),
        cv2.resize(cv2.applyColorMap(normalize_uint8(gate), cv2.COLORMAP_VIRIDIS), size),
        cv2.resize(spectrum_image(low), size),
        cv2.resize(spectrum_image(residual), size),
        cv2.resize(spectrum_image(high_raw), size),
    ]

    panel = np.concatenate(panels, axis=1)
    out_path = output_dir / f"{Path(args.image).stem}_fft_stage{stage}.png"
    cv2.imwrite(str(out_path), panel)
    print(out_path)


if __name__ == "__main__":
    main()
