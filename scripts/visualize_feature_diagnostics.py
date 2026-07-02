from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from mirfd.datasets import build_dataset
from mirfd.models import build_model
from mirfd.utils import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MIRFD internal low/high feature and FFT diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=20, help="0 or negative means all samples.")
    parser.add_argument("--cell-size", type=int, default=150)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--contact-sheet-samples", type=int, default=12)
    return parser.parse_args()


def normalize_uint8(x: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    x = x.astype(np.float32)
    if vmin is None:
        vmin = float(np.percentile(x, 1.0))
    if vmax is None:
        vmax = float(np.percentile(x, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    x = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def label_panel(panel: np.ndarray, label: str) -> np.ndarray:
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(panel, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def blank_panel(size: int, label: str = "") -> np.ndarray:
    return label_panel(np.full((size, size, 3), 255, dtype=np.uint8), label)


def to_bgr_gray(x: np.ndarray, size: int, label: str) -> np.ndarray:
    gray = normalize_uint8(x)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    return label_panel(bgr, label)


def to_heatmap(
    x: np.ndarray | None,
    size: int,
    label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    if x is None:
        return blank_panel(size, f"{label} | missing")
    heat = cv2.applyColorMap(normalize_uint8(x, vmin=vmin, vmax=vmax), cv2.COLORMAP_TURBO)
    heat = cv2.resize(heat, (size, size), interpolation=cv2.INTER_LINEAR)
    return label_panel(heat, label)


def unwrap_feature(feat: Any) -> torch.Tensor | None:
    if feat is None:
        return None
    if isinstance(feat, (list, tuple)):
        return feat[0] if feat else None
    if not torch.is_tensor(feat):
        return None
    return feat


def tensor_response(feat: Any, batch_index: int) -> np.ndarray | None:
    feat = unwrap_feature(feat)
    if feat is None:
        return None
    return feat[batch_index].detach().float().abs().mean(dim=0).cpu().numpy()


def gate_response(feat: Any, batch_index: int) -> np.ndarray | None:
    feat = unwrap_feature(feat)
    if feat is None:
        return None
    return feat[batch_index].detach().float().mean(dim=0).cpu().numpy()


def fft_response(feat: Any, batch_index: int) -> np.ndarray | None:
    feat = unwrap_feature(feat)
    if feat is None:
        return None
    sample = feat[batch_index].detach().float()
    spectrum = torch.fft.fftshift(torch.fft.fft2(sample, dim=(-2, -1)), dim=(-2, -1)).abs()
    return torch.log1p(spectrum).mean(dim=0).cpu().numpy()


def feature_at(features: dict[str, Any], key: str, stage_index: int) -> torch.Tensor | None:
    values = features.get(key)
    if values is None and key in {"high_hat", "high_for_fusion"}:
        values = features.get("high")
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        values = [values]
    if stage_index >= len(values):
        return None
    return values[stage_index]


def metadata_at(features: dict[str, Any], key: str, stage_index: int, default: str = "") -> str:
    values = features.get(key)
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        return str(values[stage_index]) if stage_index < len(values) else default
    return str(values)


def denormalized_image(image: torch.Tensor, batch_index: int, data_cfg: dict[str, Any]) -> np.ndarray:
    arr = image[batch_index, 0].detach().float().cpu().numpy()
    normalize = data_cfg.get("normalize")
    if normalize is not None:
        arr = arr * float(normalize.get("std", 1.0)) + float(normalize.get("mean", 0.0))
    return arr


def build_and_load(config_path: str, checkpoint_path: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg = load_config(config_path)
    model = build_model(cfg).to(device)
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
    model.eval()
    return model, cfg


def value_range(values: list[np.ndarray | None]) -> tuple[float | None, float | None]:
    finite = [value.reshape(-1) for value in values if value is not None and value.size > 0]
    if not finite:
        return None, None
    merged = np.concatenate(finite)
    return float(np.percentile(merged, 1.0)), float(np.percentile(merged, 99.0))


def render_sample(
    image: torch.Tensor,
    mask: torch.Tensor,
    outputs: dict[str, Any],
    batch_index: int,
    data_cfg: dict[str, Any],
    cell_size: int,
    dataset_name: str,
    threshold: float,
) -> np.ndarray:
    raw = denormalized_image(image, batch_index, data_cfg)
    gt = mask[batch_index, 0].detach().float().cpu().numpy()
    pred = torch.sigmoid(outputs["logits"][batch_index, 0]).detach().float().cpu().numpy()
    pred_bin = (pred > threshold).astype(np.float32)
    features = outputs["features"]

    rows: list[np.ndarray] = []
    rows.append(
        np.concatenate(
            [
                to_bgr_gray(raw, cell_size, f"{dataset_name} | input"),
                to_bgr_gray(gt, cell_size, "GT"),
                to_heatmap(pred, cell_size, "pred prob", 0.0, 1.0),
                to_bgr_gray(pred_bin, cell_size, "pred mask"),
                blank_panel(cell_size, "structure"),
                blank_panel(cell_size, "frequency"),
                blank_panel(cell_size, "stage"),
            ],
            axis=1,
        )
    )

    if all(key in features for key in ("stage1_low", "stage1_residual", "stage1_high")):
        low = tensor_response(features["stage1_low"], batch_index)
        residual = tensor_response(features["stage1_residual"], batch_index)
        high = tensor_response(features["stage1_high"], batch_index)
        high_min, high_max = value_range([residual, high])
        rows.append(
            np.concatenate(
                [
                    to_heatmap(low, cell_size, "S1 low"),
                    to_heatmap(residual, cell_size, "S1 residual", high_min, high_max),
                    to_heatmap(high, cell_size, "S1 high/source", high_min, high_max),
                    blank_panel(cell_size, "S1 no gate"),
                    blank_panel(cell_size, "S1 no high_hat"),
                    to_heatmap(high, cell_size, "S1 high_for_fusion", high_min, high_max),
                    blank_panel(cell_size, "stage-1"),
                ],
                axis=1,
            )
        )
        fft_low = fft_response(features["stage1_low"], batch_index)
        fft_residual = fft_response(features["stage1_residual"], batch_index)
        fft_high = fft_response(features["stage1_high"], batch_index)
        fft_min, fft_max = value_range([fft_low, fft_residual, fft_high])
        rows.append(
            np.concatenate(
                [
                    to_heatmap(fft_low, cell_size, "FFT S1 low", fft_min, fft_max),
                    to_heatmap(fft_residual, cell_size, "FFT S1 residual", fft_min, fft_max),
                    to_heatmap(fft_high, cell_size, "FFT S1 high", fft_min, fft_max),
                    blank_panel(cell_size),
                    blank_panel(cell_size),
                    to_heatmap(fft_high, cell_size, "FFT S1 fusion", fft_min, fft_max),
                    blank_panel(cell_size, "FFT stage-1"),
                ],
                axis=1,
            )
        )

    for stage_index, stage_name in enumerate(("S2", "S3", "S4")):
        low0 = tensor_response(feature_at(features, "low0", stage_index), batch_index)
        low = tensor_response(feature_at(features, "low", stage_index), batch_index)
        residual = tensor_response(feature_at(features, "residual", stage_index), batch_index)
        high_raw = tensor_response(feature_at(features, "high_raw", stage_index), batch_index)
        high_hat = tensor_response(feature_at(features, "high_hat", stage_index), batch_index)
        high_for_fusion = tensor_response(feature_at(features, "high_for_fusion", stage_index), batch_index)
        gate = gate_response(feature_at(features, "gate", stage_index), batch_index)
        source = metadata_at(features, "block_fusion_high_source", stage_index, default="?")

        low_min, low_max = value_range([low0, low])
        high_min, high_max = value_range([residual, high_raw, high_hat, high_for_fusion])
        rows.append(
            np.concatenate(
                [
                    to_heatmap(low0, cell_size, f"{stage_name} low0", low_min, low_max),
                    to_heatmap(low, cell_size, f"{stage_name} low", low_min, low_max),
                    to_heatmap(residual, cell_size, f"{stage_name} residual", high_min, high_max),
                    to_heatmap(high_raw, cell_size, f"{stage_name} high_raw", high_min, high_max),
                    to_heatmap(high_hat, cell_size, f"{stage_name} high_hat", high_min, high_max),
                    to_heatmap(high_for_fusion, cell_size, f"{stage_name} fusion:{source}", high_min, high_max),
                    to_heatmap(gate, cell_size, f"{stage_name} gate", 0.0, 1.0),
                ],
                axis=1,
            )
        )

        fft_low = fft_response(feature_at(features, "low", stage_index), batch_index)
        fft_residual = fft_response(feature_at(features, "residual", stage_index), batch_index)
        fft_high_raw = fft_response(feature_at(features, "high_raw", stage_index), batch_index)
        fft_high_hat = fft_response(feature_at(features, "high_hat", stage_index), batch_index)
        fft_high_for_fusion = fft_response(feature_at(features, "high_for_fusion", stage_index), batch_index)
        fft_min, fft_max = value_range([fft_low, fft_residual, fft_high_raw, fft_high_hat, fft_high_for_fusion])
        rows.append(
            np.concatenate(
                [
                    to_heatmap(fft_low, cell_size, f"FFT {stage_name} low", fft_min, fft_max),
                    to_heatmap(fft_residual, cell_size, f"FFT {stage_name} residual", fft_min, fft_max),
                    to_heatmap(fft_high_raw, cell_size, f"FFT {stage_name} high_raw", fft_min, fft_max),
                    to_heatmap(fft_high_hat, cell_size, f"FFT {stage_name} high_hat", fft_min, fft_max),
                    to_heatmap(fft_high_for_fusion, cell_size, f"FFT {stage_name} fusion", fft_min, fft_max),
                    blank_panel(cell_size, "log magnitude"),
                    blank_panel(cell_size, f"FFT {stage_name}"),
                ],
                axis=1,
            )
        )

    return np.concatenate(rows, axis=0)


def make_contact_sheet(image_paths: list[Path], out_path: Path, thumb_width: int = 360, cols: int = 2) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = thumb_width / image.shape[1]
        thumbs.append(cv2.resize(image, (thumb_width, max(1, int(image.shape[0] * scale))), cv2.INTER_AREA))
    if not thumbs:
        return
    rows = []
    for start in range(0, len(thumbs), cols):
        chunk = thumbs[start : start + cols]
        max_h = max(item.shape[0] for item in chunk)
        padded = []
        for item in chunk:
            if item.shape[0] < max_h:
                pad = np.full((max_h - item.shape[0], item.shape[1], 3), 255, dtype=np.uint8)
                item = np.concatenate([item, pad], axis=0)
            padded.append(item)
        while len(padded) < cols:
            padded.append(np.full((max_h, thumb_width, 3), 255, dtype=np.uint8))
        rows.append(np.concatenate(padded, axis=1))
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, cfg = build_and_load(args.config, args.checkpoint, device)
    dataset = build_dataset(cfg["data"], split=args.split, augment=False)
    if args.max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else cfg["data"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    out_dir = ensure_dir(args.output_dir)
    manifest_path = out_dir / "manifest.csv"
    preview_paths: list[Path] = []
    written = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "image_path", "mask_path", "output_path"])
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                outputs = model(images, return_features=True, return_dict=True)
                for batch_index, image_path in enumerate(batch["image_path"]):
                    stem = Path(str(image_path)).stem
                    panel = render_sample(
                        images,
                        masks,
                        outputs,
                        batch_index,
                        cfg["data"],
                        args.cell_size,
                        args.dataset_name,
                        args.threshold,
                    )
                    output_path = out_dir / f"{written:04d}_{stem}.png"
                    cv2.imwrite(str(output_path), panel)
                    if len(preview_paths) < args.contact_sheet_samples:
                        preview_paths.append(output_path)
                    writer.writerow([written, image_path, batch["mask_path"][batch_index], output_path])
                    written += 1
    make_contact_sheet(preview_paths, out_dir / "contact_sheet.png")
    print(f"{args.dataset_name}: wrote {written} diagnostic visualizations to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
