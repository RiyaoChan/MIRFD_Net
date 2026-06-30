from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from pathlib import Path
import sys

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


PRESETS = {
    "nuaa": {
        "name": "NUAA-SIRST",
        "v1_config": "configs/mirfd_nuaa_sirst_ss2d_sctrans_adamw_lr1e3.yaml",
        "v1_checkpoint": "runs/nuaa_sirst_ss2d_sctrans_adamw_lr1e3/best.pt",
        "v2_config": "configs/mirfd_nuaa_sirst_ss2d_v2.yaml",
        "v2_checkpoint": "runs/v2_ablation/nuaa_v2_spectral/best.pt",
    },
    "nudt": {
        "name": "NUDT-SIRST",
        "v1_config": "configs/mirfd_nudt_sirst_ss2d_sctrans_adamw_bs32_lr1e3.yaml",
        "v1_checkpoint": "runs/nudt_sirst_ss2d_sctrans_adamw_bs32_lr1e3/best.pt",
        "v2_config": "configs/mirfd_nudt_sirst_ss2d_v2_no_spectral.yaml",
        "v2_checkpoint": "runs/v2_ablation/nudt_v2_no_spectral/best.pt",
    },
    "irstd": {
        "name": "IRSTD-1K",
        "v1_config": "configs/mirfd_irstd_1k_ss2d_sctrans_adamw_bs32_lr1e3.yaml",
        "v1_checkpoint": "runs/irstd_1k_ss2d_sctrans_adamw_bs32_lr1e3/best.pt",
        "v2_config": "configs/mirfd_irstd_1k_ss2d_v2_no_spectral.yaml",
        "v2_checkpoint": "runs/v2_ablation/irstd_v2_no_spectral/best.pt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize V1/V2 MIRFD low and gated-high feature heatmaps.")
    parser.add_argument("--datasets", default="all", help="Comma list: nuaa,nudt,irstd or all.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="outputs/v1_v2_feature_heatmaps")
    parser.add_argument(
        "--mode",
        choices=("compare", "v2-diagnostic", "both"),
        default="compare",
        help=(
            "compare: V1/V2 low and high_hat heatmaps; "
            "v2-diagnostic: V2 low/residual/high_raw/gate/high_hat; "
            "both: write both outputs."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=-1, help="-1 means all samples.")
    parser.add_argument("--cell-size", type=int, default=160)
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
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 22), (0, 0, 0), thickness=-1)
    cv2.putText(panel, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def to_bgr_gray(x: np.ndarray, size: int, label: str) -> np.ndarray:
    gray = normalize_uint8(x)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    return label_panel(bgr, label)


def to_heatmap(x: np.ndarray, size: int, label: str, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    heat = cv2.applyColorMap(normalize_uint8(x, vmin=vmin, vmax=vmax), cv2.COLORMAP_TURBO)
    heat = cv2.resize(heat, (size, size), interpolation=cv2.INTER_LINEAR)
    return label_panel(heat, label)


def tensor_response(feat: torch.Tensor, batch_index: int) -> np.ndarray:
    resp = feat[batch_index].detach().float().abs().mean(dim=0).cpu().numpy()
    return resp


def gate_response(feat: torch.Tensor, batch_index: int) -> np.ndarray:
    resp = feat[batch_index].detach().float().mean(dim=0).cpu().numpy()
    return resp


def feature_at(features: dict, key: str, stage_index: int) -> torch.Tensor:
    if key in features:
        return features[key][stage_index]
    if key == "high_hat":
        return features["high"][stage_index]
    raise KeyError(f"Feature key '{key}' is not available.")


def denormalized_image(image: torch.Tensor, batch_index: int, data_cfg: dict) -> np.ndarray:
    arr = image[batch_index, 0].detach().float().cpu().numpy()
    normalize = data_cfg.get("normalize")
    if normalize is not None:
        arr = arr * float(normalize.get("std", 1.0)) + float(normalize.get("mean", 0.0))
    return arr


def build_and_load(config_path: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    cfg = load_config(config_path)
    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    relevant_missing = [key for key in missing if not key.endswith("gate_alpha")]
    if relevant_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is not compatible. "
            f"Missing: {relevant_missing}; unexpected: {unexpected}"
        )
    model.eval()
    return model


def render_sample(
    image: torch.Tensor,
    mask: torch.Tensor,
    outputs_v1: dict,
    outputs_v2: dict,
    batch_index: int,
    data_cfg: dict,
    cell_size: int,
    title: str,
) -> np.ndarray:
    raw = denormalized_image(image, batch_index, data_cfg)
    gt = mask[batch_index, 0].detach().float().cpu().numpy()
    pred_v1 = torch.sigmoid(outputs_v1["logits"][batch_index, 0]).detach().float().cpu().numpy()
    pred_v2 = torch.sigmoid(outputs_v2["logits"][batch_index, 0]).detach().float().cpu().numpy()

    rows: list[np.ndarray] = []
    rows.append(
        np.concatenate(
            [
                to_bgr_gray(raw, cell_size, f"{title} | input"),
                to_bgr_gray(gt, cell_size, "GT"),
                to_heatmap(pred_v1, cell_size, "V1 pred"),
                to_heatmap(pred_v2, cell_size, "V2 pred"),
            ],
            axis=1,
        )
    )

    feats_v1 = outputs_v1["features"]
    feats_v2 = outputs_v2["features"]
    for stage_index, stage_name in enumerate(("S2", "S3", "S4")):
        v1_low = tensor_response(feats_v1["low"][stage_index], batch_index)
        v1_high = tensor_response(feature_at(feats_v1, "high_hat", stage_index), batch_index)
        v2_low = tensor_response(feats_v2["low"][stage_index], batch_index)
        v2_high = tensor_response(feature_at(feats_v2, "high_hat", stage_index), batch_index)

        low_values = np.concatenate([v1_low.reshape(-1), v2_low.reshape(-1)])
        high_values = np.concatenate([v1_high.reshape(-1), v2_high.reshape(-1)])
        low_min, low_max = float(np.percentile(low_values, 1.0)), float(np.percentile(low_values, 99.0))
        high_min, high_max = float(np.percentile(high_values, 1.0)), float(np.percentile(high_values, 99.0))
        rows.append(
            np.concatenate(
                [
                    to_heatmap(v1_low, cell_size, f"V1 {stage_name} low", low_min, low_max),
                    to_heatmap(v1_high, cell_size, f"V1 {stage_name} high_hat", high_min, high_max),
                    to_heatmap(v2_low, cell_size, f"V2 {stage_name} low", low_min, low_max),
                    to_heatmap(v2_high, cell_size, f"V2 {stage_name} high_hat", high_min, high_max),
                ],
                axis=1,
            )
        )
    return np.concatenate(rows, axis=0)


def render_v2_diagnostic_sample(
    image: torch.Tensor,
    mask: torch.Tensor,
    outputs_v2: dict,
    batch_index: int,
    data_cfg: dict,
    cell_size: int,
    title: str,
) -> np.ndarray:
    raw = denormalized_image(image, batch_index, data_cfg)
    gt = mask[batch_index, 0].detach().float().cpu().numpy()
    pred_v2 = torch.sigmoid(outputs_v2["logits"][batch_index, 0]).detach().float().cpu().numpy()

    blank = np.full((cell_size, cell_size, 3), 255, dtype=np.uint8)
    rows: list[np.ndarray] = []
    rows.append(
        np.concatenate(
            [
                to_bgr_gray(raw, cell_size, f"{title} | input"),
                to_bgr_gray(gt, cell_size, "GT"),
                to_heatmap(pred_v2, cell_size, "V2 pred"),
                label_panel(blank, "V2 comps"),
                label_panel(blank, "abs/gate mean"),
            ],
            axis=1,
        )
    )

    feats_v2 = outputs_v2["features"]
    for stage_index, stage_name in enumerate(("S2", "S3", "S4")):
        low = tensor_response(feature_at(feats_v2, "low", stage_index), batch_index)
        residual = tensor_response(feature_at(feats_v2, "residual", stage_index), batch_index)
        high_raw = tensor_response(feature_at(feats_v2, "high_raw", stage_index), batch_index)
        gate = gate_response(feature_at(feats_v2, "gate", stage_index), batch_index)
        high_hat = tensor_response(feature_at(feats_v2, "high_hat", stage_index), batch_index)

        high_values = np.concatenate([residual.reshape(-1), high_raw.reshape(-1), high_hat.reshape(-1)])
        high_min, high_max = float(np.percentile(high_values, 1.0)), float(np.percentile(high_values, 99.0))
        rows.append(
            np.concatenate(
                [
                    to_heatmap(low, cell_size, f"V2 {stage_name} low"),
                    to_heatmap(residual, cell_size, f"V2 {stage_name} residual", high_min, high_max),
                    to_heatmap(high_raw, cell_size, f"V2 {stage_name} high_raw", high_min, high_max),
                    to_heatmap(gate, cell_size, f"V2 {stage_name} gate"),
                    to_heatmap(high_hat, cell_size, f"V2 {stage_name} high_hat", high_min, high_max),
                ],
                axis=1,
            )
        )
    return np.concatenate(rows, axis=0)


def make_contact_sheet(image_paths: list[Path], out_path: Path, thumb_width: int = 320, cols: int = 3) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        scale = thumb_width / img.shape[1]
        thumb = cv2.resize(img, (thumb_width, max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
    if not thumbs:
        return
    rows = []
    for start in range(0, len(thumbs), cols):
        chunk = thumbs[start : start + cols]
        max_h = max(t.shape[0] for t in chunk)
        padded = []
        for thumb in chunk:
            if thumb.shape[0] < max_h:
                pad = np.full((max_h - thumb.shape[0], thumb.shape[1], 3), 255, dtype=np.uint8)
                thumb = np.concatenate([thumb, pad], axis=0)
            padded.append(thumb)
        while len(padded) < cols:
            padded.append(np.full((max_h, thumb_width, 3), 255, dtype=np.uint8))
        rows.append(np.concatenate(padded, axis=1))
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0))


def visualize_dataset(key: str, args: argparse.Namespace) -> None:
    preset = PRESETS[key]
    device = torch.device(args.device)
    cfg = load_config(preset["v1_config"])
    data_cfg = cfg["data"]
    dataset = build_dataset(data_cfg, split=args.split, augment=False)
    if args.max_samples > 0:
        indices = list(range(min(args.max_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model_v1 = build_and_load(preset["v1_config"], preset["v1_checkpoint"], device)
    model_v2 = build_and_load(preset["v2_config"], preset["v2_checkpoint"], device)

    out_root = Path(args.output_dir)
    compare_dir = ensure_dir(out_root / key) if args.mode in {"compare", "both"} else None
    diagnostic_dir = ensure_dir(out_root / f"{key}_v2_diagnostic") if args.mode in {"v2-diagnostic", "both"} else None
    compare_manifest = compare_dir / "manifest.csv" if compare_dir is not None else None
    diagnostic_manifest = diagnostic_dir / "manifest.csv" if diagnostic_dir is not None else None
    compare_preview_paths: list[Path] = []
    diagnostic_preview_paths: list[Path] = []
    written = 0
    with ExitStack() as stack:
        compare_writer = None
        diagnostic_writer = None
        if compare_manifest is not None:
            compare_handle = stack.enter_context(compare_manifest.open("w", newline="", encoding="utf-8"))
            compare_writer = csv.writer(compare_handle)
        if diagnostic_manifest is not None:
            diagnostic_handle = stack.enter_context(diagnostic_manifest.open("w", newline="", encoding="utf-8"))
            diagnostic_writer = csv.writer(diagnostic_handle)
        if compare_manifest is not None:
            assert compare_writer is not None
            compare_writer.writerow(["index", "image_path", "mask_path", "output_path"])
        if diagnostic_manifest is not None:
            assert diagnostic_writer is not None
            diagnostic_writer.writerow(["index", "image_path", "mask_path", "output_path"])
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                outputs_v1 = model_v1(images, return_features=True, return_dict=True)
                outputs_v2 = model_v2(images, return_features=True, return_dict=True)
                image_paths = batch["image_path"]
                mask_paths = batch["mask_path"]
                for batch_index, image_path in enumerate(image_paths):
                    stem = Path(image_path).stem
                    if compare_dir is not None:
                        assert compare_writer is not None
                        panel = render_sample(
                            images,
                            masks,
                            outputs_v1,
                            outputs_v2,
                            batch_index,
                            data_cfg,
                            args.cell_size,
                            preset["name"],
                        )
                        out_path = compare_dir / f"{written:04d}_{stem}.png"
                        cv2.imwrite(str(out_path), panel)
                        if len(compare_preview_paths) < args.contact_sheet_samples:
                            compare_preview_paths.append(out_path)
                        compare_writer.writerow([written, image_path, mask_paths[batch_index], out_path])
                    if diagnostic_dir is not None:
                        assert diagnostic_writer is not None
                        diagnostic_panel = render_v2_diagnostic_sample(
                            images,
                            masks,
                            outputs_v2,
                            batch_index,
                            data_cfg,
                            args.cell_size,
                            preset["name"],
                        )
                        diagnostic_out_path = diagnostic_dir / f"{written:04d}_{stem}.png"
                        cv2.imwrite(str(diagnostic_out_path), diagnostic_panel)
                        if len(diagnostic_preview_paths) < args.contact_sheet_samples:
                            diagnostic_preview_paths.append(diagnostic_out_path)
                        diagnostic_writer.writerow([written, image_path, mask_paths[batch_index], diagnostic_out_path])
                    written += 1
    if compare_dir is not None:
        make_contact_sheet(compare_preview_paths, compare_dir / "contact_sheet.png")
        print(f"{preset['name']}: wrote {written} comparison visualizations to {compare_dir}")
    if diagnostic_dir is not None:
        make_contact_sheet(diagnostic_preview_paths, diagnostic_dir / "contact_sheet.png")
        print(f"{preset['name']}: wrote {written} V2 diagnostic visualizations to {diagnostic_dir}")


def main() -> None:
    args = parse_args()
    if args.datasets == "all":
        dataset_keys = list(PRESETS)
    else:
        dataset_keys = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    for key in dataset_keys:
        if key not in PRESETS:
            raise ValueError(f"Unsupported dataset key: {key}. Options: {', '.join(PRESETS)}")
        visualize_dataset(key, args)


if __name__ == "__main__":
    main()
