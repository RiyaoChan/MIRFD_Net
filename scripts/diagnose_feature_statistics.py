from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Any
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mirfd.datasets import build_dataset
from mirfd.models import build_model
from mirfd.utils import ensure_dir, load_config


RAW_FIELDNAMES = [
    "dataset",
    "sample_id",
    "stage",
    "stage_enabled",
    "stage_used_as_decoder_skip",
    "R_high_low",
    "R_high_residual",
    "R_high_high_raw",
    "R_high_high_hat",
    "R_high_high_for_fusion",
    "low_fg_bg",
    "residual_fg_bg",
    "high_raw_fg_bg",
    "gate_fg_bg",
    "high_hat_fg_bg",
    "high_for_fusion_fg_bg",
    "gate_fg_minus_bg",
    "block_fusion_high_source",
    "pred_iou",
    "pred_has_false_alarm",
]

SUMMARY_METRICS = [
    "stage_enabled",
    "stage_used_as_decoder_skip",
    "R_high_low",
    "R_high_residual",
    "R_high_high_raw",
    "R_high_high_hat",
    "R_high_high_for_fusion",
    "low_fg_bg",
    "residual_fg_bg",
    "high_raw_fg_bg",
    "gate_fg_bg",
    "high_hat_fg_bg",
    "high_for_fusion_fg_bg",
    "gate_fg_minus_bg",
    "pred_iou",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify MIRFD-Net internal feature spectrum and target selectivity "
            "for low/residual/high_raw/gate/high_hat branches."
        )
    )
    parser.add_argument("--config", required=True, help="Path to a MIRFD-Net config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--dataset-name", required=True, help="Dataset name written to the CSV.")
    parser.add_argument("--output-csv", required=True, help="Path for per-sample per-stage CSV output.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate. Default: test.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold. Default: 0.5.")
    parser.add_argument(
        "--fft-radius-ratio",
        type=float,
        default=0.25,
        help="Low-frequency circle radius ratio. Default: 0.25.",
    )
    parser.add_argument(
        "--min-fa-area",
        type=int,
        default=3,
        help="Minimum false-alarm connected-component area in pixels. Default: 3.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--data-root", default=None, help="Optional override for config data.root.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="Default: 1 for sample-level statistics.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional summary CSV path. Default: summary_<output-csv stem>.csv next to output-csv.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.10g}"
    return value


def _extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        for key in ("logits", "pred", "output"):
            if key in outputs:
                return outputs[key]
        raise KeyError("Model output dict does not contain logits/pred/output.")
    return outputs


def load_model_checkpoint(checkpoint_path: str | Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    ignorable_missing = {
        key
        for key in missing
        if key.endswith("gate_alpha")
    }
    relevant_missing = sorted(set(missing) - ignorable_missing)
    if relevant_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is not compatible with the config. "
            f"Missing: {relevant_missing}; unexpected: {list(unexpected)}"
        )
    if ignorable_missing:
        print(
            "Loaded checkpoint with default-initialized compatibility parameters: "
            f"{sorted(ignorable_missing)}"
        )
    return checkpoint if isinstance(checkpoint, dict) else {"model": state}


def _feature_at(features: dict[str, Any], key: str, stage_index: int, warned_keys: set[str]) -> torch.Tensor | None:
    values = features.get(key)
    if values is None and key in {"high_hat", "high_for_fusion"}:
        values = features.get("high")
    if values is None:
        if key not in warned_keys:
            warnings.warn(f"Feature key '{key}' is missing; related CSV fields will be nan.", stacklevel=2)
            warned_keys.add(key)
        return None
    if not isinstance(values, (list, tuple)):
        values = [values]
    if stage_index >= len(values):
        warn_key = f"{key}:{stage_index}"
        if warn_key not in warned_keys:
            warnings.warn(f"Feature key '{key}' has no stage index {stage_index}; fields will be nan.", stacklevel=2)
            warned_keys.add(warn_key)
        return None
    return values[stage_index]


def _has_stage1_features(features: dict[str, Any]) -> bool:
    return any(key in features for key in ("stage1_low", "stage1_residual", "stage1_high"))


def _stage_count(features: dict[str, Any]) -> int:
    counts = []
    for key in ("low", "residual", "high_raw", "high_hat", "high_for_fusion", "high", "gate"):
        values = features.get(key)
        if isinstance(values, (list, tuple)):
            counts.append(len(values))
        elif torch.is_tensor(values):
            counts.append(1)
    return max(counts) if counts else 0


def _sample_feature(feature: torch.Tensor | None, batch_index: int) -> torch.Tensor | None:
    if feature is None:
        return None
    if feature.ndim != 4:
        warnings.warn(f"Expected feature tensor [B,C,H,W], got shape {tuple(feature.shape)}.", stacklevel=2)
        return None
    return feature[batch_index : batch_index + 1].detach().float()


def _metadata_at(features: dict[str, Any], key: str, stage_index: int | None, default: str = "") -> str:
    values = features.get(key)
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        if stage_index is None:
            return str(values[0]) if values else default
        if stage_index < len(values):
            return str(values[stage_index])
        return default
    return str(values)


def response_map(feature: torch.Tensor, is_gate: bool = False) -> torch.Tensor:
    if is_gate:
        return feature.mean(dim=1, keepdim=True)
    return feature.abs().mean(dim=1, keepdim=True)


def fft_high_ratio(feature: torch.Tensor | None, radius_ratio: float, eps: float = 1e-8) -> float:
    if feature is None:
        return float("nan")
    try:
        response = response_map(feature, is_gate=False)
        _, _, height, width = response.shape
        fft = torch.fft.fft2(response, dim=(-2, -1))
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.abs(fft)[0, 0]

        yy, xx = torch.meshgrid(
            torch.arange(height, device=mag.device),
            torch.arange(width, device=mag.device),
            indexing="ij",
        )
        cy, cx = height // 2, width // 2
        dist = torch.sqrt((yy - cy).float().pow(2) + (xx - cx).float().pow(2))
        radius = float(radius_ratio) * min(height, width) / 2.0
        high_mask = dist > radius
        return float((mag[high_mask].sum() / (mag.sum() + eps)).detach().cpu())
    except Exception as exc:  # pragma: no cover - defensive for long diagnostics
        warnings.warn(f"Failed to compute FFT high ratio: {exc}", stacklevel=2)
        return float("nan")


def fg_bg_stats(
    feature: torch.Tensor | None,
    mask: torch.Tensor,
    is_gate: bool = False,
    eps: float = 1e-8,
) -> tuple[float, float, float, int, int]:
    if feature is None:
        return float("nan"), float("nan"), float("nan"), 0, 0
    try:
        response = response_map(feature, is_gate=is_gate)
        height, width = response.shape[-2:]
        mask_down = F.interpolate(mask.float(), size=(height, width), mode="area")
        fg = mask_down > 0
        bg = mask_down == 0
        fg_count = int(fg.sum().item())
        bg_count = int(bg.sum().item())
        if fg_count == 0 or bg_count == 0:
            return float("nan"), float("nan"), float("nan"), fg_count, bg_count
        fg_mean = response[fg].mean()
        bg_mean = response[bg].mean()
        fg_bg = (fg_mean + eps) / (bg_mean + eps)
        fg_minus_bg = fg_mean - bg_mean
        return (
            float(fg_bg.detach().cpu()),
            float(fg_minus_bg.detach().cpu()),
            float(fg_mean.detach().cpu()),
            fg_count,
            bg_count,
        )
    except Exception as exc:  # pragma: no cover - defensive for long diagnostics
        warnings.warn(f"Failed to compute fg/bg statistics: {exc}", stacklevel=2)
        return float("nan"), float("nan"), float("nan"), 0, 0


def sample_prediction_stats(
    logits: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
    min_fa_area: int,
    eps: float = 1e-8,
) -> tuple[float, int]:
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits)
    pred = prob > threshold
    gt = mask > 0.5
    intersection = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    pred_iou = float((intersection / (union + eps)).detach().cpu())
    return pred_iou, has_false_alarm(pred, gt, min_fa_area)


def connected_components(binary: np.ndarray) -> tuple[int, np.ndarray]:
    try:
        import cv2  # type: ignore

        return cv2.connectedComponents(binary.astype("uint8"))
    except Exception:
        pass

    try:
        from scipy import ndimage  # type: ignore

        labels, num_labels = ndimage.label(binary.astype("uint8"))
        return int(num_labels) + 1, labels.astype(np.int32)
    except Exception:
        pass

    return connected_components_numpy(binary)


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
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current_label
                            stack.append((ny, nx))
    return current_label + 1, labels


def has_false_alarm(pred: torch.Tensor, gt: torch.Tensor, min_fa_area: int) -> int:
    pred_np = pred[0, 0].detach().cpu().numpy().astype("uint8")
    gt_np = gt[0, 0].detach().cpu().numpy().astype("uint8")
    num_labels, labels = connected_components(pred_np)
    for label_id in range(1, num_labels):
        comp = labels == label_id
        area = int(comp.sum())
        overlap = int((comp & (gt_np > 0)).sum())
        if area >= min_fa_area and overlap == 0:
            return 1
    return 0


def sample_id_from_batch(batch: dict[str, Any], batch_index: int, global_index: int) -> str:
    image_paths = batch.get("image_path")
    if isinstance(image_paths, (list, tuple)) and batch_index < len(image_paths):
        return Path(str(image_paths[batch_index])).stem
    if isinstance(image_paths, str):
        return Path(image_paths).stem
    return f"sample_{global_index:06d}"


def stage_metadata(model: torch.nn.Module, stage: int) -> tuple[int, int]:
    high_skip_stages = set(getattr(model, "high_skip_stages", set()))
    use_high_residual_skip = bool(getattr(model, "use_high_residual_skip", True))
    if stage == 1:
        stage_enabled = int(stage in high_skip_stages)
    elif stage in {2, 3, 4}:
        stage_enabled = 1
    else:
        stage_enabled = 0
    stage_used_as_decoder_skip = int(
        use_high_residual_skip
        and stage in high_skip_stages
        and stage in {1, 2, 3}
    )
    return stage_enabled, stage_used_as_decoder_skip


def write_raw_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    ensure_dir(output_csv.parent)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(row.get(key, float("nan"))) for key in RAW_FIELDNAMES})


def write_summary_csv(rows: list[dict[str, Any]], summary_csv: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), int(row["stage"])), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset, stage), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary: dict[str, Any] = {"dataset": dataset, "stage": stage, "count": len(group)}
        for metric in SUMMARY_METRICS:
            mean_value, std_value = _mean_std([_safe_float(row.get(metric)) for row in group])
            summary[f"mean_{metric}"] = mean_value
            summary[f"std_{metric}"] = std_value
        fa_values = [_safe_float(row.get("pred_has_false_alarm")) for row in group]
        finite_fa = [value for value in fa_values if math.isfinite(value)]
        summary["false_alarm_rate"] = float(np.mean(finite_fa)) if finite_fa else float("nan")
        summary_rows.append(summary)

    fieldnames = ["dataset", "stage", "count"]
    for metric in SUMMARY_METRICS:
        fieldnames.extend([f"mean_{metric}", f"std_{metric}"])
    fieldnames.append("false_alarm_rate")

    ensure_dir(summary_csv.parent)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: _format_csv_value(row.get(key, float("nan"))) for key in fieldnames})
    return summary_rows


def diagnose(args: argparse.Namespace) -> tuple[Path, Path, list[dict[str, Any]]]:
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.setdefault("data", {})["root"] = args.data_root

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    load_model_checkpoint(args.checkpoint, model, device)
    model.eval()

    dataset = build_dataset(cfg["data"], split=args.split, augment=False)
    if args.max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.max_samples, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else cfg["data"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    rows: list[dict[str, Any]] = []
    warned_keys: set[str] = set()
    stage_names = [2, 3, 4]
    processed = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"diagnose {args.dataset_name}"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            outputs = model(images, return_features=True, return_dict=True)
            logits = _extract_logits(outputs)
            features = outputs.get("features", {}) if isinstance(outputs, dict) else {}
            if not features:
                warnings.warn("Model output does not contain features; no feature rows will be written.", stacklevel=2)
                break

            has_stage1 = _has_stage1_features(features)
            stage_count = min(_stage_count(features), len(stage_names))
            if not has_stage1 and stage_count == 0:
                warnings.warn("No stage features found; no feature rows will be written.", stacklevel=2)
                break
            stage_specs: list[tuple[int, int | None]] = []
            if has_stage1:
                stage_specs.append((1, None))
            stage_specs.extend((stage_names[index], index) for index in range(stage_count))

            batch_size = images.shape[0]
            for batch_index in range(batch_size):
                sample_index = processed + batch_index
                sample_id = sample_id_from_batch(batch, batch_index, sample_index)
                sample_mask = masks[batch_index : batch_index + 1]
                sample_logits = logits[batch_index : batch_index + 1]
                pred_iou, pred_has_fa = sample_prediction_stats(
                    sample_logits,
                    sample_mask,
                    threshold=args.threshold,
                    min_fa_area=args.min_fa_area,
                )

                for stage, stage_index in stage_specs:
                    stage_enabled, stage_used_as_decoder_skip = stage_metadata(model, stage)
                    if stage_index is None:
                        low = _sample_feature(_feature_at(features, "stage1_low", 0, warned_keys), batch_index)
                        residual = _sample_feature(_feature_at(features, "stage1_residual", 0, warned_keys), batch_index)
                        high_raw = _sample_feature(_feature_at(features, "stage1_high", 0, warned_keys), batch_index)
                        high_hat = high_raw
                        high_for_fusion = high_raw
                        gate = None
                        block_fusion_high_source = "stage1"
                    else:
                        low = _sample_feature(_feature_at(features, "low", stage_index, warned_keys), batch_index)
                        residual = _sample_feature(_feature_at(features, "residual", stage_index, warned_keys), batch_index)
                        high_raw = _sample_feature(_feature_at(features, "high_raw", stage_index, warned_keys), batch_index)
                        high_hat = _sample_feature(_feature_at(features, "high_hat", stage_index, warned_keys), batch_index)
                        high_for_fusion = _sample_feature(
                            _feature_at(features, "high_for_fusion", stage_index, warned_keys),
                            batch_index,
                        )
                        gate = _sample_feature(_feature_at(features, "gate", stage_index, warned_keys), batch_index)
                        block_fusion_high_source = _metadata_at(
                            features,
                            "block_fusion_high_source",
                            stage_index,
                            default="",
                        )

                    low_fg_bg, _, _, fg_count, _ = fg_bg_stats(low, sample_mask, is_gate=False)
                    if low is not None and fg_count == 0:
                        warnings.warn(
                            f"Sample {sample_id} stage {stage}: foreground disappeared after area resize; "
                            "fg/bg fields are nan for this stage.",
                            stacklevel=2,
                        )
                    residual_fg_bg, _, _, _, _ = fg_bg_stats(residual, sample_mask, is_gate=False)
                    high_raw_fg_bg, _, _, _, _ = fg_bg_stats(high_raw, sample_mask, is_gate=False)
                    high_hat_fg_bg, _, _, _, _ = fg_bg_stats(high_hat, sample_mask, is_gate=False)
                    high_for_fusion_fg_bg, _, _, _, _ = fg_bg_stats(high_for_fusion, sample_mask, is_gate=False)
                    if gate is None:
                        gate_fg_bg = float("nan")
                        gate_fg_minus_bg = float("nan")
                    else:
                        gate_fg_bg, gate_fg_minus_bg, _, _, _ = fg_bg_stats(gate, sample_mask, is_gate=True)

                    rows.append(
                        {
                            "dataset": args.dataset_name,
                            "sample_id": sample_id,
                            "stage": stage,
                            "stage_enabled": stage_enabled,
                            "stage_used_as_decoder_skip": stage_used_as_decoder_skip,
                            "R_high_low": fft_high_ratio(low, args.fft_radius_ratio),
                            "R_high_residual": fft_high_ratio(residual, args.fft_radius_ratio),
                            "R_high_high_raw": fft_high_ratio(high_raw, args.fft_radius_ratio),
                            "R_high_high_hat": fft_high_ratio(high_hat, args.fft_radius_ratio),
                            "R_high_high_for_fusion": fft_high_ratio(high_for_fusion, args.fft_radius_ratio),
                            "low_fg_bg": low_fg_bg,
                            "residual_fg_bg": residual_fg_bg,
                            "high_raw_fg_bg": high_raw_fg_bg,
                            "gate_fg_bg": gate_fg_bg,
                            "high_hat_fg_bg": high_hat_fg_bg,
                            "high_for_fusion_fg_bg": high_for_fusion_fg_bg,
                            "gate_fg_minus_bg": gate_fg_minus_bg,
                            "block_fusion_high_source": block_fusion_high_source,
                            "pred_iou": pred_iou,
                            "pred_has_false_alarm": pred_has_fa,
                        }
                    )
            processed += batch_size

    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_csv.with_name(f"summary_{output_csv.stem}.csv")
    write_raw_csv(rows, output_csv)
    summary_rows = write_summary_csv(rows, summary_csv)
    return output_csv, summary_csv, summary_rows


def main() -> None:
    args = parse_args()
    output_csv, summary_csv, summary_rows = diagnose(args)
    print(f"Wrote raw feature statistics: {output_csv}")
    print(f"Wrote summary statistics: {summary_csv}")
    print(
        "Note: pred_iou and pred_has_false_alarm are sample-level final-prediction metrics; "
        "they are repeated on stage rows and are not stage-specific outputs."
    )
    for row in summary_rows:
        dataset = row["dataset"]
        stage = row["stage"]
        mean_iou = _format_csv_value(row.get("mean_pred_iou", float("nan")))
        fa_rate = _format_csv_value(row.get("false_alarm_rate", float("nan")))
        residual_high = _format_csv_value(row.get("mean_R_high_residual", float("nan")))
        low_high = _format_csv_value(row.get("mean_R_high_low", float("nan")))
        gate_delta = _format_csv_value(row.get("mean_gate_fg_minus_bg", float("nan")))
        stage_enabled = _format_csv_value(row.get("mean_stage_enabled", float("nan")))
        used_skip = _format_csv_value(row.get("mean_stage_used_as_decoder_skip", float("nan")))
        print(
            f"{dataset} stage-{stage}: "
            f"stage_enabled={stage_enabled}, decoder_skip={used_skip}, "
            f"mean_iou={mean_iou}, fa_rate={fa_rate}, "
            f"R_high residual/low={residual_high}/{low_high}, "
            f"gate_fg_minus_bg={gate_delta}"
        )


if __name__ == "__main__":
    main()
