from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import ConcatDataset, Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTS = IMAGE_EXTS


def _as_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Invalid resize value: {value}")


def _as_normalize(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Invalid normalize value: {value}")
    mean = float(value.get("mean", 0.0))
    std = float(value.get("std", 1.0))
    if std == 0:
        raise ValueError("normalize.std must be non-zero")
    return {"mean": mean, "std": std}


def _read_split_file(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def _candidate_dirs(root: Path, split: str) -> list[tuple[Path, Path]]:
    image_names = ["images", "image", "imgs", "img", "Image", "Images"]
    mask_names = ["masks", "mask", "labels", "label", "gt", "GT", "masks_binary"]
    pairs = []
    for image_name in image_names:
        for mask_name in mask_names:
            pairs.extend(
                [
                    (root / split / image_name, root / split / mask_name),
                    (root / image_name / split, root / mask_name / split),
                    (root / image_name, root / mask_name),
                ]
            )
    return pairs


def _find_dirs(root: Path, split: str) -> tuple[Path, Path]:
    for image_dir, mask_dir in _candidate_dirs(root, split):
        if image_dir.is_dir() and mask_dir.is_dir():
            return image_dir, mask_dir
    raise ValueError(
        f"Could not infer image/mask directories under {root}. "
        "Pass image_dir and mask_dir explicitly in the config."
    )


def _collect_files(directory: Path, exts: set[str]) -> list[Path]:
    return sorted([path for path in directory.rglob("*") if path.suffix.lower() in exts])


def _candidate_stems(name: str, suffixes: list[str] | None = None) -> list[str]:
    stem = Path(name).stem
    stems = [stem]
    for suffix in suffixes or []:
        if suffix and not stem.endswith(suffix):
            stems.append(f"{stem}{suffix}")
    return list(dict.fromkeys(stems))


def _resolve_named_file(directory: Path, name: str, exts: set[str], suffixes: list[str] | None = None) -> Path | None:
    candidate = directory / name
    if candidate.is_file():
        return candidate
    for stem in _candidate_stems(name, suffixes):
        for ext in exts:
            candidate = directory / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
    return None


def _mask_lookup(mask_dir: Path, suffixes: list[str] | None = None) -> dict[str, Path]:
    masks_by_stem = {}
    suffixes = suffixes or []
    for path in _collect_files(mask_dir, MASK_EXTS):
        masks_by_stem[path.stem] = path
        for suffix in suffixes:
            if suffix and path.stem.endswith(suffix):
                masks_by_stem[path.stem[: -len(suffix)]] = path
    return masks_by_stem


def _pad_to_size(image: np.ndarray, mask: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    target_h, target_w = size
    h, w = image.shape
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return image, mask
    pad = ((0, pad_h), (0, pad_w))
    return np.pad(image, pad, mode="constant"), np.pad(mask, pad, mode="constant")


def _pad_to_multiple(
    image: np.ndarray,
    mask: np.ndarray,
    multiple: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if multiple is None:
        return image, mask
    if multiple <= 0:
        raise ValueError("pad_to_multiple must be positive")
    h, w = image.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, mask
    pad = ((0, pad_h), (0, pad_w))
    return np.pad(image, pad, mode="constant"), np.pad(mask, pad, mode="constant")


def _random_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: tuple[int, int],
    positive_prob: float = 0.0,
    max_attempts: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    image, mask = _pad_to_size(image, mask, crop_size)
    h, w = image.shape
    crop_h, crop_w = crop_size
    require_positive = random.random() < positive_prob
    best = None
    for _ in range(max_attempts):
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        cropped_image = image[top : top + crop_h, left : left + crop_w]
        cropped_mask = mask[top : top + crop_h, left : left + crop_w]
        best = (cropped_image, cropped_mask)
        if not require_positive or cropped_mask.sum() > 0:
            return best
    return best if best is not None else (image[:crop_h, :crop_w], mask[:crop_h, :crop_w])


def _augment_arrays(
    image: np.ndarray,
    mask: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    if random.random() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)

    if mode == "sctransnet":
        if random.random() < 0.5:
            image = image.transpose(1, 0)
            mask = mask.transpose(1, 0)
    elif mode == "rotate":
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
    else:
        raise ValueError(f"Unsupported augmentation_mode: {mode}")
    return image, mask


class InfraredSmallTargetDataset(Dataset):
    """Flexible grayscale image and binary mask dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        split_file: str | Path | None = None,
        mask_suffixes: list[str] | None = None,
        resize: tuple[int, int] | int | None = None,
        crop_size: tuple[int, int] | int | None = None,
        positive_crop_prob: float = 0.0,
        normalize: dict[str, float] | None = None,
        pad_to_multiple: int | None = None,
        augment: bool = False,
        augmentation_mode: str = "rotate",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.resize = _as_size(resize)
        self.crop_size = _as_size(crop_size)
        self.positive_crop_prob = float(positive_crop_prob)
        self.normalize = _as_normalize(normalize)
        self.pad_to_multiple = pad_to_multiple
        self.augment = augment
        self.augmentation_mode = augmentation_mode
        self.mask_suffixes = mask_suffixes or []

        if image_dir is None or mask_dir is None:
            self.image_dir, self.mask_dir = _find_dirs(self.root, split)
        else:
            self.image_dir = Path(image_dir)
            self.mask_dir = Path(mask_dir)
            if not self.image_dir.is_absolute():
                self.image_dir = self.root / self.image_dir
            if not self.mask_dir.is_absolute():
                self.mask_dir = self.root / self.mask_dir

        if split_file is not None:
            split_path = Path(split_file)
            if not split_path.is_absolute():
                split_path = self.root / split_path
            names = _read_split_file(split_path)
            images = [_resolve_named_file(self.image_dir, name, IMAGE_EXTS) for name in names]
            masks = [
                _resolve_named_file(self.mask_dir, name, MASK_EXTS, self.mask_suffixes)
                for name in names
            ]
            self.samples = [(img, mask) for img, mask in zip(images, masks) if img and mask]
        else:
            masks_by_stem = _mask_lookup(self.mask_dir, self.mask_suffixes)
            self.samples = []
            for image_path in _collect_files(self.image_dir, IMAGE_EXTS):
                mask_path = masks_by_stem.get(image_path.stem)
                if mask_path is not None:
                    self.samples.append((image_path, mask_path))

        if not self.samples:
            raise ValueError(f"No image/mask pairs found for split '{split}' in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pair(self, image_path: Path, mask_path: Path) -> tuple[Image.Image, Image.Image]:
        image = Image.open(image_path).convert("F" if self.normalize is not None else "L")
        mask = Image.open(mask_path).convert("L")
        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
            mask = mask.resize((self.resize[1], self.resize[0]), Image.NEAREST)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.samples[index]
        image, mask = self._load_pair(image_path, mask_path)
        image_arr = np.asarray(image, dtype=np.float32)
        mask_arr = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)

        if self.crop_size is not None and self.augment:
            image_arr, mask_arr = _random_crop(
                image_arr,
                mask_arr,
                self.crop_size,
                positive_prob=self.positive_crop_prob,
            )

        if self.pad_to_multiple is not None and not self.augment:
            image_arr, mask_arr = _pad_to_multiple(image_arr, mask_arr, self.pad_to_multiple)

        if self.augment:
            image_arr, mask_arr = _augment_arrays(image_arr, mask_arr, self.augmentation_mode)

        if self.normalize is not None:
            image_arr = (image_arr - self.normalize["mean"]) / self.normalize["std"]
        else:
            image_arr = image_arr / 255.0

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_arr)).unsqueeze(0),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_arr)).unsqueeze(0),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def _dataset_from_cfg(data_cfg: dict[str, Any], split: str, augment: bool) -> InfraredSmallTargetDataset:
    if "root" not in data_cfg:
        raise ValueError("data.root is required")
    return InfraredSmallTargetDataset(
        root=data_cfg["root"],
        split=split,
        image_dir=data_cfg.get(f"{split}_image_dir", data_cfg.get("image_dir")),
        mask_dir=data_cfg.get(f"{split}_mask_dir", data_cfg.get("mask_dir")),
        split_file=data_cfg.get(f"{split}_split_file"),
        mask_suffixes=data_cfg.get("mask_suffixes"),
        resize=data_cfg.get(f"{split}_resize", data_cfg.get("resize")),
        crop_size=data_cfg.get(f"{split}_crop_size", data_cfg.get("crop_size")),
        positive_crop_prob=data_cfg.get("positive_crop_prob", 0.0),
        normalize=data_cfg.get("normalize"),
        pad_to_multiple=data_cfg.get(f"{split}_pad_to_multiple", data_cfg.get("pad_to_multiple")),
        augment=augment and data_cfg.get("augment", True),
        augmentation_mode=data_cfg.get("augmentation_mode", "rotate"),
    )


def build_dataset(data_cfg: dict[str, Any], split: str, augment: bool | None = None) -> Dataset:
    augment = (split == "train") if augment is None else augment
    if "datasets" not in data_cfg:
        return _dataset_from_cfg(data_cfg, split, augment)

    datasets = []
    shared_keys = {
        key: value
        for key, value in data_cfg.items()
        if key not in {"datasets", "num_workers"}
    }
    for dataset_cfg in data_cfg["datasets"]:
        merged = {**shared_keys, **dataset_cfg}
        try:
            datasets.append(_dataset_from_cfg(merged, split, augment))
        except ValueError:
            if split == "val" and merged.get("test_split_file"):
                merged[f"{split}_split_file"] = merged["test_split_file"]
                datasets.append(_dataset_from_cfg(merged, split, augment))
            else:
                raise
    if not datasets:
        raise ValueError(f"No datasets configured for split '{split}'")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
