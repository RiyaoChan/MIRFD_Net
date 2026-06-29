from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


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


def _resolve_named_file(directory: Path, name: str, exts: set[str]) -> Path | None:
    candidate = directory / name
    if candidate.is_file():
        return candidate
    stem = Path(name).stem
    for ext in exts:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


class InfraredSmallTargetDataset(Dataset):
    """Flexible grayscale image and binary mask dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        split_file: str | Path | None = None,
        resize: tuple[int, int] | int | None = None,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.resize = _as_size(resize)
        self.augment = augment

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
            masks = [_resolve_named_file(self.mask_dir, name, MASK_EXTS) for name in names]
            self.samples = [(img, mask) for img, mask in zip(images, masks) if img and mask]
        else:
            masks_by_stem = {path.stem: path for path in _collect_files(self.mask_dir, MASK_EXTS)}
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
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
            mask = mask.resize((self.resize[1], self.resize[0]), Image.NEAREST)
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            k = random.randint(0, 3)
            if k:
                image = image.rotate(90 * k, expand=False)
                mask = mask.rotate(90 * k, expand=False)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.samples[index]
        image, mask = self._load_pair(image_path, mask_path)
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        return {
            "image": torch.from_numpy(image_arr).unsqueeze(0),
            "mask": torch.from_numpy(mask_arr).unsqueeze(0),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def build_dataset(data_cfg: dict[str, Any], split: str, augment: bool | None = None) -> InfraredSmallTargetDataset:
    if "root" not in data_cfg:
        raise ValueError("data.root is required")
    augment = (split == "train") if augment is None else augment
    return InfraredSmallTargetDataset(
        root=data_cfg["root"],
        split=split,
        image_dir=data_cfg.get(f"{split}_image_dir", data_cfg.get("image_dir")),
        mask_dir=data_cfg.get(f"{split}_mask_dir", data_cfg.get("mask_dir")),
        split_file=data_cfg.get(f"{split}_split_file"),
        resize=data_cfg.get("resize"),
        augment=augment and data_cfg.get("augment", True),
    )
