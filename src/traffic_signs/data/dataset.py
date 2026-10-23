"""Shared image preprocessing and manifest-backed PyTorch datasets."""

import json
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def open_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unreadable image: {path}") from exc


def transform(model: str, training: bool = False, robust: bool = False):
    size = 64 if model == "cnn" else 224
    operations = [T.Lambda(lambda im: ImageOps.pad(im, (size, size), color=(0, 0, 0)))]
    if training:
        operations += [T.RandomRotation(10)]
        if robust:
            operations += [T.ColorJitter(brightness=0.3, contrast=0.3),
                           T.RandomApply([T.GaussianBlur(3)], p=0.3)]
    operations += [T.ToTensor()]
    if model != "cnn":
        operations += [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(operations)


class ManifestDataset(Dataset):
    def __init__(self, root: Path, manifest: Path, split: str, model: str,
                 robust: bool = False, limit: int | None = None):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if not manifest.is_file():
            raise FileNotFoundError(f"Prepare the dataset first: {manifest}")
        content = json.loads(manifest.read_text())
        source = content["test"] if split == "test" else content["training"]
        self.rows = [r for r in source if split == "test" or r["split"] == split]
        if limit is not None:
            self.rows = self.rows[:limit]
        self.root = root
        self.transform = transform(model, split == "train", robust)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return self.transform(open_rgb(self.root / row["path"])), row["class_id"]
