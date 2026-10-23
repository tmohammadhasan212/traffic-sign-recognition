"""Seeded, held-out-image corruption operators with three severity levels."""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

NAMES = ("blur", "brightness", "contrast", "noise", "rotation", "occlusion")
SEVERITIES = (1, 2, 3)


def corrupt(image: Image.Image, name: str, severity: int, seed: int) -> Image.Image:
    if name not in NAMES or severity not in SEVERITIES:
        raise ValueError("Unknown corruption or severity")
    image = image.convert("RGB")
    rng = np.random.default_rng(seed)
    if name == "blur":
        return image.filter(ImageFilter.GaussianBlur((0.7, 1.5, 2.5)[severity - 1]))
    if name == "brightness":
        return ImageEnhance.Brightness(image).enhance((0.8, 0.6, 0.4)[severity - 1])
    if name == "contrast":
        return ImageEnhance.Contrast(image).enhance((0.7, 0.45, 0.2)[severity - 1])
    if name == "rotation":
        return image.rotate((8, 16, 25)[severity - 1], resample=Image.Resampling.BILINEAR)
    array = np.asarray(image).copy()
    if name == "noise":
        noise = rng.normal(0, (10, 25, 45)[severity - 1], array.shape)
        return Image.fromarray(np.clip(array.astype(float) + noise, 0, 255).astype(np.uint8))
    width, height = image.size
    block = max(1, int(min(width, height) * (0.12, 0.22, 0.32)[severity - 1]))
    x = int(rng.integers(0, max(1, width - block + 1)))
    y = int(rng.integers(0, max(1, height - block + 1)))
    array[y:y + block, x:x + block] = 0
    return Image.fromarray(array)
