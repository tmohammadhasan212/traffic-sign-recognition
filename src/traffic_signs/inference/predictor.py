"""Single-image inference, including validation-derived abstention."""

import io
import json
import time
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from traffic_signs.data.dataset import transform
from traffic_signs.data.labels import CLASS_NAMES
from traffic_signs.explainability.gradcam import gradcam, overlay
from traffic_signs.training.engine import load_checkpoint, select_device

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000


def decode_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Image must be nonempty and at most 10 MB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG", "PPM", "WEBP", "BMP"}:
                raise ValueError("Supported formats: JPEG, PNG, PPM, WebP and BMP")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image dimensions are too large")
            image.load()
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Unreadable image") from exc


class Predictor:
    def __init__(self, checkpoint: Path, calibration: Path | None = None,
                 device: str = "auto"):
        self.device = select_device(device)
        self.model, self.config = load_checkpoint(checkpoint, self.device)
        self.temperature, self.threshold = 1.0, 1.0
        self.calibrated = False
        if calibration:
            data = json.loads(calibration.read_text())
            if Path(data["checkpoint"]).resolve() != checkpoint.resolve():
                raise ValueError("Calibration belongs to a different checkpoint")
            self.temperature = float(data["temperature"])
            self.threshold = float(data["threshold"])
            self.calibrated = True
        self.preprocess = transform(self.config.model)

    def predict(self, image: Image.Image) -> dict:
        started = time.perf_counter()
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            scores = (logits / self.temperature).softmax(1)[0].cpu()
        top = torch.topk(scores, 3)
        top_three = [{"class_id": int(i), "class_name": CLASS_NAMES[int(i)],
                      "probability": float(p)} for p, i in zip(top.values, top.indices)]
        return {"predicted_class": top_three[0]["class_name"],
                "predicted_class_id": top_three[0]["class_id"],
                "confidence": top_three[0]["probability"], "top_3_predictions": top_three,
                "uncertain": not self.calibrated or top_three[0]["probability"] < self.threshold,
                "calibrated": self.calibrated,
                "inference_time_ms": (time.perf_counter() - started) * 1000,
                "disclaimer": "Research/educational demo; never use as a driving safety system."}

    def explain(self, image: Image.Image) -> bytes:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        heatmap = gradcam(self.model, tensor)
        result = overlay(image, heatmap)
        stream = io.BytesIO()
        result.save(stream, format="PNG")
        return stream.getvalue()
