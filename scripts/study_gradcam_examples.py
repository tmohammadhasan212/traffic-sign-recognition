"""Reproduce a small, local-only set of validation-selected model Grad-CAMs."""
from pathlib import Path

from PIL import Image, ImageOps

from traffic_signs.explainability.gradcam import gradcam, overlay
from traffic_signs.inference.predictor import Predictor


def main():
    run = Path("outputs/study/finetuned_resnet18/seed_2028")
    predictor = Predictor(run / "best.pt", device="cpu")
    output = Path("outputs/study/gradcam_examples")
    output.mkdir(parents=True, exist_ok=True)
    for stem in ("00093", "00002", "00026", "03296", "03960", "02379", "04130"):
        path = Path(f"data/raw/GTSRB/Final_Test/Images/{stem}.ppm")
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = predictor.preprocess(image).unsqueeze(0)
            heatmap = gradcam(predictor.model, tensor)
            # The heatmap lives in padded model-input coordinates.
            base = ImageOps.pad(image, (224, 224), color=(0, 0, 0))
            overlay(base, heatmap).save(output / f"{stem}.png")


if __name__ == "__main__":
    main()
