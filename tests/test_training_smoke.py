from dataclasses import replace
from pathlib import Path

from PIL import Image

from traffic_signs.inference.predictor import Predictor
from traffic_signs.training.config import Config
from traffic_signs.training.engine import train


def test_training_checkpoint_and_inference(tiny_gtsrb, tmp_path):
    root, manifest, _ = tiny_gtsrb
    config = replace(Config.load(Path("configs/smoke.yaml")),
                     data_root=str(root), manifest=str(manifest),
                     output_dir=str(tmp_path / "run"), device="cpu", smoke_limit=8)
    checkpoint = train(config)
    assert checkpoint.is_file()
    assert (checkpoint.parent / "history.json").is_file()
    assert (checkpoint.parent / "training_curves.png").is_file()
    predictor = Predictor(checkpoint, device="cpu")
    image_path = root / "GTSRB/Final_Test/Images/00000.ppm"
    with Image.open(image_path) as image:
        assert predictor.predict(image)["predicted_class_id"] in range(43)
