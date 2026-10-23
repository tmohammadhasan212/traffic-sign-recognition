from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from traffic_signs.calibration.temperature import choose_threshold, fit_temperature, probabilities
from traffic_signs.evaluation.metrics import classification_metrics, expected_calibration_error
from traffic_signs.explainability.gradcam import gradcam
from traffic_signs.models.factory import create_model, parameter_counts
from traffic_signs.robustness.corruptions import NAMES, corrupt
from traffic_signs.training.config import Config
from traffic_signs.training.engine import load_checkpoint, save_checkpoint


def test_model_forward_and_checkpoint(tmp_path: Path):
    model = create_model("cnn")
    model.eval()
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 43)
    assert parameter_counts(model)["trainable"] > 0
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, Config(), 1, 0.2)
    restored, config = load_checkpoint(path, torch.device("cpu"))
    assert config.model == "cnn"
    assert torch.allclose(model(torch.zeros(1, 3, 64, 64)),
                          restored(torch.zeros(1, 3, 64, 64)), atol=1e-5)


def test_resnet_forward_without_weight_download():
    model = create_model("resnet18", pretrained=False, frozen=True).eval()
    assert parameter_counts(model)["trainable"] == 512 * 43 + 43
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 43)


def test_resnet_gradcam_heatmap():
    model = create_model("resnet18", pretrained=False, frozen=True).eval()
    heatmap = gradcam(model, torch.randn(1, 3, 224, 224))
    assert heatmap.shape == (224, 224)
    assert np.isfinite(heatmap).all()
    assert 0 <= heatmap.min() <= heatmap.max() <= 1


def test_metrics_and_calibration():
    labels = np.array([0, 1, 1])
    probs = np.zeros((3, 43))
    probs[0, 0], probs[1, 1], probs[2, 1] = 1, 1, 1
    assert classification_metrics(probs, labels)["accuracy"] == 1
    assert expected_calibration_error(probs, labels) == 0
    logits = torch.tensor([[5., 0.], [0., 5.], [1., 2.]])
    targets = torch.tensor([0, 1, 1])
    temperature = fit_temperature(logits, targets)
    assert temperature > 0
    calibrated = probabilities(logits, temperature)
    assert 0 <= choose_threshold(calibrated, targets.numpy()) <= 1


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("severity", [1, 2, 3])
def test_corruption_shape_and_seed(name, severity):
    image = Image.new("RGB", (32, 24), (120, 150, 180))
    first = np.asarray(corrupt(image, name, severity, 42))
    second = np.asarray(corrupt(image, name, severity, 42))
    assert first.shape == (24, 32, 3)
    assert np.array_equal(first, second)
