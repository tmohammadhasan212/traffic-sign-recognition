import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from api.app import create_app
from traffic_signs.inference.predictor import Predictor, decode_image
from traffic_signs.models.factory import create_model
from traffic_signs.training.config import Config
from traffic_signs.training.engine import save_checkpoint


def _png(mode="RGB"):
    stream = io.BytesIO()
    Image.new(mode, (20, 40), 128).save(stream, format="PNG")
    return stream.getvalue()


def test_single_image_and_api(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(checkpoint, create_model("cnn"), Config(), 1, 0.1)
    predictor = Predictor(checkpoint, device="cpu")
    prediction = predictor.predict(decode_image(_png("L")))
    assert len(prediction["top_3_predictions"]) == 3
    assert prediction["uncertain"] is True  # No validation calibration provided.
    client = TestClient(create_app(str(checkpoint)))
    assert client.get("/health").json()["status"] == "ready"
    response = client.post("/predict", files={"file": ("image.png", _png(), "image/png")})
    assert response.status_code == 200
    assert response.json()["predicted_class_id"] in range(43)
    invalid = client.post("/predict", files={"file": ("bad.png", b"bad", "image/png")})
    assert invalid.status_code == 422
