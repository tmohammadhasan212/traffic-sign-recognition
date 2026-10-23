import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.app import create_app
from traffic_signs.inference.predictor import MAX_UPLOAD_BYTES, Predictor, decode_image
from traffic_signs.models.factory import create_model
from traffic_signs.training.config import Config
from traffic_signs.training.engine import save_checkpoint


def _png(mode="RGB"):
    stream = io.BytesIO()
    Image.new(mode, (20, 40), 128).save(stream, format="PNG")
    return stream.getvalue()


def _ppm():
    stream = io.BytesIO()
    Image.new("RGB", (20, 40), (150, 30, 20)).save(stream, format="PPM")
    return stream.getvalue()


def test_single_image_and_api(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(checkpoint, create_model("cnn"), Config(), 1, 0.1)
    predictor = Predictor(checkpoint, device="cpu")
    prediction = predictor.predict(decode_image(_png("L")))
    assert len(prediction["top_3_predictions"]) == 3
    assert all(0 <= item["probability"] <= 1 for item in prediction["top_3_predictions"])
    assert [item["probability"] for item in prediction["top_3_predictions"]] == sorted(
        [item["probability"] for item in prediction["top_3_predictions"]], reverse=True)
    assert prediction["uncertain"] is True  # No validation calibration provided.
    client = TestClient(create_app(str(checkpoint)))
    assert client.get("/health").json()["status"] == "ready"
    assert "Traffic sign recognition" in client.get("/").text
    response = client.post("/predict", files={"file": ("image.png", _png(), "image/png")})
    assert response.status_code == 200
    assert response.json()["predicted_class_id"] in range(43)
    preview = client.post("/preview", files={"file": ("sign.ppm", _ppm(), "image/x-portable-pixmap")})
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(preview.content)) as shown:
        assert shown.size == (20, 40)
    wide = io.BytesIO()
    Image.new("RGB", (1200, 80), (120, 120, 120)).save(wide, format="PNG")
    wide_preview = client.post("/preview", files={"file": ("wide.png", wide.getvalue(), "image/png")})
    with Image.open(io.BytesIO(wide_preview.content)) as shown:
        assert shown.width <= 512 and shown.height <= 512
    invalid = client.post("/predict", files={"file": ("bad.png", b"bad", "image/png")})
    assert invalid.status_code == 422
    assert client.post("/preview", files={"file": ("bad.png", b"bad", "image/png")}).status_code == 422


def test_rejects_oversize_and_unsupported_image():
    with pytest.raises(ValueError, match="at most 10 MB"):
        decode_image(b"x" * (MAX_UPLOAD_BYTES + 1))
    large = io.BytesIO()
    Image.new("1", (5100, 5100)).save(large, format="PNG")
    with pytest.raises(ValueError, match="dimensions"):
        decode_image(large.getvalue())
    gif = io.BytesIO()
    Image.new("RGB", (20, 20)).save(gif, format="GIF")
    with pytest.raises(ValueError, match="Supported formats"):
        decode_image(gif.getvalue())


def test_api_reports_missing_checkpoint(tmp_path: Path):
    client = TestClient(create_app(str(tmp_path / "missing.pt")))
    assert client.get("/health").json()["status"] == "error"
    response = client.post("/predict", files={"file": ("image.png", _png(), "image/png")})
    assert response.status_code == 503
