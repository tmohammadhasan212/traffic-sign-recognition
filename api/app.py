"""FastAPI demo. Configure TRAFFIC_SIGNS_CHECKPOINT and optionally CALIBRATION."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from traffic_signs.inference.predictor import Predictor, decode_image

DISCLAIMER = "Research/educational demo; never use as a driving safety system."
LOGGER = logging.getLogger(__name__)


def create_app(checkpoint: str | None = None, calibration: str | None = None) -> FastAPI:
    app = FastAPI(title="Traffic Sign Recognition", description=DISCLAIMER)
    checkpoint = checkpoint or os.getenv("TRAFFIC_SIGNS_CHECKPOINT")
    calibration = calibration or os.getenv("TRAFFIC_SIGNS_CALIBRATION")
    app.state.predictor = None
    app.state.load_error = None
    if checkpoint:
        try:
            app.state.predictor = Predictor(Path(checkpoint), Path(calibration) if calibration else None)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            LOGGER.error("Model startup failed: %s", exc)
            app.state.load_error = str(exc)

    @app.get("/health")
    def health():
        return {"status": "ready" if app.state.predictor else
                "error" if app.state.load_error else "unconfigured",
                "detail": app.state.load_error,
                "disclaimer": DISCLAIMER}

    @app.get("/metadata")
    def metadata():
        predictor = app.state.predictor
        return {"model": predictor.config.model if predictor else None,
                "calibrated": predictor.calibrated if predictor else False,
                "disclaimer": DISCLAIMER}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        if app.state.predictor is None:
            raise HTTPException(503, app.state.load_error or
                                "Set TRAFFIC_SIGNS_CHECKPOINT to a trained checkpoint")
        try:
            image = decode_image(await file.read(10 * 1024 * 1024 + 1))
            return app.state.predictor.predict(image)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/explain")
    async def explain(file: UploadFile = File(...)):
        if app.state.predictor is None:
            raise HTTPException(503, app.state.load_error or "Model unavailable")
        try:
            image = decode_image(await file.read(10 * 1024 * 1024 + 1))
            return Response(app.state.predictor.explain(image), media_type="image/png")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def home():
        return (Path(__file__).with_name("index.html")).read_text()

    return app


app = create_app()
