# Local verification, 2026-10-23

- Downloaded the three official GTSRB archives and verified all 39,209 training and 12,630 test images against annotations. The committed compact split has 1,046 training tracks and 261 validation tracks; no track appears in both.
- Installed the project in a Python 3.12 virtual environment and ran `python -m pytest -q`: 30 passed. A deprecation warning comes from the installed FastAPI test client's `httpx` integration.
- Ran `ruff check .`, `compileall` and `git diff --check` successfully.
- Trained the scratch CNN for one epoch with `configs/smoke.yaml` on 32 balanced training rows and 32 balanced validation rows from the official training archive. Verified checkpoint, history and training plot creation.
- Downloaded torchvision's public ResNet-18 weights into ignored local data and completed the eight-row frozen-backbone smoke run. Checkpoint loading preserved the frozen parameter count (22,059 trainable of 11,198,571 total).
- Ran validation calibration and report generation, a single-image CLI prediction on an official test image, a FastAPI `/health` and `/predict` request using that image, and all 19 clean/corruption smoke conditions on the 32-row validation subset. These checks verify wiring only. Their metrics are excluded from the benchmark comparison.
- Generated a Grad-CAM PNG from the frozen ResNet checkpoint and verified that FastAPI `/explain` returned `200 image/png`. The heatmap is a wiring check from a smoke-trained classifier, not an interpretation of a validated model.

Full four-model, three-seed training and final official test evaluation remain pending. Therefore no portfolio accuracy, calibration, robustness, latency or error-analysis claims are made here.
