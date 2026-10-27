# Robust German Traffic Sign Recognition

A reproducible PyTorch system for **43-class classification of cropped traffic signs**. It compares a compact CNN with frozen and fine-tuned ResNet-18 models, measures corruption robustness, calibrates probabilities, and abstains on low-confidence predictions. A FastAPI page and JSON API provide a lightweight demo. **Research and education only; never use as a driving safety system.**

## Research question and current evidence

Does ImageNet transfer learning and robustness augmentation improve clean and corrupted
traffic-sign classification? The controlled four-model, three-seed comparison is **not
complete**. A previous single-seed scratch CNN achieved 96.79% accuracy and 95.47%
macro F1 on the official test set; no ranking of the four models is yet supported.
The test set was already inspected for that run, so it is no longer a pristine holdout.
See the [study protocol and hypotheses](reports/EXPERIMENT_REPORT.md).

To run the fixed study after installing pretrained weights and preparing the dataset:

```bash
python scripts/run_full_experiments.py train
python scripts/run_full_experiments.py validation
# Freeze the protocol before running the following final evaluation:
python scripts/run_full_experiments.py test
```

The runner uses 2026, 2027 and 2028, stores each run separately, refuses to resume
incomplete directories, and requires all checkpoints and validation calibrations before
test evaluation. It uses a common corruption seed for paired model comparisons.

## What is here

| Component | Implementation |
|---|---|
| Data | Official GTSRB archives, verified semicolon annotations, class-balanced track-disjoint validation split |
| Models | Three-stage scratch CNN; pretrained ResNet-18 frozen head, full fine-tuning, robustness augmentation |
| Training | Seeded PyTorch loop, CPU/MPS/CUDA selection, early stopping, best checkpoint, experiment JSON |
| Evaluation | Accuracy, macro F1, 43-class precision/recall and confusion matrix, ECE, latency and model size |
| Uncertainty | Validation-only temperature and threshold fitting; coverage versus selective accuracy |
| Robustness | Blur, brightness, contrast, noise, rotation and occlusion at three severities |
| Explanations | CNN and ResNet-18 Grad-CAM overlays; interpretation is illustrative, never causal proof |
| Interfaces | Prediction CLI, FastAPI endpoints, upload and preview page |

`src/traffic_signs/` holds reusable logic; `scripts/` holds commands; `api/` holds the service; `configs/` holds runs; `tests/` holds synthetic fixtures; `reports/` holds report templates and generated outputs. See [DATA.md](DATA.md) for provenance and [the implementation guide](docs/IMPLEMENTATION_GUIDE.md) for the complete design.

## Setup

Use Python 3.10–3.13. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/prepare_data.py --download
```

The downloader creates `data/raw/` and `data/split_manifest.json`; both are excluded from Git. If you already have the official three archives, extract them into `data/raw/` and run `python scripts/prepare_data.py`. The download is large and may take time. No dataset or pretrained weights are committed.

## Train and evaluate

```bash
python scripts/train.py --config configs/smoke.yaml
python scripts/evaluate.py --checkpoint outputs/smoke/best.pt --split validation --no-latency
python scripts/train.py --config configs/smoke_frozen.yaml  # checks pretrained ResNet setup

python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/frozen.yaml
python scripts/train.py --config configs/finetune.yaml
python scripts/train.py --config configs/robust.yaml
python scripts/evaluate.py --checkpoint outputs/finetune/best.pt --split validation
python scripts/robustness.py --checkpoint outputs/finetune/best.pt --split validation
```

The smoke run uses only 32 rows per training and validation split for a wiring check. Its numbers are **not** comparable model results. For a three-seed comparison, rerun each full config with `--seed 2026`, `--seed 2027`, and `--seed 2028`, setting a distinct `--output-dir` each time. Select models and thresholds using validation only. When frozen, run one final held-out test evaluation per selected checkpoint:

```bash
python scripts/evaluate.py --checkpoint outputs/finetune/best.pt --split test
python scripts/robustness.py --checkpoint outputs/finetune/best.pt --split test
```

Validation evaluation writes `calibration.json`; the test command requires this file and checks its checkpoint identity. The default selective-accuracy target is 99%; adjust `--target-accuracy` before testing if needed. A target below the model's overall validation accuracy can produce a zero threshold and no abstention. Evaluation writes `predictions.npz`, `metrics.json`, a reliability diagram, confusion matrix and coverage chart under the run directory. Robustness writes separate JSON and chart files. The training run writes `experiment.json`, `history.json`, and `best.pt`. The experiment file records seed, configuration, Git revision, library versions, device, parameter count and split manifest path. Keep the full generated manifest with the run when sharing results.

The measured single-seed CNN run currently used by the local demo is documented in [docs/LOCAL_TRAINED_RUN.md](docs/LOCAL_TRAINED_RUN.md).

For error review, run `python scripts/analyze_errors.py --predictions outputs/finetune/test/predictions.npz --split test --output outputs/finetune/test/errors.json`. For three-seed mean and sample standard deviation, run `python scripts/aggregate_seeds.py outputs/seed2026/test/metrics.json outputs/seed2027/test/metrics.json outputs/seed2028/test/metrics.json --output outputs/finetune/aggregate.json`.

## CLI and demo

```bash
python scripts/predict.py --image path/to/cropped_sign.png \
  --checkpoint outputs/finetune/best.pt \
  --calibration outputs/finetune/calibration.json \
  --explain-output outputs/finetune/example_gradcam.png

export TRAFFIC_SIGNS_CHECKPOINT=outputs/finetune/best.pt
export TRAFFIC_SIGNS_CALIBRATION=outputs/finetune/calibration.json
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

To check the interface immediately after the smoke commands, set the two environment variables to `outputs/smoke/best.pt` and `outputs/smoke/calibration.json`. That checkpoint has seen only a tiny balanced subset; its classifications have no benchmark meaning.

Open `http://127.0.0.1:8000/` for upload, preview, prediction, top three scores and Grad-CAM on either model. The page previews GTSRB PPM files through a PNG conversion endpoint. `GET /health`, `GET /metadata`, `POST /preview`, `POST /predict`, and `POST /explain` are also available. Example:

```bash
curl -F 'file=@path/to/cropped_sign.png' http://127.0.0.1:8000/predict
```

The API returns a predicted name and ID, confidence, top three probabilities, uncertainty flag, calibration state, inference latency and disclaimer. Without validation calibration, every result is marked uncertain. The model classifies the image supplied; it does not detect whether a sign is present. A no-sign or out-of-distribution image can still receive a high score.

## Method and results

The validation split is 20% of each training class's **tracks**, seed 2026; the official test set is untouched through development. The final protocol uses the same preprocessing and metric implementation for every model. Calibration is fitted on validation logits and never refitted on test logits. Corruptions are applied only to validation or test rows and logged with severity and seed. CPU latency uses batch size one, ten warmups, 50 measured runs, and reports the median and hardware.

| Model | Seed(s) | Test accuracy | Macro F1 | ECE | Corruption summary |
|---|---|---|---|---|---|
| Scratch CNN | 2026 | 96.79% | 95.47% | 0.0088 | Not measured |
| Frozen ResNet-18 | Not run | Not measured | Not measured | Not measured | Not measured |
| Fine-tuned ResNet-18 | Not run | Not measured | Not measured | Not measured | Not measured |
| Robust fine-tuned ResNet-18 | Not run | Not measured | Not measured | Not measured | Not measured |

The single-seed full-data CNN run is detailed in [the local run report](docs/LOCAL_TRAINED_RUN.md). At its validation-selected threshold, official test coverage was 93.55% and accuracy among accepted predictions was 99.14%. Earlier data and smoke checks are in the [verification notes](reports/VERIFICATION.md). The other model runs, corruption comparison, and three-seed aggregate results remain outstanding. [The report template](reports/EXPERIMENT_REPORT.md) gives the analysis protocol, including per-class confusion, false predictions and two recurring failure modes. Fill it only from saved predictions. No demo screenshot or GTSRB image is bundled; add one only if source terms permit it.

## Limits and attribution

This is classification of **cropped signs**, not detection in road scenes. Synthetic corruption families approximate some degradations but do not measure all weather or camera behavior. Confidence thresholding cannot guarantee safety or detect all unfamiliar inputs. Grad-CAM can emphasize background or borders, so overlays require cautious reading. Training and deployment data may differ by country, sign design and camera. The code is MIT licensed; dataset rights remain with its authors. GTSRB: Stallkamp, Schlipsing, Salmen and Igel, [official public archive](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). ResNet-18 weights are provided by torchvision; review their source terms when redistributing anything derived from them.

Run `pytest` and `ruff check .` for local checks. Future work: evaluate out-of-distribution sets, stronger corruptions and interval estimates for selective accuracy.
