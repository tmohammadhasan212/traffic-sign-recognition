# Implementation guide

This is a chronological explanation of the repository and its design. Run commands from the repository root after `pip install -e '.[dev]'`.

## 1. Objective and architecture

The input is one **cropped** traffic-sign image. The output is a 43-class probability distribution, the best name and ID, top three scores, a validation-calibrated uncertainty decision, and latency. `src/traffic_signs/` contains the package; `scripts/` are thin entry points; `api/` serves the same predictor as the CLI. This layout keeps model logic testable without a notebook or web server.

## 2. Environment and data provenance

Use Python 3.10–3.13 and install the project with the commands in README. The official GTSRB archive is described in `DATA.md`; `scripts/prepare_data.py --download` fetches the training images, final test images and final test ground truth. The downloader records a UTC timestamp and SHA-256 for each archive. It checks archive member paths before extraction. Dataset and model weights live in ignored directories.

## 3. Parsing and integrity

`data.prepare.scan_training` visits class folders 00000–00042, reads semicolon CSV rows, checks each row's `ClassId`, filename, image existence, uniqueness and decoding, then compares the directory's PPM files with CSV references. `scan_test` reads official final-test annotations and checks the same alignment. The normal command requires the published 39,209 training and 12,630 test rows. A test-only `--allow-subset` switch supports synthetic fixtures. Failures identify the missing or mismatched file.

## 4. Tracks, leakage and split manifest

Training filenames encode a five-digit track prefix and frame suffix. Frames in a track depict one physical sign, so a frame-random split leaks near-duplicates. We define each group as `<class ID>/<track prefix>`, sort each class's groups, shuffle deterministically with NumPy `default_rng(seed)`, and allocate 20% of tracks to validation. Whole tracks move together. The saved JSON manifest lists every relative path, label, track and split, plus a path-list digest. The committed `configs/split.json` states the seed and algorithm; `configs/gtsrb_track_split.json` commits official track assignments and is checked on future preparations. The generated file inventory is ignored with dataset files. The official test section is never folded into training or validation. Augmentation is applied when the training Dataset loads an image, after split assignment.

## 5. Preprocessing, Dataset and DataLoader

`ManifestDataset` uses the saved manifest as the sole row source, returns `(tensor, class_id)`, and raises a useful error for a missing manifest or unreadable image. `ImageOps.pad` preserves aspect ratio by adding black borders. CNN input is 64 × 64 with channels in `[0,1]`; ResNet input is 224 × 224 with ImageNet mean and standard deviation. Evaluation, CLI and API call the same `transform` function. Training adds a modest rotation; robust runs add color jitter and occasional blur. Horizontal flips are excluded because sign direction matters. DataLoaders use a seeded generator, deterministic worker count configuration and no shuffle outside training.

## 6. Scratch baseline

`SmallCNN` has three Conv–BatchNorm–ReLU–MaxPool stages, adaptive pooling, and a two-layer classifier with dropout. It is deliberately compact and starts with random weights. `parameter_counts` exposes total and trainable parameter counts, saved with each experiment. Its 43 output logits are passed to cross-entropy; no softmax is applied during loss calculation.

## 7. Training loop and optimization

`Config` validates model, optimizer, device, scheduler and positive numeric settings. YAML files cover baseline, frozen, full fine-tuning, robust fine-tuning and a small smoke run. `seed_everything` seeds Python, NumPy and PyTorch, asks PyTorch for deterministic operations, and disables cuDNN benchmarking. Device selection prefers CUDA, then Apple MPS, then CPU; explicit unavailable requests fail. The loop supports AdamW or SGD, optional cosine scheduler, weight decay and patience-based stopping. Each batch clears gradients, runs forward and backward passes, and steps the optimizer. Validation is inference-only. The checkpoint with lowest validation cross-entropy is retained. The frozen ResNet keeps its backbone parameters and BatchNorm statistics fixed, while its new classifier learns. History records losses, accuracy and validation macro F1 per epoch.

## 8. Checkpoints and reproducibility

`best.pt` stores the state dictionary, full validated config, epoch and validation loss. Loading rebuilds the architecture without redownloading pretrained weights, then checks state compatibility. `experiment.json` records seed, Git revision, package versions, device, split manifest path and parameter counts. `history.json` supports later training curves. For precise reproduction, preserve the generated manifest and archive hashes alongside the ignored outputs. Determinism can still vary slightly across accelerators or PyTorch builds; record the host and versions.

## 9. Transfer learning

The ResNet-18 factory requests torchvision's default pretrained weights for new runs, replaces the last layer with 43 outputs, and applies the ImageNet normalization above. The frozen mode trains only the new last layer. Full fine-tuning updates all layers. Robust fine-tuning uses the same architecture and validation pipeline with extra training augmentation. These four variants therefore share labels, split contract and metrics.

## 10. Core evaluation

`collect` gathers held-out logits and targets. `classification_metrics` computes accuracy, macro F1 over all 43 labels, per-class precision/recall/support, a 43 × 43 confusion matrix, and expected calibration error (ECE). The reporter saves raw logits and targets as `predictions.npz`, a machine-readable `metrics.json`, confusion plot, reliability diagram and coverage curve. It records model parameters and in-memory tensor size. CPU latency is the median of 50 batch-one forward passes after ten warmups, with the host and method documented. The first evaluation should be validation; official test should be run only after decisions are fixed.

`scripts/analyze_errors.py` aligns saved logits to manifest rows, checks label order, lists frequent confusion pairs and selects high-margin false predictions for manual inspection. `scripts/aggregate_seeds.py` computes a mean and sample standard deviation from at least three saved reports.

## 11. Robustness

`robustness.corruptions.corrupt` implements blur, darkening, reduced contrast, Gaussian noise, rotation and black-square occlusion at severities 1, 2 and 3. Noise and occlusion use a seeded NumPy generator. The script operates only on a selected validation or test manifest section, deriving per-image seeds from the base seed, path, corruption and severity. It writes clean accuracy separately, each corrupted accuracy, the sample count, and a chart. Because these are generated from held-out rows, the source training set is never augmented for evaluation. Robust augmentation in training is a separate option.

## 12. Probability calibration and abstention

The validation evaluation command fits a positive scalar temperature by minimizing cross-entropy over validation logits. It then searches observed confidence thresholds and chooses the lowest threshold whose retained predictions reach a requested validation accuracy. The file records both numbers, the checkpoint path and that validation was used. Test evaluation requires this file and checks checkpoint identity; it cannot fit parameters on test data. Coverage is the fraction retained; selective accuracy is accuracy among retained predictions. If no validation threshold meets the target, the selected threshold is 1.0, which effectively abstains on all normal softmax outputs. This is a decision rule, not a safety guarantee.

## 13. Grad-CAM

The explanation module hooks ResNet-18's final convolutional block, backpropagates the selected class logit, averages feature gradients, and forms a positive weighted activation map. It upsamples and overlays that map in red. The CLI can save it; the API serves it from `POST /explain`; the page shows it when available. Inspect both plausible sign-focused and misleading border/background-focused examples in a real report. The method shows local sensitivity in one model, not a causal proof of how a human or system reasons.

## 14. Inference and FastAPI

`Predictor` loads one checkpoint, optional matching calibration, and the exact evaluation preprocessing. It returns three highest class probabilities, a predicted label, elapsed wall time and an uncertainty flag. An uncalibrated checkpoint always abstains. `decode_image` accepts JPEG, PNG, PPM, WebP or BMP, converts grayscale to RGB, caps bytes and pixels, and turns decoding failures into a clear error. The API maps these to HTTP 422, and responds with 503 when no checkpoint is configured. `POST /preview` converts supported uploads, including PPM, into a browser-readable PNG thumbnail. `GET /health`, `GET /metadata`, `POST /predict` and `POST /explain` support automation; `/` is a small upload-and-preview page. The service is a cropped-sign classifier: it cannot assert that an arbitrary photograph contains a sign.

## 15. Testing and smoke workflow

Tests construct a synthetic 43-class archive-shaped fixture with two independent tracks per class. They verify mapping, deterministic track separation, annotation alignment, dataset loading, transformations, metrics, temperature, corruption determinism, forward pass, checkpoint loading, grayscale prediction, API success and invalid input. `configs/smoke.yaml` limits rows and epochs so the complete training/evaluation/inference path can be exercised without a full benchmark run. Smoke metrics are wiring checks and must never appear in the model comparison table.

## 16. Git, reporting and final experiments

The history records package, data, training, calibration, demo, tests and documentation stages separately. `.gitignore` excludes archives, data, weights, outputs and caches. The report template at `reports/EXPERIMENT_REPORT.md` tells the operator how to record three seeds, mean and spread, corruption degradation, per-class errors, two repeated failure modes and both plausible and misleading Grad-CAM examples. No result is inferred from code or synthetic fixtures. Full GTSRB training, final test comparison and failure-case analysis require the external dataset and compute, so the current result cells remain pending until those runs happen.
