# Experiment report template

**Status:** Full GTSRB experiments have not been run in this repository state. Do not interpret synthetic fixture or smoke results as benchmark measurements.

## Protocol

Record archive SHA-256 hashes, download date, generated split manifest digest, Git revision, library versions, hardware, and run seeds. Use three full training seeds per model where compute permits. Select epochs, model, temperature and threshold on validation only; evaluate the official test set once per final selected model. Report mean and standard deviation across seeds, plus individual values. Keep raw `predictions.npz` for regeneration.

## Model table

| Model | Seed | Best epoch | Parameters | Size | Clean accuracy | Macro F1 | ECE | CPU median ms |
|---|---|---|---|---|---|---|---|---|
| Scratch CNN | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Frozen ResNet-18 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Fine-tuned ResNet-18 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Robust fine-tuned ResNet-18 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

## Robustness and selective prediction

Insert the generated `robustness.png`, `reliability.png`, `coverage.png`, and `confusion.png` with the exact run directory. Summarize accuracy drop relative to clean for each corruption and severity. State threshold, validation target accuracy, test coverage and selective accuracy. Include per-class precision and recall from `metrics.json`.

## Error analysis

Review misclassified test indices from `predictions.npz` and the manifest. Identify difficult classes and frequent confusion pairs from the confusion matrix. Show at least two recurring failure modes, with counts and representative permitted images. Examine Grad-CAM overlays for plausible focus and for misleading background focus. The heatmap is a visualization of local gradient-weighted activations, not causal proof.

## Compute and limitations

Record total training time, hardware, number of full runs and failed runs. Discuss the crop-only task, domain shift, limits of synthetic corruptions, and confidence on out-of-distribution images. No claim belongs here until supported by actual runs.
