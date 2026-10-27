# Experiment report — study protocol and current status

## Preregistered hypotheses for the proposed comparison

These hypotheses precede the proposed new runs, but the existing single-seed CNN test result was already inspected before this protocol was written. They are not a fully preregistered, untouched-test study.

- H1: A frozen ImageNet-pretrained ResNet-18 outperforms the scratch CNN on clean classification.
- H2: Full fine-tuning outperforms a frozen backbone.
- H3: Robust augmentation improves corrupted-image accuracy versus ordinary fine-tuning.
- H4: Robust augmentation may trade clean accuracy for robustness.
- H5: Validation-fitted temperature scaling reduces ECE without changing accuracy.
- H6: Rejecting low-confidence examples increases accuracy at reduced coverage.

All six remain **unresolved for the proposed multi-seed comparison**. The earlier CNN result alone cannot establish comparative or multi-seed claims.

## Existing evidence and limitations

One previously trained CNN (seed 2026) reached 96.79% test accuracy and 95.47% macro F1, with 99.14% selective accuracy at 93.55% coverage. See [local run details](../docs/LOCAL_TRAINED_RUN.md). The other models and seeds have not been evaluated. The test set was already inspected for this CNN and demo examples, so it is no longer a pristine holdout for future decisions. Avoid using these observations to adjust the planned configurations; confirm conclusions on a new independent dataset where possible.

Reanalysis of the saved CNN test logits gives raw ECE 0.0147 and calibrated ECE
0.0088 (15 equal-width bins); raw NLL 0.1317 and calibrated NLL 0.1117.
Temperature 1.8875 was fitted on validation. Accuracy is unchanged by positive
scalar temperature scaling. There were 406 errors among 12,630 examples; the
nominal 95% Wilson interval for accuracy is 96.46%–97.08%. At the fixed
validation threshold 0.7699, selective accuracy is 99.14% (nominal 95% Wilson
interval 98.95%–99.29%) at 93.55% coverage. These binomial intervals treat
examples as independent and may be too narrow for correlated traffic-sign frames.
The existing error ranking and confusion counts are in
[existing CNN errors](results/error_analysis/existing_cnn_errors.json); the ranking
uses top-two logit margin as a proxy, not calibrated incorrect-class probability.
These are descriptive observations from an already-examined test set, not evidence
for comparisons with models that have yet to be evaluated.

## Remaining study

The comparison requires 12 full training runs, validation-only calibration, and final test and corruption evaluations. Black-padding and ROI ablations are pending. No comparative hypothesis has been established.

**Status:** The complete multi-seed GTSRB comparison has not been run. Do not interpret synthetic fixture or smoke results as benchmark measurements.

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
