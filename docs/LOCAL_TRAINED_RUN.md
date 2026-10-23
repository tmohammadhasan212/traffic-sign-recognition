# Local trained run — 2026-10-23

This run replaces the eight-image smoke checkpoint in the local demo. It is a
single-seed, full-data CNN run, not the complete multi-model, three-seed study
described in the project specification.

## Protocol

- Official GTSRB images; 31,379 training, 7,830 track-separated validation,
  and 12,630 official test images.
- `python scripts/train.py --config configs/baseline.yaml` (seed 2026, 20 epochs,
  CPU only). Best checkpoint: epoch 15 by validation cross-entropy.
- `python scripts/evaluate.py --checkpoint outputs/baseline/best.pt --split validation --target-accuracy 0.99`
  fitted temperature and abstention threshold on validation data.
- `python scripts/evaluate.py --checkpoint outputs/baseline/best.pt --split test`
  applied those fixed values once to the official test set.

| Metric | Validation | Official test |
| --- | ---: | ---: |
| Accuracy | 96.93% | 96.79% |
| Macro F1 | 95.70% | 95.47% |
| Expected calibration error | 0.0071 | 0.0088 |
| Coverage at fixed threshold | 94.04% | 93.55% |
| Accuracy among accepted predictions | 99.01% | 99.14% |

The validation-fitted temperature is 1.8875 and the confidence threshold is
0.7699. These are empirical results for cropped signs from this dataset; they
do not establish reliability on road scenes or unfamiliar sign designs.

## Local demo

```bash
TRAFFIC_SIGNS_CHECKPOINT=outputs/baseline/best.pt \
TRAFFIC_SIGNS_CALIBRATION=outputs/baseline/calibration.json \
uvicorn api.app:app --host 127.0.0.1 --port 8765
```

In browser checks, held-out `00093.ppm` was classified as Stop at 99.9%, and
held-out `00001.ppm` as Speed limit (30 km/h) at a displayed 100.0%. A blank
wide PNG abstained at 47.2%. The CNN Grad-CAM overlay rendered for the stop
sign after adding support for its final spatial feature block.

The run artifacts are under `outputs/baseline/` and are ignored by Git. The
full model comparison, three-seed spread, and robustness study remain separate
work from this demo validation.
