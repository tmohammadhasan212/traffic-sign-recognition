"""Derive inspectable errors and seed aggregates from saved artifacts."""

import json
from pathlib import Path

import numpy as np

from traffic_signs.data.labels import CLASS_NAMES


def error_cases(predictions: Path, manifest: Path, split: str, limit: int = 30) -> dict:
    saved = np.load(predictions)
    logits, targets = saved["logits"], saved["targets"]
    content = json.loads(manifest.read_text())
    rows = (content["test"] if split == "test" else
            [r for r in content["training"] if r["split"] == split])
    if len(rows) != len(targets):
        raise ValueError("Prediction count does not match manifest split")
    prediction = logits.argmax(1)
    if any(row["class_id"] != target for row, target in zip(rows, targets)):
        raise ValueError("Prediction targets do not align with manifest rows")
    margins = np.sort(logits, axis=1)[:, -1] - np.sort(logits, axis=1)[:, -2]
    wrong = np.flatnonzero(prediction != targets)
    ranked = wrong[np.argsort(-margins[wrong])][:limit]
    cases = [{"path": rows[int(i)]["path"], "true_id": int(targets[i]),
              "true_name": CLASS_NAMES[int(targets[i])], "predicted_id": int(prediction[i]),
              "predicted_name": CLASS_NAMES[int(prediction[i])], "logit_margin": float(margins[i])}
             for i in ranked]
    pairs = {}
    for i in wrong:
        key = f"{int(targets[i])}->{int(prediction[i])}"
        pairs[key] = pairs.get(key, 0) + 1
    return {"incorrect": len(wrong), "total": len(targets), "most_confident_errors": cases,
            "confusion_pairs": sorted(pairs.items(), key=lambda item: -item[1])[:20]}


def aggregate(paths: list[Path]) -> dict:
    """Aggregate independent seeds of the same configuration and split manifest."""
    if len(paths) < 3:
        raise ValueError("Supply at least three independent seed reports")
    reports, identities, seeds = [], [], []
    for path in paths:
        metadata = json.loads((path.parent.parent / "experiment.json").read_text())
        config = metadata["config"].copy()
        seeds.append(config.pop("seed"))
        config.pop("output_dir")
        identities.append((config, metadata["split_path_sha256"]))
        reports.append(json.loads(path.read_text()))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds")
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Incompatible configurations or split manifests")
    result = {"seeds": seeds, "split_path_sha256": identities[0][1]}
    fields = {"accuracy": ("metrics", "accuracy"), "macro_f1": ("metrics", "macro_f1"),
              "ece": ("metrics", "ece"), "raw_ece": ("calibration", "raw_ece"),
              "raw_nll": ("calibration", "raw_nll"),
              "calibrated_nll": ("calibration", "calibrated_nll"),
              "temperature": ("temperature",), "threshold": ("threshold",),
              "coverage": ("selected_coverage",), "selective_accuracy": ("selected_accuracy",),
              "latency_ms": ("cpu_latency", "median_ms")}
    for metric, keys in fields.items():
        values = []
        for report in reports:
            value = report
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            values.append(value)
        if any(value is None for value in values):
            continue
        array = np.array(values, dtype=float)
        result[metric] = {"mean": float(array.mean()), "std": float(array.std(ddof=1)),
                          "values": dict(zip(map(str, seeds), array.tolist()))}
    return result
