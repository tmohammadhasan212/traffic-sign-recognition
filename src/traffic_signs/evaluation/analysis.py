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
    if len(paths) < 3:
        raise ValueError("Supply at least three independent seed reports")
    reports = [json.loads(path.read_text()) for path in paths]
    result = {}
    for metric in ("accuracy", "macro_f1", "ece"):
        values = np.array([r["metrics"][metric] for r in reports])
        result[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
                          "values": values.tolist()}
    return result
