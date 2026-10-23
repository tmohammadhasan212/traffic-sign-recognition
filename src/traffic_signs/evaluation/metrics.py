"""One metric protocol for all model variants."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from traffic_signs.data.labels import NUM_CLASSES


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray,
                               bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def classification_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict:
    if len(targets) == 0 or probabilities.shape != (len(targets), NUM_CLASSES):
        raise ValueError("Expected a nonempty N x 43 probability array")
    predictions = probabilities.argmax(axis=1)
    precision, recall, _, support = precision_recall_fscore_support(
        targets, predictions, labels=np.arange(NUM_CLASSES), zero_division=0
    )
    return {"accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(f1_score(targets, predictions, labels=np.arange(NUM_CLASSES),
                                       average="macro", zero_division=0)),
            "ece": expected_calibration_error(probabilities, targets),
            "precision_per_class": precision.tolist(), "recall_per_class": recall.tolist(),
            "support_per_class": support.tolist(),
            "confusion_matrix": confusion_matrix(targets, predictions,
                                                  labels=np.arange(NUM_CLASSES)).tolist()}
