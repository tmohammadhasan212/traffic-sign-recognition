"""Validation-only temperature fitting and selective prediction."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.ndim != 2 or len(logits) != len(targets) or not len(targets):
        raise ValueError("Logits and targets must have compatible nonempty shapes")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)
    inputs = logits.detach().double()
    labels = targets.detach().long()

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(inputs / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20).item())


def probabilities(logits: torch.Tensor, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return (logits / temperature).softmax(dim=1).numpy()


def choose_threshold(probs: np.ndarray, targets: np.ndarray,
                     target_accuracy: float = 0.95) -> float:
    if not 0 < target_accuracy <= 1 or not len(targets):
        raise ValueError("Invalid target accuracy or empty validation data")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == targets
    candidates = np.unique(np.r_[0.0, confidence, 1.0])
    valid = [t for t in candidates if (confidence >= t).any() and
             correct[confidence >= t].mean() >= target_accuracy]
    return float(min(valid)) if valid else 1.0


def coverage_curve(probs: np.ndarray, targets: np.ndarray,
                   thresholds: list[float] | None = None) -> list[dict]:
    thresholds = thresholds or [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == targets
    result = []
    for threshold in thresholds:
        selected = confidence >= threshold
        result.append({"threshold": threshold, "coverage": float(selected.mean()),
                       "accuracy": float(correct[selected].mean()) if selected.any() else None})
    return result


def save_calibration(path: Path, temperature: float, threshold: float,
                     validation_checkpoint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"temperature": temperature, "threshold": threshold,
                                "fitted_on": "validation", "checkpoint": validation_checkpoint}, indent=2))
