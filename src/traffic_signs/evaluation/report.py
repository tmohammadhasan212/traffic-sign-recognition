"""Reproducible metrics and figures from saved predictions."""

import json
import platform
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from traffic_signs.calibration.temperature import coverage_curve, probabilities
from traffic_signs.evaluation.metrics import (
    classification_metrics,
    negative_log_likelihood,
    wilson_interval,
)
from traffic_signs.models.factory import parameter_counts
from traffic_signs.plotting import plt


def cpu_latency(model: torch.nn.Module, image_size: int, runs: int = 50) -> dict:
    model = model.cpu().eval()
    image = torch.zeros(1, 3, image_size, image_size)
    with torch.no_grad():
        for _ in range(10):
            model(image)
        samples = []
        for _ in range(runs):
            start = time.perf_counter()
            model(image)
            samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(samples)), "runs": runs, "batch_size": 1,
            "hardware": platform.platform(), "method": "10 warmups; perf_counter; median"}


def figures(output: Path, probs: np.ndarray, targets: np.ndarray, metrics: dict,
            curve: list[dict], raw_probs: np.ndarray | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0, 1, 11)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", label="Perfect")
    for label, values in (("Raw", raw_probs), ("Calibrated", probs)):
        if values is None:
            continue
        confidence = values.max(1)
        correct = values.argmax(1) == targets
        observed = []
        for index, (low, high) in enumerate(pairwise(edges)):
            mask = (confidence >= low) & (confidence <= high if index == len(edges) - 2
                                          else confidence < high)
            observed.append(correct[mask].mean() if mask.any() else np.nan)
        ax.plot((edges[:-1] + edges[1:]) / 2, observed, marker="o", label=label)
    ax.set(xlabel="Confidence", ylabel="Accuracy", title="Reliability")
    ax.legend()
    fig.savefig(output / "reliability.png", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(metrics["confusion_matrix"], cmap="Blues")
    ax.set(xlabel="Predicted class", ylabel="True class", title="Confusion matrix")
    fig.savefig(output / "confusion.png", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot([r["coverage"] for r in curve],
            [r["accuracy"] if r["accuracy"] is not None else np.nan for r in curve], marker="o")
    ax.set(xlabel="Coverage", ylabel="Selective accuracy", title="Coverage vs accuracy")
    fig.savefig(output / "coverage.png", bbox_inches="tight")
    plt.close(fig)


def write_report(output: Path, logits: torch.Tensor, targets: torch.Tensor,
                 temperature: float, model: torch.nn.Module, threshold: float,
                 include_latency: bool = True) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", logits=logits.numpy(), targets=targets.numpy())
    probs = probabilities(logits, temperature)
    labels = targets.numpy()
    metrics = classification_metrics(probs, labels)
    raw_probs = probabilities(logits, 1.0)
    raw_metrics = classification_metrics(raw_probs, labels)
    calibration = {"raw_ece": raw_metrics["ece"], "calibrated_ece": metrics["ece"],
                   "raw_nll": negative_log_likelihood(raw_probs, labels),
                   "calibrated_nll": negative_log_likelihood(probs, labels)}
    curve = coverage_curve(probs, labels)
    accepted = probs.max(1) >= threshold
    correct = probs.argmax(1) == labels
    result = {"metrics": metrics, "calibration": calibration,
              "accuracy_wilson_95": wilson_interval(int(correct.sum()), len(labels)),
              "selected_accuracy_wilson_95": wilson_interval(
                  int(correct[accepted].sum()), int(accepted.sum())), "temperature": temperature, "threshold": threshold,
              "selected_coverage": float(accepted.mean()),
              "selected_accuracy": float((probs.argmax(1)[accepted] == labels[accepted]).mean())
              if accepted.any() else None,
              "coverage_curve": curve, "parameters": parameter_counts(model),
              "model_size_bytes": sum(p.numel() * p.element_size() for p in model.parameters())}
    if include_latency:
        result["cpu_latency"] = cpu_latency(model, 64 if hasattr(model, "features") else 224)
    figures(output, probs, labels, metrics, curve, raw_probs)
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    return result
