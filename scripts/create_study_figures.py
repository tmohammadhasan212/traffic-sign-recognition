"""Create lightweight figures and representative errors from completed study results."""
import json
from pathlib import Path

import numpy as np

from traffic_signs.evaluation.analysis import error_cases
from traffic_signs.plotting import plt

ROOT = Path("outputs/study")
OUT = Path("reports/results")
MODELS = {"cnn": "Scratch CNN", "frozen_resnet18": "Frozen ResNet-18",
          "finetuned_resnet18": "Fine-tuned ResNet-18", "robust_resnet18": "Robust ResNet-18"}
SEEDS = (2026, 2027, 2028)


def main():
    summary = json.loads((OUT / "final_summary.json").read_text())
    figures = OUT / "figures"
    errors = OUT / "error_analysis"
    figures.mkdir(parents=True, exist_ok=True)
    errors.mkdir(parents=True, exist_ok=True)
    for metric, ylabel in (("accuracy", "Test accuracy (%)"), ("macro_f1", "Macro F1 (%)"),
                           ("ece", "Calibrated ECE")):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        factor = 1 if metric == "ece" else 100
        values = [summary[m][metric]["mean"] * factor for m in MODELS]
        spread = [summary[m][metric]["std"] * factor for m in MODELS]
        ax.bar(list(MODELS.values()), values, yerr=spread, capsize=4)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(figures / f"{metric}_comparison.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    names = ("blur", "brightness", "contrast", "noise", "rotation", "occlusion")
    x = np.arange(len(names))
    for offset, model in enumerate(MODELS):
        vals = [np.mean([summary[model]["robustness"][f"{n}:{s}"]["accuracy"]["mean"]
                         for s in (1, 2, 3)]) * 100 for n in names]
        ax.bar(x + (offset - 1.5) * 0.2, vals, width=0.2, label=MODELS[model])
    ax.set(xticks=x, xticklabels=names, ylabel="Accuracy (%)", ylim=(0, 105),
           title="Corruption accuracy: mean over 3 severities and 3 seeds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "robustness_comparison.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, label in MODELS.items():
        curves = [json.loads((ROOT / model / f"seed_{seed}/test/metrics.json").read_text())
                  ["coverage_curve"] for seed in SEEDS]
        # Curves are evaluated at the same fixed confidence thresholds in every run.
        usable = [i for i in range(len(curves[0]))
                  if all(c[i]["accuracy"] is not None for c in curves)]
        ax.plot([np.mean([c[i]["coverage"] for c in curves]) for i in usable],
                [np.mean([c[i]["accuracy"] for c in curves]) for i in usable],
                marker="o", label=label)
    ax.set(xlabel="Coverage", ylabel="Selective accuracy", title="Threshold sweep (test, descriptive)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "coverage_accuracy.png", dpi=150)
    plt.close(fig)
    for model in MODELS:
        seeds = [(json.loads((ROOT / model / f"seed_{seed}/experiment.json").read_text())
                  ["best_validation_loss"], seed) for seed in SEEDS]
        _, seed = min(seeds)
        run = ROOT / model / f"seed_{seed}"
        result = error_cases(run / "test/predictions.npz", Path("data/split_manifest.json"), "test")
        result["selected_seed"] = seed
        result["selection_rule"] = "lowest best validation loss within model"
        report = json.loads((run / "test/metrics.json").read_text())
        for metric in ("recall_per_class", "precision_per_class"):
            result[f"lowest_{metric}"] = sorted(enumerate(report["metrics"][metric]),
                                                 key=lambda p: p[1])[:10]
        (errors / f"{model}_errors.json").write_text(json.dumps(result, indent=2) + "\n")
    # Representative model selected by lowest mean validation loss across models.
    validation_means = {m: np.mean([json.loads((ROOT / m / f"seed_{s}/experiment.json")
                        .read_text())["best_validation_loss"] for s in SEEDS]) for m in MODELS}
    model = min(validation_means, key=validation_means.get)
    seed = min(SEEDS, key=lambda s: json.loads((ROOT / model / f"seed_{s}/experiment.json")
                                               .read_text())["best_validation_loss"])
    run = ROOT / model / f"seed_{seed}"
    metrics = json.loads((run / "test/metrics.json").read_text())["metrics"]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(metrics["confusion_matrix"], cmap="Blues", interpolation="nearest")
    ax.set(xlabel="Predicted class", ylabel="True class", title=f"{MODELS[model]}, seed {seed}")
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix_best_model.png", dpi=150)
    plt.close(fig)
    import shutil
    shutil.copyfile(run / "training_curves.png", figures / "training_curves_best_model.png")
    shutil.copyfile(run / "test/reliability.png", figures / "reliability_best_model.png")
    (OUT / "representative.json").write_text(json.dumps({"model": model, "seed": seed,
        "rule": "lowest mean validation loss across models, then lowest validation loss seed",
        "model_validation_losses": validation_means}, indent=2) + "\n")


if __name__ == "__main__":
    main()
