"""Summarize a complete 4-model, 3-seed study without reading image data."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_full_experiments import MODELS, SEEDS

from traffic_signs.evaluation.analysis import aggregate
from traffic_signs.robustness.corruptions import NAMES, SEVERITIES


def summarize(root: Path) -> dict:
    result = {}
    expected = {(name, severity) for name in NAMES for severity in SEVERITIES} | {("clean", 0)}
    for model in MODELS:
        runs = [root / model / f"seed_{seed}" for seed in SEEDS]
        expected_config = {
            "cnn": ("cnn", False, False, False),
            "frozen_resnet18": ("resnet18", True, True, False),
            "finetuned_resnet18": ("resnet18", True, False, False),
            "robust_resnet18": ("resnet18", True, False, True),
        }[model]
        for run, seed in zip(runs, SEEDS):
            config = json.loads((run / "experiment.json").read_text())["config"]
            actual = (config["model"], config["pretrained"], config["frozen"],
                      config["robust_augmentation"])
            if config["seed"] != seed or actual != expected_config:
                raise ValueError(f"Unexpected model or seed configuration: {run}")
        summary = aggregate([run / "test/metrics.json" for run in runs])
        summary["per_seed_details"] = {}
        for run, seed in zip(runs, SEEDS):
            report = json.loads((run / "test/metrics.json").read_text())
            meta = json.loads((run / "experiment.json").read_text())
            summary["per_seed_details"][str(seed)] = {
                key: report[key] for key in ("accuracy_wilson_95", "selected_accuracy_wilson_95",
                                             "parameters", "model_size_bytes") if key in report}
            for key in ("best_epoch", "best_validation_loss", "training_duration_seconds",
                        "git_revision", "device"):
                if key in meta:
                    summary["per_seed_details"][str(seed)][key] = meta[key]
        conditions = []
        for run in runs:
            records = json.loads((run / "robustness_test/results.json").read_text())
            mapping = {(r["corruption"], r["severity"]): r for r in records}
            if (len(records) != len(expected) or set(mapping) != expected or
                    any(r["split"] != "test" or r["seed"] != 2026 for r in records)):
                raise ValueError(f"Unexpected corruption protocol: {run}")
            conditions.append(mapping)
        summary["robustness"] = {}
        for name, severity in sorted(expected):
            values = [mapping[(name, severity)]["accuracy"] for mapping in conditions]
            drops = [value - mapping[("clean", 0)]["accuracy"]
                     for value, mapping in zip(values, conditions)]
            summary["robustness"][f"{name}:{severity}"] = {
                "accuracy": {"values": dict(zip(map(str, SEEDS), values)),
                             "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))},
                "change_from_clean": {"values": dict(zip(map(str, SEEDS), drops)),
                                      "mean": float(np.mean(drops)),
                                      "std": float(np.std(drops, ddof=1))}}
        result[model] = summary
    digests = {r["split_path_sha256"] for r in result.values()}
    if len(digests) != 1:
        raise ValueError("Models use different split manifests")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/study"))
    parser.add_argument("--output", type=Path, default=Path("reports/results"))
    args = parser.parse_args()
    result = summarize(args.root)  # Validate every input before writing final artifacts.
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "final_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    metrics = ("accuracy", "macro_f1", "ece", "raw_ece", "raw_nll", "calibrated_nll",
               "temperature", "threshold", "coverage", "selective_accuracy", "latency_ms")
    with (args.output / "final_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "metric", "mean", "sample_std", *map(str, SEEDS)])
        for model, summary in result.items():
            for metric in metrics:
                if metric in summary:
                    values = summary[metric]
                    writer.writerow([model, metric, values["mean"], values["std"],
                                     *(values["values"][str(seed)] for seed in SEEDS)])


if __name__ == "__main__":
    main()
