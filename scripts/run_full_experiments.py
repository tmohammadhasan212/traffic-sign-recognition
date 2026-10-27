"""Run the fixed study in distinct development and final-evaluation stages."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

MODELS = {"cnn": "baseline", "frozen_resnet18": "frozen",
          "finetuned_resnet18": "finetune", "robust_resnet18": "robust"}
SEEDS = (2026, 2027, 2028)


def commands(stage: str, root: Path):
    for model, config in MODELS.items():
        for seed in SEEDS:
            directory = root / model / f"seed_{seed}"
            checkpoint = directory / "best.pt"
            if stage == "train":
                yield directory, [sys.executable, "scripts/train.py", "--config",
                                  f"configs/{config}.yaml", "--seed", str(seed),
                                  "--output-dir", str(directory)]
            elif stage == "validation":
                yield directory / "validation", [sys.executable, "scripts/evaluate.py",
                    "--checkpoint", str(checkpoint), "--split", "validation"]
            else:
                yield directory / "test", [sys.executable, "scripts/evaluate.py",
                    "--checkpoint", str(checkpoint), "--split", "test"]
                yield directory / "robustness_test", [sys.executable, "scripts/robustness.py",
                    "--checkpoint", str(checkpoint), "--split", "test", "--seed", "2026"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "validation", "test"))
    parser.add_argument("--root", type=Path, default=Path("outputs/study"))
    args = parser.parse_args()
    jobs = list(commands(args.stage, args.root))
    # Preflight all inputs before touching the official test set.
    if args.stage != "train":
        for model in MODELS:
            for seed in SEEDS:
                run = args.root / model / f"seed_{seed}"
                required = [run / "best.pt"]
                if args.stage == "test":
                    required += [run / "calibration.json", run / "validation/metrics.json"]
                for path in required:
                    if not path.is_file():
                        raise FileNotFoundError(f"Required input missing: {path}")
    for destination, command in jobs:
        if destination.exists():
            marker = (destination / "experiment.json" if args.stage == "train"
                      else destination / ("results.json" if destination.name == "robustness_test"
                                          else "metrics.json"))
            if not marker.is_file() or (args.stage == "train" and
                    "training_duration_seconds" not in json.loads(marker.read_text())):
                raise RuntimeError(f"Incomplete output at {destination}; inspect and move it before retrying")
            print(f"Skipping completed {destination}", flush=True)
            continue
        print("Running: " + " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
