"""Evaluate clean and corrupted held-out images; never used to fit parameters."""

import argparse
import json
import logging
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from traffic_signs.data.dataset import ManifestDataset, open_rgb, transform
from traffic_signs.robustness.corruptions import NAMES, SEVERITIES, corrupt
from traffic_signs.training.engine import load_checkpoint, select_device

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--split", choices=["validation", "test"], default="validation")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--output", type=Path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    device = select_device("auto")
    model, config = load_checkpoint(args.checkpoint, device)
    dataset = ManifestDataset(Path(config.data_root), Path(config.manifest), args.split,
                              config.model)
    preprocessing = transform(config.model)
    output = args.output or args.checkpoint.parent / f"robustness_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.no_grad():
        for name in ("clean",) + NAMES:
            for severity in ((0,) if name == "clean" else SEVERITIES):
                correct = 0
                for row in dataset.rows:
                    image = open_rgb(dataset.root / row["path"])
                    seed = (args.seed + zlib.crc32(f"{row['path']}:{name}:{severity}".encode())) % 2**32
                    if name != "clean":
                        image = corrupt(image, name, severity, seed)
                    tensor = preprocessing(image).unsqueeze(0).to(device)
                    correct += int(model(tensor).argmax(1).item() == row["class_id"])
                record = {"corruption": name, "severity": severity, "seed": args.seed,
                          "split": args.split, "accuracy": correct / len(dataset),
                          "count": len(dataset)}
                records.append(record)
                logging.info("%s", record)
    (output / "results.json").write_text(json.dumps(records, indent=2))
    fig, ax = plt.subplots()
    clean = records[0]["accuracy"]
    ax.axhline(clean, linestyle="--", color="black", label="Clean")
    for name in NAMES:
        values = [r["accuracy"] for r in records if r["corruption"] == name]
        ax.plot(SEVERITIES, values, marker="o", label=name)
    ax.set(xlabel="Severity", ylabel="Accuracy", title=f"Robustness on {args.split}")
    ax.legend()
    fig.savefig(output / "robustness.png", bbox_inches="tight")
    plt.close(fig)
