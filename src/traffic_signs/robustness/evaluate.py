"""Batched held-out corruption evaluation."""

import json
import zlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from traffic_signs.data.dataset import ManifestDataset, open_rgb, transform
from traffic_signs.robustness.corruptions import NAMES, SEVERITIES, corrupt
from traffic_signs.training.config import Config


class CorruptedDataset(Dataset):
    def __init__(self, source: ManifestDataset, model: str, name: str, severity: int, seed: int):
        self.source = source
        self.transform = transform(model)
        self.name, self.severity, self.seed = name, severity, seed

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        row = self.source.rows[index]
        image = open_rgb(self.source.root / row["path"])
        if self.name != "clean":
            image_seed = (self.seed + zlib.crc32(
                f"{row['path']}:{self.name}:{self.severity}".encode())) % 2**32
            image = corrupt(image, self.name, self.severity, image_seed)
        return self.transform(image), row["class_id"]


@torch.no_grad()
def evaluate(model: torch.nn.Module, config: Config, split: str, device: torch.device,
             seed: int) -> list[dict]:
    if split == "test" and config.smoke_limit is not None:
        raise ValueError("Smoke checkpoints cannot be used for final test reports")
    source = ManifestDataset(Path(config.data_root), Path(config.manifest), split,
                             config.model, limit=config.smoke_limit)
    if not source:
        raise ValueError("Empty held-out split")
    model.eval()
    records = []
    for name in ("clean",) + NAMES:
        for severity in ((0,) if name == "clean" else SEVERITIES):
            data = DataLoader(CorruptedDataset(source, config.model, name, severity, seed),
                              batch_size=config.batch_size, num_workers=config.num_workers)
            correct = 0
            for images, labels in data:
                outputs = model(images.to(device))
                correct += int((outputs.argmax(1).cpu() == labels).sum())
            records.append({"corruption": name, "severity": severity, "seed": seed,
                            "seed_derivation": "base seed + crc32(path:type:severity) modulo 2^32",
                            "split": split, "accuracy": correct / len(source), "count": len(source)})
    return records


def save_results(records: list[dict], output: Path) -> None:
    from traffic_signs.plotting import plt

    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(records, indent=2))
    fig, ax = plt.subplots()
    ax.axhline(records[0]["accuracy"], linestyle="--", color="black", label="Clean")
    for name in NAMES:
        values = [r["accuracy"] for r in records if r["corruption"] == name]
        ax.plot(SEVERITIES, values, marker="o", label=name)
    ax.set(xlabel="Severity", ylabel="Accuracy", title=f"Robustness on {records[0]['split']}")
    ax.legend()
    fig.savefig(output / "robustness.png", bbox_inches="tight")
    plt.close(fig)
