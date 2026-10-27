"""Training, checkpointing, prediction collection and experiment metadata."""

import json
import logging
import platform
import random
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from traffic_signs.data.dataset import ManifestDataset
from traffic_signs.evaluation.metrics import classification_metrics
from traffic_signs.models.factory import create_model, parameter_counts
from traffic_signs.training.config import Config

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(requested)


def loader(config: Config, split: str) -> DataLoader:
    dataset = ManifestDataset(Path(config.data_root), Path(config.manifest), split, config.model,
                              config.robust_augmentation, config.smoke_limit)
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=split == "train",
                      num_workers=config.num_workers, generator=generator)


@torch.no_grad()
def collect(model: nn.Module, data: DataLoader, device: torch.device):
    model.eval()
    logits, targets = [], []
    for images, labels in data:
        logits.append(model(images.to(device)).cpu())
        targets.append(labels)
    if not logits:
        raise ValueError("Empty dataset")
    return torch.cat(logits), torch.cat(targets)


def save_checkpoint(path: Path, model: nn.Module, config: Config, epoch: int,
                    validation_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config.dump(), "epoch": epoch,
                "validation_loss": validation_loss}, path)


def load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = Config(**checkpoint["config"])
    model = create_model(config.model, pretrained=False, frozen=config.frozen)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as exc:
        raise ValueError(f"Incompatible checkpoint: {path}") from exc
    return model.to(device).eval(), config


def train(config: Config) -> Path:
    seed_everything(config.seed)
    device = select_device(config.device)
    model = create_model(config.model, config.pretrained, config.frozen).to(device)
    train_loader, validation_loader = loader(config, "train"), loader(config, "validation")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = (torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
                 if config.optimizer == "adamw" else
                 torch.optim.SGD(parameters, lr=config.learning_rate, momentum=0.9,
                                 weight_decay=config.weight_decay))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
                 if config.scheduler == "cosine" else None)
    criterion = nn.CrossEntropyLoss()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=False).stdout.strip() or "unavailable"
    split_metadata = json.loads(Path(config.manifest).read_text())
    (output / "experiment.json").write_text(json.dumps({"config": config.dump(), "git_revision": revision,
        "versions": {name: version(name) for name in
                     ("torch", "torchvision", "numpy", "Pillow", "scikit-learn")},
        "python": platform.python_version(),
        "device": str(device), "parameters": parameter_counts(model),
        "split_manifest": str(Path(config.manifest).resolve()),
        "split_path_sha256": split_metadata["path_sha256"],
        "started_unix": time.time()}, indent=2, default=str))
    started = time.monotonic()
    best, stale, history = float("inf"), 0, []
    for epoch in range(config.epochs):
        model.train()
        if config.frozen:
            model.eval()  # Keep pretrained batch-normalization running statistics fixed.
            model.fc.train()
        total_loss, count, correct = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            count += len(labels)
        logits, targets = collect(model, validation_loader, device)
        val_loss = criterion(logits, targets).item()
        val_probs = logits.softmax(1).numpy()
        val_metrics = classification_metrics(val_probs, targets.numpy())
        record = {"epoch": epoch + 1, "train_loss": total_loss / count,
                  "train_accuracy": correct / count, "validation_loss": val_loss,
                  "validation_accuracy": val_metrics["accuracy"],
                  "validation_macro_f1": val_metrics["macro_f1"]}
        history.append(record)
        LOGGER.info("Epoch %s: %s", epoch + 1, record)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        if val_loss < best:
            best, stale = val_loss, 0
            save_checkpoint(output / "best.pt", model, config, epoch + 1, val_loss)
        else:
            stale += 1
            if stale >= config.patience:
                break
        if scheduler:
            scheduler.step()
    from traffic_signs.plotting import plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [entry["epoch"] for entry in history]
    axes[0].plot(epochs, [entry["train_loss"] for entry in history], label="Train")
    axes[0].plot(epochs, [entry["validation_loss"] for entry in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    axes[1].plot(epochs, [entry["train_accuracy"] for entry in history], label="Train")
    axes[1].plot(epochs, [entry["validation_accuracy"] for entry in history], label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    for axis in axes:
        axis.legend()
    fig.savefig(output / "training_curves.png", bbox_inches="tight")
    plt.close(fig)
    metadata_path = output / "experiment.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update({"training_duration_seconds": time.monotonic() - started,
                     "best_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
                     "best_validation_loss": best, "hardware": platform.platform()})
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return output / "best.pt"
