"""Validated experiment configuration."""

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    name: str = "baseline"
    model: str = "cnn"
    pretrained: bool = False
    frozen: bool = False
    robust_augmentation: bool = False
    data_root: str = "data/raw"
    manifest: str = "data/split_manifest.json"
    output_dir: str = "outputs/baseline"
    seed: int = 2026
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    optimizer: str = "adamw"
    weight_decay: float = 0.0001
    scheduler: str = "cosine"
    patience: int = 5
    device: str = "auto"
    num_workers: int = 0
    smoke_limit: int | None = None

    def __post_init__(self):
        if self.model not in {"cnn", "resnet18"} or self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("Invalid model or optimizer")
        if self.scheduler not in {"none", "cosine"} or self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("Invalid scheduler or device")
        if min(self.epochs, self.batch_size, self.patience) <= 0 or self.learning_rate <= 0:
            raise ValueError("Epochs, batch size, patience and learning rate must be positive")
        if self.model == "cnn" and (self.pretrained or self.frozen):
            raise ValueError("CNN cannot be pretrained or frozen")

    @classmethod
    def load(cls, path: Path) -> "Config":
        values = yaml.safe_load(path.read_text())
        if not isinstance(values, dict):
            raise ValueError("Config must be a mapping")
        return cls(**values)

    def dump(self) -> dict:
        return asdict(self)
