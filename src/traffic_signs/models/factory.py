"""Model definitions and transfer-learning modes."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from traffic_signs.data.labels import NUM_CLASSES


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 16, 256),
                                        nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, NUM_CLASSES))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def create_model(name: str, pretrained: bool = False, frozen: bool = False) -> nn.Module:
    if name == "cnn":
        if pretrained or frozen:
            raise ValueError("The scratch CNN cannot use pretrained or frozen modes")
        return SmallCNN()
    if name != "resnet18":
        raise ValueError(f"Unknown model: {name}")
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    if frozen:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
