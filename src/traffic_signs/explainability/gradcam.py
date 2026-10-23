"""Grad-CAM for ResNet-18's last convolutional block."""

import numpy as np
import torch
from PIL import Image


def gradcam(model: torch.nn.Module, tensor: torch.Tensor,
            class_id: int | None = None) -> np.ndarray:
    if not hasattr(model, "layer4"):
        raise ValueError("Grad-CAM is supported for ResNet-18 checkpoints")
    activations = []
    gradients = []
    forward = model.layer4.register_forward_hook(lambda _m, _i, output: activations.append(output))
    backward = model.layer4.register_full_backward_hook(
        lambda _m, _gi, output: gradients.append(output[0]))
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        target = int(logits.argmax(1)[0]) if class_id is None else class_id
        logits[0, target].backward()
        weight = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weight * activations[0]).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=tensor.shape[-2:], mode="bilinear",
                                              align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / cam.max().clamp_min(1e-8)).detach().cpu().numpy()
    finally:
        forward.remove()
        backward.remove()


def overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    base = image.convert("RGB")
    heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(base.size)
    alpha = np.asarray(heat, dtype=np.float32) / 255
    pixels = np.asarray(base, dtype=np.float32).copy()
    red = np.zeros_like(pixels)
    red[:, :, 0] = 255
    return Image.fromarray(np.uint8(np.clip(pixels * (1 - 0.45 * alpha[..., None]) +
                                             red * 0.45 * alpha[..., None], 0, 255)))
