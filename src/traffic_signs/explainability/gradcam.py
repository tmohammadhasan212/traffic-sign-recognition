"""Grad-CAM for the final spatial feature block of either model."""

import numpy as np
import torch
from PIL import Image


def gradcam(model: torch.nn.Module, tensor: torch.Tensor,
            class_id: int | None = None) -> np.ndarray:
    if hasattr(model, "layer4"):
        layer = model.layer4
    elif hasattr(model, "features"):
        layer = model.features[10]
    else:
        raise ValueError("Grad-CAM is unsupported for this model")
    activations = []
    gradients = []
    forward = layer.register_forward_hook(lambda _m, _i, output: activations.append(output))
    backward = layer.register_full_backward_hook(
        lambda _m, _gi, output: gradients.append(output[0]))
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        # Frozen backbones still need an autograd path through their activations.
        logits = model(tensor.detach().requires_grad_(True))
        target = int(logits.argmax(1)[0]) if class_id is None else class_id
        logits[0, target].backward()
        if not gradients:
            raise ValueError("Grad-CAM gradients were unavailable for this model")
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
