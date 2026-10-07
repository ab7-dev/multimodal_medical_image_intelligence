"""Small Grad-CAM implementation for a selected convolutional layer."""

import cv2
import numpy as np
import torch


def normalize_heatmap(heatmap):
    heatmap = np.maximum(np.asarray(heatmap, dtype=np.float32), 0)
    maximum = float(heatmap.max()) if heatmap.size else 0.0
    if maximum <= 0:
        return np.zeros_like(heatmap, dtype=np.float32)
    return heatmap / maximum


def largest_region(heatmap):
    normalized = normalize_heatmap(heatmap)
    if normalized.size == 0 or float(normalized.max()) <= 0:
        return None
    binary = (normalized >= 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    image_height, image_width = normalized.shape[:2]
    return {
        "x": x / image_width,
        "y": y / image_height,
        "w": width / image_width,
        "h": height / image_height,
    }


def localization_is_usable(heatmap):
    """Reject near-full-frame or edge-touching CAMs rather than imply localization.

    This conservative presentation heuristic is not a clinical localization test.
    """
    region = largest_region(heatmap)
    if region is None:
        return False
    x, y, width, height = (region[key] for key in ("x", "y", "w", "h"))
    touches_edge = x <= 0.01 or y <= 0.01 or x + width >= 0.99 or y + height >= 0.99
    near_full_frame = width > 0.85 or height > 0.85 or width * height > 0.5
    return not touches_edge and not near_full_frame


def generate_gradcam(model, input_tensor, target_layer, output_index):
    activations = []
    gradients = []

    def capture_activation(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    handle = target_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        output = model(input_tensor)
        output[0, output_index].backward()
    finally:
        handle.remove()
    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
    activation = activations[-1].detach()
    gradient = gradients[-1].detach()
    channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((channel_weights * activation).sum(dim=1))[0].cpu().numpy()
    return normalize_heatmap(cam)
