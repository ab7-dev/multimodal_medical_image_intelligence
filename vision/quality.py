"""Basic, non-diagnostic input quality checks for chest radiographs."""

from pathlib import Path

import cv2
import numpy as np


BLUR_THRESHOLD = 100.0
BRIGHT_MIN = 0.15
BRIGHT_MAX = 0.85
CONTRAST_MIN = 0.05


def check_quality(image):
    if isinstance(image, (str, Path)):
        source = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    else:
        source = np.asarray(image)
    if source is None or source.size == 0:
        raise ValueError(f"Could not read image: {image}")

    original_shape = source.shape
    clearly_color = False
    if source.ndim == 3 and source.shape[2] >= 3:
        color = source[:, :, :3].astype(np.float32)
        clearly_color = float(np.mean(np.max(color, axis=2) - np.min(color, axis=2))) > 18.0

    gray = source if source.ndim == 2 else cv2.cvtColor(source[:, :, :3], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA).astype(np.uint8)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    height, width = original_shape[:2]
    aspect_ratio = width / height
    extreme_aspect = aspect_ratio < 0.4 or aspect_ratio > 1.6
    # Grayscale photos can fool these simple appearance heuristics.
    is_chest_xray = not clearly_color and not extreme_aspect

    issues = []
    if blur < BLUR_THRESHOLD:
        issues.append("blurry")
    if brightness < BRIGHT_MIN:
        issues.append("too_dark")
    if brightness > BRIGHT_MAX:
        issues.append("too_bright")
    if contrast < CONTRAST_MIN:
        issues.append("low_contrast")
    if not is_chest_xray:
        issues.append("not_chest_xray")
    return {
        "blur": blur,
        "brightness": brightness,
        "contrast": contrast,
        "is_chest_xray": is_chest_xray,
        "status": "good" if not issues else "poor",
        "issues": issues,
    }
