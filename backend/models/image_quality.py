"""
Image Quality Assessment (IQA) for Medical Chest Radiographs.
Detects blurriness, contrast deficiency, under/over-exposure, and generates an objective quality score.
"""

import cv2
import numpy as np
from typing import Dict, Any

class ImageQualityAuditor:
    """Evaluates medical image fidelity to prevent false positives on degraded inputs."""
    
    def __init__(self, blur_threshold: float = 85.0, min_contrast: float = 38.0):
        self.blur_threshold = blur_threshold
        self.min_contrast = min_contrast

    def assess_quality(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs comprehensive optical and radiological quality checks on the input image."""
        if image_bgr is None or image_bgr.size == 0:
            return {
                "quality_score": 0.0,
                "status": "INVALID",
                "is_acceptable": False,
                "issues": ["Invalid image data"],
                "recommendation": "Unable to decode image file. Please provide a valid DICOM/PNG/JPG."
            }

        # Convert to grayscale
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr

        # 1. Blur Detection using Variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(laplacian.var())
        is_blurry = lap_var < self.blur_threshold

        # 2. Contrast & Dynamic Range
        mean_intensity = float(np.mean(gray))
        contrast_std = float(np.std(gray))
        is_low_contrast = contrast_std < self.min_contrast

        # 3. Exposure Clipping Check
        total_pixels = float(gray.size)
        under_exposed_pct = float(np.sum(gray < 10) / total_pixels) * 100.0
        over_exposed_pct = float(np.sum(gray > 245) / total_pixels) * 100.0

        is_underexposed = under_exposed_pct > 25.0
        is_overexposed = over_exposed_pct > 20.0

        # Compile identified issues
        issues = []
        confidence_penalty = 1.0  # multiplier for downstream model confidence

        if is_blurry:
            issues.append(f"Excessive motion/geometric blur detected (Sharpness index: {lap_var:.1f})")
            confidence_penalty *= 0.70

        if is_low_contrast:
            issues.append(f"Suboptimal radiological contrast dynamic range ({contrast_std:.1f})")
            confidence_penalty *= 0.85

        if is_underexposed:
            issues.append(f"Significant underexposure ({under_exposed_pct:.1f}% clipped dark pixels)")
            confidence_penalty *= 0.80

        if is_overexposed:
            issues.append(f"Severe overexposure / burnout ({over_exposed_pct:.1f}% clipped white pixels)")
            confidence_penalty *= 0.75

        # Compute composite quality score (0 - 100)
        sharpness_score = min(100.0, (lap_var / (self.blur_threshold * 1.8)) * 100.0)
        contrast_score = min(100.0, (contrast_std / (self.min_contrast * 1.5)) * 100.0)
        exposure_penalty = (under_exposed_pct + over_exposed_pct) * 0.8
        
        quality_score = max(5.0, min(100.0, (0.5 * sharpness_score + 0.5 * contrast_score) - exposure_penalty))

        if quality_score >= 75.0:
            status = "EXCELLENT"
            recommendation = "Optimal diagnostic quality for visual AI assessment."
        elif quality_score >= 48.0:
            status = "ACCEPTABLE"
            recommendation = "Diagnostic quality is adequate; some minor artifacts observed."
        else:
            status = "DEGRADED"
            recommendation = "Doctor, caution is advised: Image quality is degraded. Visual findings have reduced confidence."

        return {
            "quality_score": round(quality_score, 1),
            "status": status,
            "is_acceptable": quality_score >= 48.0,
            "sharpness_index": round(lap_var, 1),
            "contrast_index": round(contrast_std, 1),
            "mean_luminance": round(mean_intensity, 1),
            "underexposed_pct": round(under_exposed_pct, 1),
            "overexposed_pct": round(over_exposed_pct, 1),
            "issues": issues,
            "confidence_penalty": round(confidence_penalty, 2),
            "recommendation": recommendation
        }
