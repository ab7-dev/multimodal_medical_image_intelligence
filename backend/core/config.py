"""
Configuration module for MedAssist-AI: Multimodal Medical Image Intelligence System.
Optimized for Ryzen 7 + NVIDIA RTX 3050 (4GB-6GB VRAM) & 16GB RAM.
"""

import os
import torch
from dataclasses import dataclass

@dataclass
class HardwareConfig:
    device: str
    gpu_name: str = "N/A"
    vram_total_mb: float = 0.0
    vram_allocated_mb: float = 0.0
    is_cuda: bool = False

def get_hardware_config() -> HardwareConfig:
    """Detects available GPU acceleration and memory limits."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 2)
        return HardwareConfig(
            device="cuda",
            gpu_name=gpu_name,
            vram_total_mb=round(total_mem, 1),
            vram_allocated_mb=round(allocated_mem, 1),
            is_cuda=True
        )
    return HardwareConfig(
        device="cpu",
        gpu_name="Ryzen 7 (CPU Fallback)",
        vram_total_mb=0.0,
        vram_allocated_mb=0.0,
        is_cuda=False
    )

# Target Chest X-ray Pathology Classes
PATHOLOGIES = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia"
]

# Anatomical Zones for Chest Radiographs
ANATOMICAL_ZONES = [
    {"name": "Right Apical Zone", "box": [0.05, 0.08, 0.32, 0.45]},
    {"name": "Right Mid Zone / Hilum", "box": [0.32, 0.08, 0.62, 0.45]},
    {"name": "Right Lower Lobe / Base", "box": [0.62, 0.08, 0.90, 0.48]},
    {"name": "Left Apical Zone", "box": [0.05, 0.55, 0.32, 0.92]},
    {"name": "Left Mid Zone / Hilum", "box": [0.32, 0.55, 0.62, 0.92]},
    {"name": "Left Lower Lobe / Base", "box": [0.62, 0.52, 0.90, 0.92]},
    {"name": "Cardiothoracic Silhouette", "box": [0.45, 0.32, 0.85, 0.68]},
    {"name": "Mediastinum / Trachea", "box": [0.08, 0.42, 0.48, 0.58]}
]

# IQA (Image Quality Assessment) Thresholds
IQA_BLUR_THRESHOLD = 80.0       # Laplacian variance below this is blurry
IQA_MIN_CONTRAST = 35.0         # Standard deviation of luminance below this is low contrast
IQA_CLIP_RATIO_MAX = 0.15       # Over 15% saturated white or black pixels is over/underexposed
