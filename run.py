"""
Launcher script for MedAssist-AI: Multimodal Medical Image Intelligence System.
Optimized for Ryzen 7 + RTX 3050 (4GB-6GB VRAM) & 16GB RAM.
"""

import sys
import os
import uvicorn
import torch

# Ensure current directory is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.config import get_hardware_config

def main():
    hw = get_hardware_config()
    print("=" * 70)
    print("  MEDASSIST-AI: MULTIMODAL MEDICAL IMAGE INTELLIGENCE SYSTEM")
    print("=" * 70)
    print(f"  Device:           {hw.device.upper()} ({hw.gpu_name})")
    if hw.is_cuda:
        print(f"  GPU Acceleration: Enabled (NVIDIA CUDA)")
        print(f"  Total VRAM:       {hw.vram_total_mb} MB")
    else:
        print(f"  CPU Acceleration: Enabled (Ryzen 7 Vectorized Fallback)")
    print(f"  Computer Vision:  DenseNet-121 + Grad-CAM Heatmaps + ROI Contours")
    print(f"  Multimodal Logic: Clinical Notes Fusion + Anti-Hallucination Gate")
    print("=" * 70)
    print("  Server starting at: http://localhost:8000")
    print("  Opening Clinical Diagnostic Cockpit in browser...")
    print("=" * 70)

    uvicorn.run(
        "backend.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
