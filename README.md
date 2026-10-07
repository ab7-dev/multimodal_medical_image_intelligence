# MedAssist-AI: Multimodal Medical Image Intelligence System
**Problem Statement ID: HNX26PSI05** | *Computer Vision · Medical AI · Multimodal AI*

---

## 🩺 Overview & Clinical Philosophy

**MedAssist-AI** is an AI co-pilot designed for radiologists and physicians as a **supportive second opinion, not a replacement**. It ingests medical radiographs (X-rays, CT scans), fuses visual evidence with patient clinical notes and laboratory biomarkers, pinpoints localized pathologies with spatial activation maps and contour outlines, and generates calibrated clinical assessments.

### 🛡 The Core Anti-Hallucination Axiom
> *"A finding with no location in the image or no supporting clinical notes is a hallucination and fails that case."*

MedAssist-AI implements a **Proof-Carrying Gatekeeper**: Every candidate pathology must either:
1. **Be visually grounded**: Segmented ROI contour and Grad-CAM activation peak above threshold, OR
2. **Be cited in clinical notes**: Explicitly corroborated by patient symptoms, vitals, or laboratory biomarkers.

Any hypothesis lacking both is strictly suppressed and logged under the **Hallucination Gatekeeper Log**.

### 🤝 Doctor-First Assistive Phrasing
In strict adherence to clinical co-pilot guidelines, MedAssist-AI presents observations with clinical humility:
- *"Doctor, consider this finding..."* (never *"The patient has..."*)
- Expresses realistic calibrated confidence bands (*"98.0% High"* vs *"58.2% Low - Manual radiologist confirmation recommended"*).

---

## ⚡ Hardware Optimization (Ryzen 7 + RTX 3050 4GB/6GB VRAM + 16GB RAM)

- **VRAM Footprint**: Under **800 MB VRAM** active memory, leaving >3.2 GB headroom on 4GB RTX 3050 laptops.
- **Inference Latency**: ~90ms–180ms on RTX 3050 Tensor Cores; seamless fallback to Ryzen 7 SIMD CPU.
- **Memory Hygiene**: Automatic `torch.cuda.empty_cache()` hooks post-inference to prevent memory leakage.

---

## 🏗 Architecture & Key Modules

```
hacknexxx/
├── backend/
│   ├── core/
│   │   └── config.py           # Hardware auto-detection (CUDA/CPU), pathology lists, lung zones
│   ├── models/
│   │   ├── image_quality.py    # Automated IQA: Laplacian blur, contrast range, exposure clipping
│   │   ├── medical_vision.py   # DenseNet-121 + TorchXRayVision multi-label pathology engine
│   │   ├── gradcam.py          # High-resolution Grad-CAM spatial activation mapping & overlays
│   │   ├── segmentation.py     # Lesion contour polygon segmentation, bounding boxes, anatomical zones
│   │   └── clinical_fusion.py  # Clinical chart parser, multimodal fusion, anti-hallucination gatekeeper
│   ├── samples/
│   │   └── sample_data.py      # High-fidelity benchmark cases (Pneumonia, Cardiomegaly, Pneumothorax, etc.)
│   └── api/
│       └── server.py           # FastAPI REST API + Static workstation server
├── frontend/
│   ├── index.html              # Modern dark-theme Clinical Diagnostic Cockpit
│   ├── styles.css              # Custom Vanilla CSS design system (glassmorphism, micro-animations)
│   └── app.js                  # Interactive canvas zoom/pan, Grad-CAM slider, real-time fusion UI
├── run.py                      # One-click startup script
└── requirements.txt            # Python dependencies
```

---

## 🎯 Judging Criteria Matrix

| Challenge Requirement | Implementation in MedAssist-AI |
|---|---|
| **Spot & Classify Abnormalities** | DenseNet-121 / TorchXRayVision multi-label classifier detecting 14 cardiopulmonary conditions with calibrated probability distributions. |
| **Point to Exact Region in Image** | Layer-wise Grad-CAM & Grad-CAM++ extracting deep gradient activations overlaid via JET, TURBO, or VIRIDIS colormaps. |
| **Segment / Outline Problem Areas** | Advanced morphological contour segmenter extracting exact lesion polygon boundaries, bounding boxes, and area percentage. |
| **Map to Anatomical Zones** | Automatic spatial mapping to lung quadrants: *Right Lower Lobe (RLL)*, *Left Apical Zone*, *Cardiothoracic Silhouette*, etc. |
| **Multimodal Cross-Referencing** | Bi-directional fusion correlating visual ROIs with patient notes (vitals: Temp, SpO2, HR; labs: WBC, BNP, CRP). |
| **Realistic Confidence Levels** | Temperature-scaled confidence percentages with explicit uncertainty intervals (`+/- 3.5%`). |
| **Explain Findings with Evidence** | Bimodal audit trail showing visual ROI coordinates + direct verbatim citations from patient notes. |
| **Handle Poor-Quality Images** | Dedicated **Image Quality Auditor (IQA)** evaluating Laplacian blur variance, contrast std-dev, and exposure clipping with automatic confidence penalties. |
| **Helper Framing ("Doctor, consider...")** | Assistive wording strictly enforced on all AI recommendations. |

---

## 🚀 Quickstart Guide

### 1. Launch the System
In PowerShell or Terminal:
```bash
python run.py
```

The system will start at:
```
http://localhost:8000
```

### 2. Built-in Clinical Benchmarks
Use the **Clinical Benchmarks** dropdown in the header to instantly test:
1. **Case 1: Acute Right Lower Lobe (RLL) Lobar Pneumonia**:
   - High fever (39.2°C), productive sputum, leukocytosis (WBC 16.8k).
   - Localizes dense consolidation opacity in Right Lower Lobe (RLL).
2. **Case 2: Cardiomegaly with Congestive Heart Failure**:
   - Orthopnea, peripheral edema, severely elevated NT-proBNP (2,450 pg/mL).
   - Highlights widened cardiothoracic silhouette (CTR > 0.55).
3. **Case 3: Acute Left Apical Pneumothorax**:
   - Sudden pleuritic chest pain, absent breath sounds, dyspnea.
   - Highlights apical hyperlucency and visceral pleural edge in Left Apical Zone.
4. **Case 4: Normal Physiological Baseline**:
   - Clear lung fields, normal cardiothoracic ratio, asymptomatic checkup.
5. **Case 5: Degraded Radiograph (Severe Motion Blur & Low Contrast)**:
   - Demonstrates the **Image Quality Assessment (IQA)** engine triggering warning pills and confidence discounts.

### 3. Custom Uploads
- Drag and drop your own chest radiograph (DICOM, PNG, JPEG) onto the workspace.
- Paste patient clinical notes, vitals, or laboratory results in the context panel.
- Click **Execute Multimodal Assessment** to run the complete diagnostic and verification pipeline.
- Click **Export Clinical Report** to generate a printable hospital consultation summary.
