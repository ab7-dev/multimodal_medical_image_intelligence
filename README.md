# -Proof-Carrying-Data-Analyst
AI-powered data analyst that processes messy files, understands questions, generates analysis code, verifies results, detects reliability issues, and delivers transparent, trustworthy insights with executable proof.

## Chest X-ray Vision (Part A)

Backend-only pretrained chest X-ray scoring for Pneumonia, Effusion, and Pneumothorax, with input-quality checks and optional Grad-CAM overlays. Model scores are not diagnoses or calibrated probabilities.

### Run

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python vision/check_model.py
python -c "from vision.analyze import analyze_image; print(analyze_image('data/images/example.png'))"
```

Place test images under `data/images/`. NIH evaluation needs `data/nih/Data_Entry_2017.csv`, `data/nih/BBox_List_2017.csv`, and the referenced NIH images under `data/nih/`. No patient images, sample predictions, model weights, or heatmaps are committed. Evaluation remains not evaluated until labels and images are supplied.

Broad or edge-touching CAM overlays are suppressed by a conservative display safeguard; remaining heatmaps are not clinically validated.
