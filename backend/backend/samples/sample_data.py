"""
Realistic Medical Sample Data Generator.
Includes both real pneumonia case and real clean/normal case uploaded by user, plus synthetic benchmarks.
"""

import os
import cv2
import numpy as np

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data")

SAMPLE_CASES = [
    {
        "id": "case_real_patient_pneumonia",
        "title": "Real Patient Case 1: Left Lower Lobe Pneumonia & Infiltration",
        "category": "Real Radiograph",
        "description": "Actual hospital PA radiograph with pronounced dense alveolar opacity in the left mid-to-lower hemithorax.",
        "image_filename": "real_pneumonia_case.png",
        "expected_finding": "Pneumonia",
        "expected_zone": "Left Lower Lobe / Base",
        "clinical_notes": (
            "PATIENT PROFILE: 49 y/o female admitted with productive cough and left pleuritic chest pain.\n"
            "CHIEF COMPLAINT: 4-day history of high fever (39.1°C), chills, and productive cough with purulent sputum. Sharp pain in left lower lateral chest aggravated by deep inspiration.\n"
            "PHYSICAL EXAM & VITALS: Temp: 39.1 C. HR: 104 bpm. BP: 128/82 mmHg. SpO2: 92% on ambient air. Coarse inspiratory crackles and bronchial breath sounds over the left lower lung field; dullness to percussion on left base.\n"
            "LABORATORY FINDINGS: WBC count: 15,600/mcL with left shift (82% neutrophils). Procalcitonin: 1.85 ng/mL. CRP: 86 mg/L (elevated).\n"
            "IMPRESSION: Suspected community-acquired bacterial pneumonia involving left lower lobe."
        )
    },
    {
        "id": "case_real_clean_normal",
        "title": "Real Patient Case 2: Clean Normal Radiograph (No Pneumonia)",
        "category": "Clear Baseline",
        "description": "Actual healthy adult PA radiograph with clear, dark, well-aerated bilateral lung fields and normal heart size.",
        "image_filename": "clean_normal_case.png",
        "expected_finding": "Normal / Clear Hemithoraces",
        "expected_zone": "Clear Hemithoraces",
        "clinical_notes": (
            "PATIENT PROFILE: 34 y/o male presenting for routine occupational physical clearance.\n"
            "CHIEF COMPLAINT: Completely asymptomatic. Denies fever, cough, sputum production, dyspnea, or pleuritic pain.\n"
            "PHYSICAL EXAM & VITALS: Temp: 36.6 C. HR: 72 bpm. BP: 118/74 mmHg. SpO2: 99% on room air. Clear vesicular breath sounds bilaterally with no rales, crackles, or rhonchi. Heart sounds regular S1/S2.\n"
            "LABORATORY FINDINGS: WBC count: 6,400/mcL (normal reference range). EKG normal sinus rhythm.\n"
            "IMPRESSION: Unremarkable baseline examination. Normal cardiopulmonary status."
        )
    },
    {
        "id": "case_pneumonia",
        "title": "Case 3: Right Lower Lobe (RLL) Lobar Pneumonia",
        "category": "Pneumonia",
        "description": "54-year-old male with acute onset fever, chills, and productive cough with purulent sputum.",
        "image_filename": "case1_pneumonia_rll.png",
        "expected_finding": "Pneumonia",
        "expected_zone": "Right Lower Lobe / Base",
        "clinical_notes": (
            "PATIENT PROFILE: 54 y/o male admitted via Emergency Dept.\n"
            "CHIEF COMPLAINT: High-grade fever for 3 days, shaking chills, productive cough with yellow-green sputum, and right-sided pleuritic discomfort.\n"
            "PHYSICAL EXAM & VITALS: Temp: 39.2 C. HR: 108 bpm. BP: 132/84 mmHg. SpO2: 91% on room air. Decreased breath sounds and coarse crackles over the right lower hemithorax with dullness to percussion.\n"
            "LABORATORY FINDINGS: WBC count: 16,800/mcL (leukocytosis). Procalcitonin: 2.1 ng/mL. CRP: 78 mg/L.\n"
            "PREVIOUS HISTORY: Mild type 2 diabetes. No previous pneumonia."
        )
    },
    {
        "id": "case_cardiomegaly",
        "title": "Case 4: Cardiomegaly with Congestive Heart Failure",
        "category": "Cardiomegaly",
        "description": "68-year-old female presenting with worsening orthopnea, paroxysmal nocturnal dyspnea, and peripheral edema.",
        "image_filename": "case2_cardiomegaly_chf.png",
        "expected_finding": "Cardiomegaly",
        "expected_zone": "Cardiothoracic Silhouette",
        "clinical_notes": (
            "PATIENT PROFILE: 68 y/o female with chronic hypertension.\n"
            "CHIEF COMPLAINT: Progressive shortness of breath over 2 weeks, requiring 3 pillows to sleep (orthopnea), bilateral ankle swelling.\n"
            "PHYSICAL EXAM & VITALS: Temp: 36.8 C. HR: 88 bpm. BP: 168/96 mmHg. SpO2: 93% on ambient air. Elevated JVP noted (+4 cm). Bilateral pitting pedal edema. S3 gallop audible at apex.\n"
            "LABORATORY FINDINGS: NT-proBNP: 2,450 pg/mL (severely elevated). Serum Creatinine: 1.4 mg/dL. Troponin I negative.\n"
            "PREVIOUS HISTORY: Longstanding essential hypertension, CHF NYHA Class II."
        )
    },
    {
        "id": "case_pneumothorax",
        "title": "Case 5: Acute Left Apical Pneumothorax",
        "category": "Pneumothorax",
        "description": "24-year-old tall male with sudden sharp left-sided chest pain and dyspnea after physical exertion.",
        "image_filename": "case3_pneumothorax_left.png",
        "expected_finding": "Pneumothorax",
        "expected_zone": "Left Apical Zone",
        "clinical_notes": (
            "PATIENT PROFILE: 24 y/o male, tall slender asthenic habitus.\n"
            "CHIEF COMPLAINT: Sudden acute pleuritic chest pain on left side radiating to shoulder, followed by acute dyspnea.\n"
            "PHYSICAL EXAM & VITALS: Temp: 36.6 C. HR: 114 bpm (tachycardia). BP: 124/76 mmHg. SpO2: 89% on room air. Markedly decreased breath sounds on left hemithorax with hyperresonance to percussion.\n"
            "LABORATORY FINDINGS: WBC count: 7,400/mcL (normal). EKG shows sinus tachycardia without ischemic ST changes.\n"
            "PREVIOUS HISTORY: Active smoker (5 pack-years)."
        )
    },
    {
        "id": "case_poor_quality",
        "title": "Case 6: Degraded Radiograph (Motion Blur & Underexposure)",
        "category": "Quality Warning",
        "description": "Suboptimal portable ICU radiograph with motion blur and underexposure. Tests Image Quality Assessment (IQA).",
        "image_filename": "case5_poor_quality_blurry.png",
        "expected_finding": "Quality Warning",
        "expected_zone": "Global Degraded Field",
        "clinical_notes": (
            "PATIENT PROFILE: 72 y/o bedridden ICU patient.\n"
            "CHIEF COMPLAINT: Portable bedside radiograph performed during severe patient agitation and tachypnea.\n"
            "PHYSICAL EXAM & VITALS: Temp: 37.8 C. HR: 122 bpm. SpO2: 90%.\n"
            "TECHNOLOGIST NOTE: High motion artifact during exposure; portable bedside generator under-penetrated due to patient repositioning."
        )
    }
]

def generate_all_samples():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    return SAMPLE_CASES
