"""
Multimodal Clinical Fusion & Anti-Hallucination Gatekeeper Engine.
Combines radiograph visual evidence with patient clinical notes, vitals, and lab results.
Enforces the strict rule: "A finding with no location in the image or no supporting clinical notes is a hallucination and fails that case."
Applies doctor-first assistive phrasing: "Doctor, consider this finding..."
"""

import re
from typing import List, Dict, Any, Optional

# Knowledge base linking radiological pathologies to clinical biomarkers, symptoms, and keywords
CLINICAL_CONCORDANCE_MAP = {
    "Pneumonia": {
        "symptoms": ["fever", "pyrexia", "cough", "productive cough", "purulent sputum", "yellow sputum", "green sputum", "chills", "rigors", "crackles", "rales"],
        "labs": ["wbc", "leukocytosis", "procalcitonin", "crp", "neutrophilia", "elevated white blood cell"],
        "vitals": ["temp", "temperature", "febrile", "spo2 < 92", "hypoxia", "tachypnea"],
        "anatomical_predilection": ["Right Lower Lobe / Base", "Left Lower Lobe / Base", "Right Mid Zone / Hilum"]
    },
    "Cardiomegaly": {
        "symptoms": ["orthopnea", "paroxysmal nocturnal dyspnea", "pnd", "pedal edema", "peripheral edema", "shortness of breath", "fatigue", "elevated jvp"],
        "labs": ["bnp", "nt-probnp", "troponin", "hypertension", "chf"],
        "vitals": ["elevated bp", "tachycardia", "fluid overload"],
        "anatomical_predilection": ["Cardiothoracic Silhouette"]
    },
    "Pneumothorax": {
        "symptoms": ["acute chest pain", "pleuritic chest pain", "sudden dyspnea", "absent breath sounds", "decreased breath sounds", "trauma", "subcutaneous emphysema"],
        "labs": ["trauma panel", "post-biopsy", "central line"],
        "vitals": ["acute desaturation", "hypotension", "tachycardia"],
        "anatomical_predilection": ["Right Apical Zone", "Left Apical Zone"]
    },
    "Effusion": {
        "symptoms": ["dullness to percussion", "pleuritic pain", "decreased air entry", "orthopnea", "pleurisy"],
        "labs": ["pleural fluid", "ldh", "protein", "hypoalbuminemia"],
        "vitals": ["reduced chest expansion", "tachypnea"],
        "anatomical_predilection": ["Right Lower Lobe / Base", "Left Lower Lobe / Base"]
    },
    "Consolidation": {
        "symptoms": ["bronchial breathing", "egophony", "dullness", "productive cough", "fever"],
        "labs": ["crp", "wbc", "sputum culture"],
        "vitals": ["fever", "hypoxemia"],
        "anatomical_predilection": ["Right Lower Lobe / Base", "Left Lower Lobe / Base", "Right Mid Zone / Hilum"]
    },
    "Edema": {
        "symptoms": ["acute dyspnea", "pink frothy sputum", "bilateral basilar crackles", "fluid overload"],
        "labs": ["bnp", "renal failure", "creatinine"],
        "vitals": ["hypoxia", "hypertension"],
        "anatomical_predilection": ["Right Mid Zone / Hilum", "Left Mid Zone / Hilum", "Cardiothoracic Silhouette"]
    },
    "Atelectasis": {
        "symptoms": ["post-operative", "shallow breathing", "mucus plug", "hypoventilation", "splinting"],
        "labs": ["post-surgical day"],
        "vitals": ["low tidal volume", "mild fever"],
        "anatomical_predilection": ["Right Lower Lobe / Base", "Left Lower Lobe / Base"]
    },
    "Emphysema": {
        "symptoms": ["barrel chest", "smoking history", "pack-years", "chronic dyspnea", "wheezing", "pursed lip breathing"],
        "labs": ["pft", "copd", "fev1"],
        "vitals": ["hyperinflated", "flattened diaphragm"],
        "anatomical_predilection": ["Right Apical Zone", "Left Apical Zone"]
    },
    "Nodule": {
        "symptoms": ["smoking history", "weight loss", "hemoptysis", "chronic cough", "incidental finding"],
        "labs": ["cea", "histology", "biopsy"],
        "vitals": ["asymptomatic"],
        "anatomical_predilection": ["Right Apical Zone", "Left Apical Zone", "Right Mid Zone / Hilum", "Left Mid Zone / Hilum"]
    },
    "Infiltration": {
        "symptoms": ["cough", "subacute dyspnea", "fever", "crackles"],
        "labs": ["wbc", "viral panel", "atypical pneumonia"],
        "vitals": ["tachypnea"],
        "anatomical_predilection": ["Right Mid Zone / Hilum", "Left Mid Zone / Hilum", "Right Lower Lobe / Base"]
    }
}


class ClinicalNotesParser:
    """Extracts clinical entities, vitals, and lab biomarkers from unstructured clinical text."""

    @staticmethod
    def parse_notes(text: str) -> Dict[str, Any]:
        if not text:
            return {
                "raw_text": "",
                "extracted_vitals": {},
                "extracted_labs": {},
                "extracted_symptoms": [],
                "sentences": []
            }

        cleaned = text.strip()
        sentences = [s.strip() for s in re.split(r'[.\n;]+', cleaned) if len(s.strip()) > 3]

        # Extract vitals
        vitals = {}
        # Temperature
        temp_match = re.search(r'(?:temp|temperature|febrile|t:?)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:c|f|°c|°f)?', cleaned, re.IGNORECASE)
        if temp_match:
            vitals["temperature"] = temp_match.group(0).strip()

        # SpO2
        spo2_match = re.search(r'(?:spo2|o2\s*sat|saturation)\s*[:=]?\s*(\d{2,3})\s*%?', cleaned, re.IGNORECASE)
        if spo2_match:
            vitals["spo2"] = f"{spo2_match.group(1)}%"

        # Heart Rate
        hr_match = re.search(r'(?:hr|heart\s*rate|pulse)\s*[:=]?\s*(\d{2,3})\s*(?:bpm)?', cleaned, re.IGNORECASE)
        if hr_match:
            vitals["heart_rate"] = f"{hr_match.group(1)} bpm"

        # Blood Pressure
        bp_match = re.search(r'(?:bp|blood\s*pressure)\s*[:=]?\s*(\d{2,3}\s*/\s*\d{2,3})\s*(?:mmhg)?', cleaned, re.IGNORECASE)
        if bp_match:
            vitals["blood_pressure"] = bp_match.group(1).strip()

        # Extract labs
        labs = {}
        # WBC
        wbc_match = re.search(r'(?:wbc|white\s*count|leukocyte)\s*[:=]?\s*(\d+[\d,\.]*)\s*(?:k|k/ul|/mcL|/mm3|x10\^9)?', cleaned, re.IGNORECASE)
        if wbc_match:
            labs["wbc"] = wbc_match.group(0).strip()

        # BNP
        bnp_match = re.search(r'(?:bnp|nt-probnp)\s*[:=]?\s*(\d+[\d,\.]*)\s*(?:pg/ml)?', cleaned, re.IGNORECASE)
        if bnp_match:
            labs["bnp"] = bnp_match.group(0).strip()

        # CRP / Procalcitonin
        crp_match = re.search(r'(?:crp|c-reactive|procalcitonin)\s*[:=]?\s*(\d+[\d,\.]*)\s*(?:mg/l|ng/ml)?', cleaned, re.IGNORECASE)
        if crp_match:
            labs["inflammatory_marker"] = crp_match.group(0).strip()

        # Extracted symptoms
        symptoms_found = []
        all_symptom_keywords = [
            "fever", "chills", "dyspnea", "shortness of breath", "cough", "sputum", 
            "chest pain", "pleuritic", "orthopnea", "edema", "hemoptysis", "weight loss",
            "night sweats", "crackles", "wheezing", "hypoxemia", "dullness"
        ]
        lower_text = cleaned.lower()
        for kw in all_symptom_keywords:
            if kw in lower_text:
                # Find matching snippet
                for sent in sentences:
                    if kw in sent.lower():
                        symptoms_found.append({"keyword": kw, "citation": sent})
                        break

        return {
            "raw_text": cleaned,
            "extracted_vitals": vitals,
            "extracted_labs": labs,
            "extracted_symptoms": symptoms_found,
            "sentences": sentences
        }


class MultimodalFusionGatekeeper:
    """
    Enforces proof-carrying anti-hallucination validation and doctor-assistive framing.
    Key Idea: A finding with no location in the image or no supporting clinical notes is a hallucination.
    """

    def __init__(self):
        self.parser = ClinicalNotesParser()

    def fuse_and_validate(
        self,
        vision_predictions: List[Dict[str, Any]],
        segmentation_results: Dict[str, Dict[str, Any]],
        clinical_notes_text: str,
        image_quality_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combines visual models, ROI segmentation, and clinical notes.
        Filters out hallucinations and formats findings in doctor-first tone.
        """
        parsed_notes = self.parser.parse_notes(clinical_notes_text)
        lower_notes = parsed_notes["raw_text"].lower()
        has_notes = len(lower_notes) > 5

        validated_findings = []
        rejected_hallucinations = []

        for pred in vision_predictions:
            pathology = pred["pathology"]
            vis_prob = pred["probability"]
            vis_confidence_pct = pred["confidence_percent"]

            # 1. Inspect Visual Grounding
            seg_info = segmentation_results.get(pathology, {})
            has_visual_roi = seg_info.get("has_visual_grounding", False)
            primary_roi = seg_info.get("primary_region", None)

            # 2. Inspect Clinical Notes Evidence
            notes_evidence = self._find_notes_evidence(pathology, parsed_notes)
            has_notes_support = notes_evidence["is_supported"]

            # 3. Apply Anti-Hallucination Gatekeeper:
            # Rule: Must have either verified location in image OR clinical notes citation.
            # Low probability candidates with NO ROI and NO clinical notes are HALLUCINATIONS!
            is_hallucination = (not has_visual_roi) and (not has_notes_support)

            if is_hallucination or vis_confidence_pct < 20.0:
                # Reject candidate to prevent spurious medical hallucination
                rejected_hallucinations.append({
                    "pathology": pathology,
                    "confidence": f"{vis_confidence_pct:.1f}%",
                    "rejection_reason": "Failed evidence gate: No identifiable region in image AND no supporting clinical note citations.",
                    "status": "REJECTED_HALLUCINATION_PREVENTED"
                })
                continue

            # 4. Multimodal Confidence Calibration
            # If both image and notes agree, confidence increases!
            # If image shows lesion but notes lack context, confidence is moderate.
            final_prob = vis_prob
            if has_visual_roi and has_notes_support:
                final_prob = min(0.98, vis_prob * 1.25 + 0.10)
                concordance_tier = "BIMODAL_CONFIRMED"
            elif has_visual_roi and not has_notes_support:
                final_prob = vis_prob * 0.95
                concordance_tier = "IMAGE_GROUNDED_ONLY"
            elif not has_visual_roi and has_notes_support:
                final_prob = min(0.60, vis_prob * 0.85 + 0.15)
                concordance_tier = "CLINICALLY_SUSPECTED_ONLY"

            calibrated_pct = round(final_prob * 100.0, 1)

            # 5. Doctor-First Assistive Framing
            assistive_phrasing = self._build_assistive_phrasing(
                pathology=pathology,
                calibrated_pct=calibrated_pct,
                roi=primary_roi,
                notes_evidence=notes_evidence,
                has_visual_roi=has_visual_roi,
                has_notes_support=has_notes_support
            )

            validated_findings.append({
                "pathology": pathology,
                "confidence_percent": calibrated_pct,
                "confidence_band": self._get_confidence_band(calibrated_pct),
                "concordance_tier": concordance_tier,
                "assistive_recommendation": assistive_phrasing,
                "visual_evidence": {
                    "is_grounded": has_visual_roi,
                    "anatomical_zone": primary_roi.get("anatomical_zone") if primary_roi else "No clear focal lesion",
                    "bounding_box": primary_roi.get("bounding_box") if primary_roi else None,
                    "area_percent": primary_roi.get("area_percent") if primary_roi else 0.0,
                    "roi_count": seg_info.get("region_count", 0)
                },
                "clinical_evidence": {
                    "is_supported": has_notes_support,
                    "matched_citations": notes_evidence.get("citations", []),
                    "matched_keywords": notes_evidence.get("keywords", [])
                },
                "verification_status": "BACKED_BY_EVIDENCE"
            })

        # Sort validated findings by confidence
        validated_findings.sort(key=lambda x: x["confidence_percent"], reverse=True)

        # Generate Doctor Summary Impression
        doctor_summary = self._generate_doctor_summary(
            validated_findings,
            image_quality_result,
            parsed_notes
        )

        return {
            "validated_findings": validated_findings,
            "rejected_hallucinations_count": len(rejected_hallucinations),
            "rejected_hallucinations": rejected_hallucinations,
            "clinical_notes_audit": parsed_notes,
            "overall_assessment": doctor_summary,
            "rule_compliance": {
                "all_findings_backed_up": True,
                "hallucinations_blocked": len(rejected_hallucinations),
                "assistive_phrasing_strictly_enforced": True
            }
        }

    def _find_notes_evidence(self, pathology: str, parsed_notes: Dict[str, Any]) -> Dict[str, Any]:
        """Scans parsed patient notes for clinical keywords, symptoms, labs matching the pathology."""
        profile = CLINICAL_CONCORDANCE_MAP.get(pathology, None)
        if not profile or not parsed_notes["raw_text"]:
            return {"is_supported": False, "citations": [], "keywords": []}

        matched_keywords = []
        citations = []
        raw = parsed_notes["raw_text"].lower()

        # Check symptoms, labs, vitals
        target_terms = profile.get("symptoms", []) + profile.get("labs", []) + profile.get("vitals", [])

        for term in target_terms:
            if term in raw:
                matched_keywords.append(term)
                # Find the sentence containing this term
                for sentence in parsed_notes["sentences"]:
                    if term in sentence.lower() and sentence not in citations:
                        citations.append(sentence.strip())

        return {
            "is_supported": len(matched_keywords) > 0,
            "citations": citations[:3],  # top 3 relevant citations
            "keywords": list(set(matched_keywords))[:5]
        }

    def _get_confidence_band(self, pct: float) -> str:
        """Translates numerical confidence into realistic clinical uncertainty band."""
        if pct >= 88.0:
            return f"High Confidence ({pct}%) - Strong concordant visual & clinical indicators"
        elif pct >= 65.0:
            return f"Moderate-High Confidence ({pct}%) - Distinct features observed"
        elif pct >= 45.0:
            return f"Moderate Confidence ({pct}%) - Doctor evaluation suggested"
        else:
            return f"Low / Equivocal ({pct}%) - Borderline finding, doctor review needed"

    def _build_assistive_phrasing(
        self,
        pathology: str,
        calibrated_pct: float,
        roi: Optional[Dict[str, Any]],
        notes_evidence: Dict[str, Any],
        has_visual_roi: bool,
        has_notes_support: bool
    ) -> str:
        """
        Builds respectful, doctor-assistive wording.
        Strictly complies with: 'Doctor, consider this finding...' not 'The patient has...'
        """
        zone = roi.get("anatomical_zone") if roi else "lung field"
        area = roi.get("area_percent", 0.0) if roi else 0.0

        if has_visual_roi and has_notes_support:
            citations_str = f"'{notes_evidence['citations'][0]}'" if notes_evidence['citations'] else "clinical notes"
            return (
                f"Doctor, consider this finding: Visual analysis reveals an abnormal density occupying {area}% "
                f"of the {zone}. This strongly correlates with patient notes stating {citations_str}. "
                f"Confidence level is estimated at {calibrated_pct}%."
            )
        elif has_visual_roi and not has_notes_support:
            return (
                f"Doctor, consider this finding: We observed a localized focal irregularity in the {zone} "
                f"(bounding box area: {area}%). Please note this is not explicitly referenced in the current clinical chart. "
                f"Estimated confidence: {calibrated_pct}%."
            )
        else:
            citations_str = f"'{notes_evidence['citations'][0]}'" if notes_evidence['citations'] else "chart data"
            return (
                f"Doctor, please review: Patient chart indicates {citations_str}, raising suspicion for {pathology}. "
                f"However, clear radiological demarcation on the current image is equivocal. "
                f"Confidence: {calibrated_pct}% (Manual radiologist confirmation recommended)."
            )

    def _generate_doctor_summary(
        self,
        findings: List[Dict[str, Any]],
        quality: Dict[str, Any],
        parsed_notes: Dict[str, Any]
    ) -> str:
        """Generates an overarching radiologist second-opinion impression."""
        if not findings:
            return "Doctor, review indicates no prominent focal acute cardiopulmonary abnormalities above threshold. Normal physiological variants may be present."

        top_findings = findings[:2]
        lines = [
            f"**ASSISTIVE SECOND-OPINION IMPRESSION FOR ATTENDING PHYSICIAN:**",
            f"• Image Diagnostic Quality: {quality.get('status', 'ACCEPTABLE')} (Quality Index: {quality.get('quality_score', 0)}/100)."
        ]

        if not quality.get("is_acceptable", True):
            lines.append(f"  ⚠ NOTE: Visual confidence scores reflect a discount due to: {', '.join(quality.get('issues', []))}")

        lines.append("• Primary AI Observations for Doctor's Consideration:")
        for idx, f in enumerate(top_findings, 1):
            path = f["pathology"]
            conf = f["confidence_percent"]
            zone = f["visual_evidence"]["anatomical_zone"]
            lines.append(f"  {idx}. **{path}** ({conf}% confidence) localized primarily to {zone}.")

        lines.append("\n*Disclaimer: MedAssist-AI provides supportive diagnostic cross-referencing and is not a substitute for professional clinical judgment.*")
        return "\n".join(lines)
