"""
Gemini Multimodal Medical Vision Integration.
Provides SOTA radiograph interpretation, bounding box localization, and clinical correlation.
Supports Gemini 2.0 Flash & Gemini 1.5 Flash via Google Generative AI REST API.
"""

import os
import json
import base64
import requests
from typing import Dict, Any, Optional

SYSTEM_CLINICAL_PROMPT = """You are MedVision AI, an expert multimodal radiographic second-opinion assistant for radiologists and physicians.
Your task:
1. Carefully inspect the chest radiograph.
2. Differentiate between:
   - A CLEAR / NORMAL radiograph (well-aerated, dark bilateral lung fields, sharp costophrenic angles, normal cardiothoracic ratio < 0.50).
   - A PATHOLOGICAL radiograph (focal consolidation/opacity, lobar pneumonia, effusion, pneumothorax, cardiomegaly).
   CRITICAL: If the radiograph is CLEAR and NORMAL without acute focal opacities, DO NOT hallucinate pneumonia or consolidation. Accurately report it as "Normal / Clear Hemithoraces" with 0 findings.
3. If an abnormality is present:
   - Provide the exact pathology name (e.g. Pneumonia, Consolidation, Effusion, Pneumothorax, Cardiomegaly).
   - Identify the exact anatomical zone (e.g. "Left Lower Lobe / Base", "Right Lower Lobe / Base", "Cardiothoracic Silhouette").
   - Provide normalized bounding box coordinates [ymin, xmin, ymax, xmax] scaled 0 to 1000.
   - Correlate with the provided patient clinical notes.
   - Format recommendation in doctor-first assistive phrasing: "Doctor, consider this finding: ..."
   - Estimate calibrated confidence percentage (e.g. 96%).

Output ONLY valid JSON matching this schema:
{
  "is_normal": boolean,
  "impression": string,
  "findings": [
    {
      "pathology": string,
      "confidence_percent": number,
      "anatomical_zone": string,
      "box_2d": [ymin, xmin, ymax, xmax],
      "assistive_recommendation": string,
      "chart_citation": string
    }
  ]
}
"""

class GeminiMedicalVision:
    """Calls Gemini Vision API for state-of-the-art medical multimodal analysis."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def analyze_radiograph(
        self,
        image_bytes: bytes,
        clinical_notes: str = "",
        api_key_override: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        active_key = (api_key_override or self.api_key or "").strip()
        if not active_key or len(active_key) < 8:
            return None

        # Base64 encode image
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        prompt_text = f"Patient Clinical Notes & Vitals:\n{clinical_notes or 'No clinical chart provided. Routine radiograph assessment.'}\n\nPlease interpret this chest radiograph with high diagnostic precision according to the system instructions."

        # Support OpenAI keys (sk-...)
        if active_key.startswith("sk-"):
            try:
                headers = {
                    "Authorization": f"Bearer {active_key}",
                    "Content-Type": "application/json"
                }
                messages = [
                    {"role": "system", "content": SYSTEM_CLINICAL_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_img}"
                                }
                            }
                        ]
                    }
                ]
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": "gpt-4o",
                        "messages": messages,
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"}
                    },
                    timeout=30
                )
                if resp.status_code == 200:
                    raw_text = resp.json()["choices"][0]["message"]["content"]
                    return self._clean_json(raw_text)
                else:
                    print(f"OpenAI API Error {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"OpenAI API connection error: {e}")
            return None

        # Google Gemini Vision (gemini-2.0-flash / gemini-1.5-flash)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": SYSTEM_CLINICAL_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": b64_img
                            }
                        },
                        {"text": prompt_text}
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        }

        models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={active_key}"
            try:
                resp = requests.post(url, json=payload, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        raw_text = candidates[0]["content"]["parts"][0]["text"]
                        return self._clean_json(raw_text)
                else:
                    print(f"Gemini API ({model_name}) error {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"Gemini API connection error: {e}")

        return None

    @staticmethod
    def _clean_json(raw_text: str) -> Optional[Dict[str, Any]]:
        try:
            text = raw_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            print(f"Failed to parse model JSON: {e}")
            return None
