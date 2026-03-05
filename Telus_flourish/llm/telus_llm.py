"""
TELUS FLOURISH – Farmer Summary Generator
----------------------------------------
Attempts GPT-based summary first.
Falls back to rule-based summary if LLM fails.
"""

import requests
import json
from datetime import datetime

# ==================================================
# GPT CONFIG (PRIMARY)
# ==================================================
GPT_ENDPOINT = "https://rr-test-gpt-120-9219s.paas.ai.telus.com/v1/chat/completions"
GPT_API_KEY = "1df668838dee5b8410e8e21a76fd9bb9"
GPT_MODEL = "gpt-oss-120b-sovereign:latest"

HEADERS = {
    "Authorization": f"Bearer {GPT_API_KEY}",
    "Content-Type": "application/json"
}

# ==================================================
# RULE-BASED FALLBACK
# ==================================================
def rule_based_summary(data: dict) -> str:
    disease = data.get("disease", "Crop disease")
    treated_pct = float(data.get("treated_area_pct", 0))
    reduction = float(data.get("chemical_reduction_pct", 0))
    confidence = float(data.get("confidence", 0))
    spray_area = float(data.get("spray_area_ha", 0))
    spray_volume = float(data.get("spray_volume_l", 0))
    pesticide = data.get("pesticide", "recommended fungicide")

    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")

    if confidence >= 0.85:
        confidence_text = "high confidence"
    elif confidence >= 0.65:
        confidence_text = "moderate confidence"
    else:
        confidence_text = "low confidence"

    if treated_pct < 3:
        severity = "Low severity"
        advisory = "Disease presence is minimal. Continue monitoring."
        warning = "Avoid unnecessary chemical application."
    elif treated_pct < 7:
        severity = "Moderate severity"
        advisory = "Targeted precision spraying is recommended."
        warning = "Limit spraying strictly to affected zones."
    else:
        severity = "High severity"
        advisory = "Precision spraying followed by monitoring is advised."
        warning = "Avoid full-field spraying unless spread increases."

    return f"""
Disease Update: {disease}

Current situation
- Severity level: {severity}
- Disease detected with {confidence_text}
- Affected area: {treated_pct:.1f}% of the field
- Estimated affected area: {spray_area:.3f} hectares

Recommended action
- Apply {pesticide}
- Spray only the affected area
- Estimated spray volume: {spray_volume:.1f} liters

Expected impact
- Precision spraying can reduce chemical usage by approximately {reduction:.1f}%

Advisory
- {advisory}

Important notice
- {warning}

Last updated
- {timestamp}
""".strip()

# ==================================================
# MAIN ENTRY POINT
# ==================================================
def generate_farmer_brief(data: dict) -> str:
    prompt = f"""
        You are an agricultural decision-support assistant writing for a farmer.

        Your task is to summarize crop disease analysis results in a clear,
        neutral, and actionable way. Do NOT assume that any spraying has already
        been done. Base your summary strictly on the provided data.

        Tone:
        - Professional
        - Calm
        - Practical
        - Farmer-friendly

        Formatting rules:
        - Use clear section headings (no markdown symbols like ###)
        - Use bullet points where appropriate
        - Keep sentences short and direct
        - Avoid emojis

        Content rules:
        - Do not introduce new numbers
        - Do not speculate beyond the data
        - Do not exaggerate risk
        - Do not recommend full-field spraying unless stated

        Use the following structure exactly:

        Disease Update: <disease name>

        Current situation
        - Severity level (low / moderate / high)
        - Estimated affected area (% and hectares)
        - Confidence level of detection

        Impact assessment
        - One short paragraph explaining why this level of infection matters
        for crop health and yield if unmanaged

        Recommended action
        - Recommended product
        - Area to spray
        - Estimated spray volume

        Safety and advisory
        - One clear advisory statement encouraging precision application
        and continued monitoring

        Field analysis data:
        Disease: {data['disease']}
        Affected area (%): {data['treated_area_pct']:.1f}
        Affected area (ha): {data['spray_area_ha']:.3f}
        Spray volume (L): {data['spray_volume_l']:.1f}
        Chemical reduction potential (%): {data['chemical_reduction_pct']:.1f}
        Confidence score: {data['confidence']:.2f}
        Recommended pesticide: {data['pesticide']}
        """


    payload = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a professional agronomy advisor."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }

    try:
        response = requests.post(
            GPT_ENDPOINT,
            headers=HEADERS,
            json=payload,
            timeout=20
        )

        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"].strip()

    except Exception:
        # Silent fallback (no error shown to user)
        return rule_based_summary(data)