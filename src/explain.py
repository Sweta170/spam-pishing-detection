"""
explain.py
==========
LLM Explanation Layer for Flagged Messages.

IMPORTANT ARCHITECTURAL PRINCIPLE:
The LLM ONLY explains WHY a message is suspicious.
The ML model (Naive Bayes / Logistic Regression) makes the decision (Spam or Ham).
The LLM NEVER classifies or overrides the decision. This keeps latency low,
costs predictable, and guarantees explainability without hallucinated decisions.

If no API key is set in the environment, the function skips gracefully without failing.
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load variables from .env file if available
load_dotenv()

def explain_flagged_message(message: str) -> str:
    """
    Takes an SMS/message already flagged as spam/phishing by the ML model
    and calls an LLM API to generate a concise, one-line explanation of the red flag.
    
    Reads API keys from environment variables:
    - GEMINI_API_KEY (Google Gemini API)
    - OPENAI_API_KEY (OpenAI API)
    
    Returns a one-line explanation string, or a graceful fallback message if skipped.
    """
    # 1. Check for API keys in environment variables (NEVER hardcode keys!)
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not gemini_key and not openai_key:
        return (
            "[Explanation skipped: No API key configured. Set GEMINI_API_KEY or "
            "OPENAI_API_KEY environment variable to enable live AI reasoning.]"
        )

    system_instruction = (
        "You are a cybersecurity analyst assistant. The following SMS message was ALREADY "
        "classified as spam/phishing by our machine learning model. "
        "In exactly ONE concise sentence (under 25 words), explain the primary red flag "
        "(e.g., artificial urgency, prize bait, unverified toll-free number, or credential request). "
        "Do NOT decide whether it is spam or not. Only explain the red flag."
    )

    prompt = f"{system_instruction}\n\nMessage to explain: \"{message}\""

    # Option A: Google Gemini API (if GEMINI_API_KEY is available)
    if gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 60,
                "temperature": 0.2
            }
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=8)
            if response.status_code == 200:
                data = response.json()
                explanation = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return explanation
            else:
                return f"[Gemini API returned status {response.status_code}: {response.text[:80]}]"
        except Exception as e:
            return f"[Explanation request failed: {str(e)}]"

    # Option B: OpenAI API (if OPENAI_API_KEY is available)
    if openai_key:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Message: \"{message}\""}
            ],
            "max_tokens": 60,
            "temperature": 0.2
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=8)
            if response.status_code == 200:
                data = response.json()
                explanation = data["choices"][0]["message"]["content"].strip()
                return explanation
            else:
                return f"[OpenAI API returned status {response.status_code}: {response.text[:80]}]"
        except Exception as e:
            return f"[Explanation request failed: {str(e)}]"

    return "[Explanation skipped: No valid LLM provider responded.]"
