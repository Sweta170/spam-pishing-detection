"""
app.py
======
Production-ready FastAPI web service for Spam & Phishing Detection.

Features:
- GET  /        : Interactive Web UI for live testing in your browser!
- POST /predict : Real-time text classification (Ham vs. Spam) with confidence scores
                  and LLM explanation for flagged messages.
- GET  /health  : API health status and loaded model info.
- Swagger Docs  : Available at /docs.
"""

import os
import joblib
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from src.model_utils import clean_text
from src.explain import explain_flagged_message

# Initialize FastAPI application
app = FastAPI(
    title="SMS Spam & Phishing Detection API",
    description="A beginner-friendly Machine Learning API for classifying SMS/phishing messages with an LLM explanation layer.",
    version="1.0.0"
)

# Paths to serialized artifacts
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join("models", "production_model.joblib")

# Global variables for in-memory model caching
vectorizer = None
model = None

@app.on_event("startup")
def load_artifacts():
    """Loads the trained TF-IDF vectorizer and model into memory on server start."""
    global vectorizer, model
    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model artifacts not found! Run 'python train.py' first.")
    
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    print("✓ Model artifacts successfully loaded into memory.")


# Request Schema with pre-filled example for Swagger UI
class MessageRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="The SMS or email text to classify."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "Congratulations! You've won a $1000 Walmart gift card. Call 0800-123-456 now."}
            ]
        }
    }

# Response Schema
class PredictionResponse(BaseModel):
    message: str
    prediction: str  # "ham" or "spam"
    is_spam: bool
    confidence: float
    spam_probability: float
    explanation: Optional[str] = None
    model_name: str = "Multinomial Naive Bayes"


@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def web_ui():
    """Serves a clean, beginner-friendly web UI for direct testing in browser."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SMS Spam & Phishing Detector</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            body { background: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }
            .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; width: 100%; max-width: 600px; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
            h1 { font-size: 24px; font-weight: 700; margin-bottom: 8px; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
            p.sub { font-size: 14px; color: #94a3b8; margin-bottom: 24px; }
            label { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #cbd5e1; display: block; margin-bottom: 8px; }
            textarea { width: 100%; height: 110px; background: #0f172a; border: 1px solid #475569; border-radius: 8px; padding: 12px; color: #fff; font-size: 15px; resize: vertical; margin-bottom: 16px; outline: none; }
            textarea:focus { border-color: #38bdf8; }
            .btn { width: 100%; background: #0284c7; color: white; border: none; padding: 12px; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
            .btn:hover { background: #0369a1; }
            .btn:disabled { background: #475569; cursor: not-allowed; }
            .quick-buttons { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
            .quick-btn { background: #334155; color: #cbd5e1; border: none; padding: 6px 12px; border-radius: 6px; font-size: 12px; cursor: pointer; }
            .quick-btn:hover { background: #475569; color: white; }
            #result { margin-top: 24px; display: none; padding: 16px; border-radius: 8px; }
            .spam { background: #451a1a; border: 1px solid #ef4444; color: #fecaca; }
            .ham { background: #064e3b; border: 1px solid #10b981; color: #a7f3d0; }
            .badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-weight: 700; font-size: 13px; margin-bottom: 8px; }
            .badge-spam { background: #ef4444; color: white; }
            .badge-ham { background: #10b981; color: white; }
            .meta { font-size: 13px; opacity: 0.9; margin-top: 4px; }
            .explanation { margin-top: 10px; padding-top: 10px; border-top: 1px dashed rgba(255,255,255,0.2); font-size: 13px; }
            .footer-links { margin-top: 24px; text-align: center; font-size: 12px; color: #64748b; }
            .footer-links a { color: #38bdf8; text-decoration: none; margin: 0 8px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🛡️ SMS Spam & Phishing Detector</h1>
            <p class="sub">Production ML Classifier (Multinomial Naive Bayes) with LLM Explanation Layer</p>
            
            <label>Test Quick Samples:</label>
            <div class="quick-buttons">
                <button class="quick-btn" onclick="setMsg('hello , how are you??')">Normal Greeting</button>
                <button class="quick-btn" onclick="setMsg('Hey, are we still meeting for lunch at 12:30?')">Lunch Plan</button>
                <button class="quick-btn" onclick="setMsg('URGENT: Your bank account is suspended! Call 0800-999-111 now.')">Bank Phishing</button>
                <button class="quick-btn" onclick="setMsg('Congratulations! You won a $1000 Walmart gift card! Text WIN to 88088.')">Prize Scam</button>
            </div>

            <label for="msgInput">Enter SMS or Phishing Text:</label>
            <textarea id="msgInput" placeholder="Type or paste any SMS message here..."></textarea>
            
            <button class="btn" id="submitBtn" onclick="classifyText()">Detect Spam</button>

            <div id="result"></div>

            <div class="footer-links">
                <a href="/docs" target="_blank">Swagger API Docs</a> • 
                <a href="/health" target="_blank">Health Check</a>
            </div>
        </div>

        <script>
            function setMsg(text) {
                document.getElementById('msgInput').value = text;
            }

            async function classifyText() {
                const input = document.getElementById('msgInput').value.trim();
                const resDiv = document.getElementById('result');
                const btn = document.getElementById('submitBtn');

                if (!input) {
                    alert('Please enter a message first!');
                    return;
                }

                btn.disabled = true;
                btn.innerText = 'Analyzing...';
                resDiv.style.display = 'none';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: input })
                    });
                    
                    const data = await response.json();
                    btn.disabled = false;
                    btn.innerText = 'Detect Spam';
                    resDiv.style.display = 'block';

                    if (data.is_spam) {
                        resDiv.className = 'spam';
                        resDiv.innerHTML = `
                            <span class="badge badge-spam">🚨 SPAM / PHISHING DETECTED</span>
                            <div class="meta"><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}% | <strong>Model:</strong> ${data.model_name}</div>
                            ${data.explanation ? `<div class="explanation">💡 <strong>Red Flag Analysis:</strong> ${data.explanation}</div>` : ''}
                        `;
                    } else {
                        resDiv.className = 'ham';
                        resDiv.innerHTML = `
                            <span class="badge badge-ham">✅ LEGITIMATE MESSAGE (HAM)</span>
                            <div class="meta"><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}% | <strong>Model:</strong> ${data.model_name}</div>
                            <div class="meta">Spam Probability: ${(data.spam_probability * 100).toFixed(2)}%</div>
                        `;
                    }
                } catch (err) {
                    btn.disabled = false;
                    btn.innerText = 'Detect Spam';
                    resDiv.style.display = 'block';
                    resDiv.className = 'spam';
                    resDiv.innerHTML = `<strong>Error:</strong> ${err.message}`;
                }
            }
        </script>
    </body>
    </html>
    """


@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint and API metadata."""
    return {
        "status": "healthy",
        "service": "SMS Spam & Phishing Detection API",
        "model_loaded": model is not None,
        "docs_url": "/docs"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_spam(payload: MessageRequest):
    """
    Classifies an input SMS/text as either legitimate ('ham') or 'spam'/'phishing'.
    If flagged as spam, delegates to the LLM explanation layer for a 1-line reason.
    """
    if vectorizer is None or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready or failed to load."
        )

    raw_text = payload.message.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Input message cannot be empty or only spaces.")

    # 1. Clean the incoming text using the same cleaning logic as training
    cleaned_text = clean_text(raw_text)

    # 2. Vectorize using the pre-fitted TF-IDF vocabulary
    features = vectorizer.transform([cleaned_text])

    # 3. Model prediction (0 = ham, 1 = spam)
    pred_int = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    is_spam = bool(pred_int == 1)
    label = "spam" if is_spam else "ham"
    confidence = float(probabilities[pred_int])
    spam_prob = float(probabilities[1])

    # 4. LLM Explanation Layer (ONLY triggered if flagged as spam!)
    explanation = None
    if is_spam:
        explanation = explain_flagged_message(raw_text)

    return PredictionResponse(
        message=raw_text,
        prediction=label,
        is_spam=is_spam,
        confidence=round(confidence, 4),
        spam_probability=round(spam_prob, 4),
        explanation=explanation,
        model_name="Multinomial Naive Bayes (Zero False Positives)"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
