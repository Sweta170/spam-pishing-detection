"""
demo.py
Interactive demo script to test custom SMS messages against the trained model.
"""

import os
import joblib
from src.model_utils import clean_text
from src.explain import explain_flagged_message

def run_demo():
    print("=" * 70)
    print("   LIVE INFERENCE DEMO - SMS SPAM & PHISHING DETECTION")
    print("=" * 70)

    # Load artifacts
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.joblib")
    model_path = os.path.join("models", "production_model.joblib")

    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        print("[ERROR] Model artifacts not found. Please run train.py first.")
        return

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    print("✓ Successfully loaded TF-IDF vectorizer and Multinomial Naive Bayes model.\n")

    # Sample messages representing different scenarios
    test_messages = [
        "Hey, are we still meeting for lunch at 12:30?",
        "Can you send me the lecture notes from this morning's class?",
        "URGENT: Your bank account has been compromised. Call 0800-999-111 immediately to verify identity.",
        "Congratulations! You won a $1000 Walmart gift card! Text WIN to 88088 now to claim.",
        "Your Amazon parcel delivery failed. Click http://suspicious-track.link/update to reschedule.",
        "Are you free tomorrow evening for a quick catch up call?"
    ]

    for idx, msg in enumerate(test_messages, 1):
        # 1. Clean
        cleaned = clean_text(msg)
        # 2. Vectorize
        vec = vectorizer.transform([cleaned])
        # 3. Predict & Probabilities
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]

        is_spam = bool(pred == 1)
        label = "🚨 SPAM / PHISHING" if is_spam else "✅ LEGITIMATE (HAM)"
        confidence = probs[pred] * 100
        spam_prob = probs[1] * 100

        print(f"Message #{idx}:")
        print(f"  Raw Text: \"{msg}\"")
        print(f"  Cleaned:  \"{cleaned}\"")
        print(f"  Verdict:  {label}")
        print(f"  Confidence: {confidence:.2f}% (Spam Probability: {spam_prob:.2f}%)")

        if is_spam:
            reason = explain_flagged_message(msg)
            print(f"  LLM Explanation: {reason}")
        print("-" * 70)

if __name__ == "__main__":
    run_demo()
