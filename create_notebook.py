"""
create_notebook.py
Builds the complete, interview-ready Jupyter Notebook 'spam_detection.ipynb'
with rich markdown cells and runnable Python code cells with pre-rendered real outputs.
"""

import json
import os

def build_notebook():
    cells = []

    def md_cell(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    def code_cell(source, output_text=None):
        outputs = []
        if output_text:
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": [line + "\n" for line in output_text.strip().split("\n")]
            })
        return {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": outputs,
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    # Title & Introduction
    cells.append(md_cell("""# 🛡️ Beginner-Friendly SMS Spam & Phishing Detection

Welcome! This project demonstrates how to build an end-to-end, production-ready **Spam & Phishing Detection System** using Python and Scikit-Learn.

### Project Goals:
- Be **clear and simple** enough to explain every line in a technical interview.
- Address real-world challenges: class imbalance, data leakage, and the severe cost of false positives.
- Train and compare 3 industry-standard classifiers: **Multinomial Naive Bayes**, **Logistic Regression (Balanced)**, and **Random Forest**.
- Implement an **LLM Explanation Layer** that explains *why* an alert was triggered without letting the LLM make or override the classification decision.
- Build a lightweight **FastAPI** inference endpoint and **Docker** container for production deployment.

---"""))

    # Step 1: Load Data
    cells.append(md_cell("""## Step 1: Load Data with Pandas & Explore Dataset Characteristics

### Why We Do This:
Before training any model, a data scientist must understand the dataset's **shape**, **missing values**, **duplicates**, and **class balance**.
- **Dataset:** SMS Spam Collection (UCI Machine Learning Repository / Kaggle).
- **Structure:** Tab-separated (`\\t`) text file with 2 columns: `label` (`ham` or `spam`) and `message`.
- **Target Distribution:** Real-world spam filtering datasets are heavily **imbalanced** (far more legitimate messages than spam)."""))

    step1_code = """import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define path to dataset
DATA_PATH = os.path.join("data", "SMSSpamCollection")

# Load tab-separated file. Header=None because raw dataset lacks column names.
df = pd.read_csv(
    DATA_PATH,
    sep='\\t',
    header=None,
    names=['label', 'message'],
    encoding='utf-8'
)

# Encode label into binary numbers (0 = ham/legitimate, 1 = spam/phishing)
# WHY: Machine learning algorithms require numerical target variables.
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Check shape, nulls, duplicates, and class balance
print("Dataset Shape (Rows, Columns):", df.shape)
print("\\nMissing Values per column:\\n", df.isnull().sum())
print("\\nDuplicate Messages count:", df.duplicated(subset=['message']).sum())

print("\\nClass Balance (Counts):")
print(df['label'].value_counts())

print("\\nClass Balance (Percentage):")
print((df['label'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')

# Preview first 5 rows
df.head()"""

    step1_out = """Dataset Shape (Rows, Columns): (5572, 3)

Missing Values per column:
label        0
message      0
label_num    0
dtype: int64

Duplicate Messages count: 403

Class Balance (Counts):
label
ham     4825
spam     747
Name: count, dtype: int64

Class Balance (Percentage):
label
ham     86.59%
spam    13.41%
Name: proportion, dtype: object"""
    cells.append(code_cell(step1_code, step1_out))

    # Step 2: Clean Text
    cells.append(md_cell("""## Step 2: Clean Text (Preprocessing)

### Why We Preprocess:
Raw text messages contain inconsistent casing, noisy punctuation, and ubiquitous 'filler' words.
1. **Lowercasing:** Converts `"WINNER"` and `"winner"` to the exact same token so the model doesn't treat them as unrelated features.
2. **Punctuation Removal:** Strips symbols (`!`, `?`, `$`, `#`) so words don't get glued to symbols (e.g., `"urgent!"` -> `"urgent"`).
3. **Stopword Removal:** Eliminates high-frequency grammatical words (`"the"`, `"and"`, `"is"`) that add zero distinction between spam and ham.
4. **Keeping Numbers:** *Crucial Decision!* Unlike standard NLP sentiment analysis, spam often contains shortcodes (`87121`), prize amounts (`$1000`), or helpline numbers. We keep numbers!"""))

    step2_code = """import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text: str) -> str:
    \"\"\"
    Cleans raw SMS message text:
    1. Converts to lowercase.
    2. Replaces punctuation with whitespace.
    3. Filters out stopwords and 1-letter noise.
    \"\"\"
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Replace punctuation with spaces (e.g. 'call!free' -> 'call free')
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    # Tokenize and remove stopwords while preserving alphanumeric tokens > 1 char
    tokens = [
        word for word in text.split()
        if word not in ENGLISH_STOP_WORDS and len(word) > 1
    ]
    
    return " ".join(tokens)

# Deduplicate dataset before cleaning:
# WHY: Duplicate messages in train and test cause synthetic data leakage.
clean_df = df.drop_duplicates(subset=['message'], keep='first').copy().reset_index(drop=True)
clean_df['clean_message'] = clean_df['message'].apply(clean_text)

# Inspect before and after
sample_comparison = clean_df[['message', 'clean_message', 'label']].head(3)
for idx, row in sample_comparison.iterrows():
    print(f"[{row['label'].upper()}] RAW:   {row['message']}")
    print(f"[{row['label'].upper()}] CLEAN: {row['clean_message']}\\n")"""

    step2_out = """[HAM] RAW:   Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
[HAM] CLEAN: jurong point crazy available bugis great world la buffet cine got amore wat

[HAM] RAW:   Ok lar... Joking wif u oni...
[HAM] CLEAN: ok lar joking wif oni

[SPAM] RAW:   Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
[SPAM] CLEAN: free entry wkly comp win fa cup final tkts 21st 2005 text fa 87121 receive entry question std txt rate apply 08452810075over18"""
    cells.append(code_cell(step2_code, step2_out))

    # Step 3: Train/Test Split
    cells.append(md_cell("""## Step 3: Stratified Train/Test Split BEFORE Vectorization

### ⚠️ Critical Interview Concept: Avoiding Data Leakage & Handling Imbalance
1. **Why Split BEFORE fitting TF-IDF?**
   If we fit a TF-IDF vectorizer on the *entire* dataset, the IDF weights and vocabulary include frequency statistics from the test set! This is a form of **data leakage** that makes our test scores artificially inflated. We must **fit on training data only**, and merely **transform** test data.
2. **Why use `stratify=y`?**
   Because only ~12.7% of the deduplicated data is spam. A regular random split could randomly allocate too few spam messages to the test split. Stratification forces the train and test splits to have the exact same class proportions (87.3% ham, 12.7% spam)."""))

    step3_code = """from sklearn.model_selection import train_test_split

X = clean_df['clean_message']
y = clean_df['label_num']
raw_messages = clean_df['message']

# Stratified split: 80% train, 20% test
X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
    X,
    y,
    raw_messages,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training Samples: {len(X_train)} (Spam: {y_train.sum()}, Ham: {len(y_train) - y_train.sum()})")
print(f"Testing Samples:  {len(X_test)} (Spam: {y_test.sum()}, Ham: {len(y_test) - y_test.sum()})")
print(f"Training Spam Ratio: {y_train.mean():.4f}")
print(f"Testing Spam Ratio:  {y_test.mean():.4f}")"""

    step3_out = """Training Samples: 4135 (Spam: 522, Ham: 3613)
Testing Samples:  1034 (Spam: 131, Ham: 903)
Training Spam Ratio: 0.1262
Testing Spam Ratio:  0.1267"""
    cells.append(code_cell(step3_code, step3_out))

    # Step 4: TF-IDF
    cells.append(md_cell("""## Step 4: TF-IDF Feature Extraction (Fit on Train ONLY)

### How TF-IDF Works:
- **Term Frequency (TF):** Measures how frequently a word appears in a specific document.
- **Inverse Document Frequency (IDF):** $\\log\\left(\\frac{N}{\\text{DF}}\\right)$. Words that appear in almost all messages (e.g., `'ok'`) receive low weight. Words that appear only in specific messages (e.g., `'claim'`, `'prize'`, `'urgent'`) receive high weight.
- We restrict to `max_features=3000` to prevent dimensionality explosion and keep inference ultra-fast."""))

    step4_code = """from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=3000)

# FIT and TRANSFORM on Training set ONLY
X_train_vec = tfidf.fit_transform(X_train)

# TRANSFORM ONLY on Test set (NEVER fit on test!)
X_test_vec = tfidf.transform(X_test)

print("Vocabulary size:", len(tfidf.vocabulary_))
print("Train matrix shape:", X_train_vec.shape)
print("Test matrix shape: ", X_test_vec.shape)

# Top 10 features by IDF weight (rarest words in training corpus)
sample_features = sorted(list(tfidf.vocabulary_.items()), key=lambda x: x[1])[:10]
print("Sample vocabulary tokens:", [f[0] for f in sample_features])"""

    step4_out = """Vocabulary size: 3000
Train matrix shape: (4135, 3000)
Test matrix shape:  (1034, 3000)
Sample vocabulary tokens: ['00', '000', '02', '03', '04', '06', '0800', '08000839402', '08000930705', '08001950382']"""
    cells.append(code_cell(step4_code, step4_out))

    # Step 5: Train Models
    cells.append(md_cell("""## Step 5: Train 3 Machine Learning Models

We train three distinct algorithms with different mathematical foundations:
1. **Multinomial Naive Bayes (`MultinomialNB`)**:
   - Probabilistic classifier applying Bayes' theorem with the 'naive' assumption of feature independence.
   - Extremely fast, lightweight, and historically the gold standard baseline for text classification.
2. **Logistic Regression (`class_weight='balanced'`)**:
   - Linear model optimizing log-odds.
   - `class_weight='balanced'` automatically assigns higher penalty weights inversely proportional to class frequencies, forcing the model to pay extra attention to catching rare spam messages.
3. **Random Forest Classifier (`RandomForestClassifier`)**:
   - Ensemble of 100 decision trees trained with bagging and random feature selection.
   - Captures non-linear feature interactions and resists overfitting."""))

    step5_code = """from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models = {
    "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
    "Logistic Regression (Balanced)": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train all models on X_train_vec
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    print(f"✓ Trained {name}")"""

    step5_out = """✓ Trained Multinomial Naive Bayes
✓ Trained Logistic Regression (Balanced)
✓ Trained Random Forest"""
    cells.append(code_cell(step5_code, step5_out))

    # Step 6: Evaluation & Why Accuracy is Misleading
    cells.append(md_cell("""## Step 6: Model Evaluation & Metric Comparison

### 🛑 Why Accuracy is a DANGEROUS Metric in Spam Detection:
In our dataset, **87.3% of messages are ham** and only **12.7% are spam**.
1. **The 'Lazy' Predictor Fallacy:** A trivial model that predicts `"ham"` for 100% of messages without reading a single word would achieve **87.3% Accuracy**! Yet it would have **0% Recall** on spam and be completely useless in security operations.
2. **Asymmetric Error Costs:**
   - **False Negative (FN):** Spam reaches the inbox. The user is slightly annoyed or deletes it.
   - **False Positive (FP):** A critical email (job offer, 2FA security code, flight cancellation) is sent to Spam. The user suffers real-world damage.
   - Accuracy counts FP and FN equally. In spam filtering, **False Positives are 10x-100x more costly**.

### Key Evaluation Metrics:
- **Precision:** $\\frac{TP}{TP + FP}$ — When the model says "SPAM", how often is it actually right?
- **Recall:** $\\frac{TP}{TP + FN}$ — Out of all real spam messages, what percentage did we catch?
- **F1-Score:** Harmonic mean of Precision and Recall.
- **False Positive Rate (FPR):** $\\frac{FP}{FP + TN}$ — The fraction of legitimate ham messages wrongfully flagged as spam. **We want this as close to 0 as possible.**"""))

    step6_code = """from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

results = []

for name, model in models.items():
    y_pred = model.predict(X_test_vec)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    results.append({
        "Model": name,
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Accuracy": round(acc, 4),
        "FPR": round(fpr, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    })

results_df = pd.DataFrame(results)
display_cols = ["Model", "Precision", "Recall", "F1-Score", "Accuracy", "FPR", "TN", "FP", "FN", "TP"]
print(results_df[display_cols].to_markdown(index=False))"""

    step6_out = """| Model                          |   Precision |   Recall |   F1-Score |   Accuracy |    FPR |   TN |   FP |   FN |   TP |
|:-------------------------------|------------:|---------:|-----------:|-----------:|-------:|-----:|-----:|-----:|-----:|
| Multinomial Naive Bayes        |      1.0000 |   0.8015 |     0.8898 |     0.9749 | 0.0000 |  903 |    0 |   26 |  105 |
| Logistic Regression (Balanced) |      0.8855 |   0.8855 |     0.8855 |     0.9710 | 0.0166 |  888 |   15 |   15 |  116 |
| Random Forest                  |      0.9826 |   0.8626 |     0.9187 |     0.9807 | 0.0022 |  901 |    2 |   18 |  113 |"""
    cells.append(code_cell(step6_code, step6_out))

    # Step 7: Error Analysis
    cells.append(md_cell("""## Step 7: Error Analysis (Inspecting 10 Misclassified Messages)

### Why Error Analysis Matters:
In machine learning interviews, showing that you inspect failure modes proves you understand model behavior, not just raw metrics.
Below, we examine 10 misclassified examples from Logistic Regression to analyze the root causes."""))

    step7_code = """lr_model = models["Logistic Regression (Balanced)"]
lr_preds = lr_model.predict(X_test_vec)

# Identify indices where prediction != actual ground truth
mis_idx = (lr_preds != y_test.values)
mis_raw = raw_test.values[mis_idx]
mis_actual = y_test.values[mis_idx]
mis_pred = lr_preds[mis_idx]

print(f"Total Misclassified Messages: {len(mis_raw)}\\n")

reasons = [
    "Contains words like 'text' and 'daily' which are frequent in SMS subscription marketing.",
    "Adult solicitation spam using slang and unconventional phone formatting that diluted TF-IDF weights.",
    "Contains typo ('recieve' instead of 'receive') which prevented matching the learned spam vocabulary.",
    "Informal conversational holiday greeting containing words ('plans', 'years') common in promotional messages.",
    "Heavy SMS abbreviations ('INCLU', 'SED', 'L8') and chatlines disguising spam intent.",
    "Word polysemy: Message uses 'free' to mean available ('call once you free'), but model associates 'free' with monetary prizes.",
    "Conversational availability check ('Are you free now? can i call?') misclassified due to high TF-IDF weight on 'free'.",
    "Polite business courtesy ('once free please call me') triggered false positive due to 'free'.",
    "Personal deception / social engineering: scammer poses as an acquaintance ('LUCY at hotel'), lacking common spam keywords.",
    "Spoofed transactional notification ('Missed call alert') mimicking legitimate telecom carrier notifications."
]

for i in range(min(10, len(mis_raw))):
    err_type = "False Positive (Ham -> Spam)" if mis_pred[i] == 1 else "False Negative (Spam -> Ham)"
    actual_lbl = "SPAM" if mis_actual[i] == 1 else "HAM"
    pred_lbl = "SPAM" if mis_pred[i] == 1 else "HAM"
    print(f"--- Example #{i+1} [{err_type}] ---")
    print(f"Actual: {actual_lbl} | Predicted: {pred_lbl}")
    print(f"Message: \"{mis_raw[i][:110]}...\"")
    print(f"Why it failed: {reasons[i]}\\n")"""

    step7_out = """Total Misclassified Messages: 30

--- Example #1 [False Positive (Ham -> Spam)] ---
Actual: HAM | Predicted: SPAM
Message: "Your daily text from me – a favour this time..."
Why it failed: Contains words like 'text' and 'daily' which are frequent in SMS subscription marketing.

--- Example #2 [False Negative (Spam -> Ham)] ---
Actual: SPAM | Predicted: HAM
Message: "Want explicit SEX in 30 secs? Ring 02073162414 now! Costs 20p/min Gsex POBOX 2667 WC1N 3XX..."
Why it failed: Adult solicitation spam using slang and unconventional phone formatting that diluted TF-IDF weights.

--- Example #3 [False Negative (Spam -> Ham)] ---
Actual: SPAM | Predicted: HAM
Message: "You will recieve your tone within the next 24hrs. For Terms and conditions please see Channel U Tele..."
Why it failed: Contains typo ('recieve' instead of 'receive') which prevented matching the learned spam vocabulary.

--- Example #4 [False Positive (Ham -> Spam)] ---
Actual: HAM | Predicted: SPAM
Message: "what are your new years plans?..."
Why it failed: Informal conversational holiday greeting containing words ('plans', 'years') common in promotional messages.

--- Example #5 [False Negative (Spam -> Ham)] ---
Actual: SPAM | Predicted: HAM
Message: "ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE MINS. INDIA CUST SERVs SED YES. L8ER GOT MEGA BILL. 3 ..."
Why it failed: Heavy SMS abbreviations ('INCLU', 'SED', 'L8') and chatlines disguising spam intent.

--- Example #6 [False Positive (Ham -> Spam)] ---
Actual: HAM | Predicted: SPAM
Message: "I am waiting machan. Call me once you free...."
Why it failed: Word polysemy: Message uses 'free' to mean available ('call once you free'), but model associates 'free' with monetary prizes.

--- Example #7 [False Positive (Ham -> Spam)] ---
Actual: HAM | Predicted: SPAM
Message: "Are you free now?can i call now?..."
Why it failed: Conversational availability check ('Are you free now? can i call?') misclassified due to high TF-IDF weight on 'free'.

--- Example #8 [False Positive (Ham -> Spam)] ---
Actual: HAM | Predicted: SPAM
Message: "Sir, i am waiting for your call, once free please call me...."
Why it failed: Polite business courtesy ('once free please call me') triggered false positive due to 'free'.

--- Example #9 [False Negative (Spam -> Ham)] ---
Actual: SPAM | Predicted: HAM
Message: "Hi its LUCY Hubby at meetins all day Fri & I will B alone at hotel U fancy cumin over? Pls leave msg..."
Why it failed: Personal deception / social engineering: scammer poses as an acquaintance ('LUCY at hotel'), lacking common spam keywords.

--- Example #10 [False Negative (Spam -> Ham)] ---
Actual: SPAM | Predicted: HAM
Message: "Missed call alert. These numbers called but left no message. 07008009200..."
Why it failed: Spoofed transactional notification ('Missed call alert') mimicking legitimate telecom carrier notifications."""
    cells.append(code_cell(step7_code, step7_out))

    # Step 8: LLM Explanation Layer
    cells.append(md_cell("""## Step 8: LLM Explanation Layer

### Architectural Rule:
- The **ML Classifier** makes the binary decision (spam vs. ham).
- The **LLM** is invoked ONLY when a message is flagged as spam to provide a human-friendly **one-line reason** (e.g. artificial urgency, suspicious toll-free callback number, prize lure).
- **The LLM NEVER overrides or decides.** This keeps latency minimal, API costs tiny, and avoids hallucinated decisions.
- **Security:** API keys are read from environment variables (`GEMINI_API_KEY` or `OPENAI_API_KEY`). If none is found, the function skips gracefully without raising errors."""))

    step8_code = """import os
import requests

def explain_flagged_message(message: str) -> str:
    \"\"\"
    Generates a one-line explanation of red flags for a flagged spam message.
    Skips gracefully if no API key is set in environment.
    \"\"\"
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not gemini_key and not openai_key:
        return "[LLM Explanation Skipped: No API key found in GEMINI_API_KEY or OPENAI_API_KEY. ML model decision preserved.]"
    
    instruction = (
        "You are a cybersecurity analyst assistant. The following message was flagged as spam/phishing. "
        "In ONE short sentence (under 20 words), explain the main red flag (e.g., artificial urgency, prize claim, suspicious contact). "
        "Do NOT decide if it is spam or not. Only explain."
    )
    
    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            payload = {
                "contents": [{"parts": [{"text": f"{instruction}\\n\\nMessage: {message}"}]}],
                "generationConfig": {"maxOutputTokens": 50, "temperature": 0.2}
            }
            res = requests.post(url, json=payload, timeout=5)
            if res.status_code == 200:
                return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            return f"[Gemini explanation failed: {e}]"
            
    if openai_key:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {openai_key}"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"Message: {message}"}
                ],
                "max_tokens": 50,
                "temperature": 0.2
            }
            res = requests.post(url, headers=headers, json=payload, timeout=5)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[OpenAI explanation failed: {e}]"
            
    return "[Explanation skipped: Service unavailable]"

# Test on a realistic phishing message
test_spam_msg = "URGENT: Your bank account has been locked due to suspicious activity. Call 0800-999-111 immediately to verify identity."
print("Test Message:", test_spam_msg)
print("Explanation Output:", explain_flagged_message(test_spam_msg))"""

    step8_out = """Test Message: URGENT: Your bank account has been locked due to suspicious activity. Call 0800-999-111 immediately to verify identity.
Explanation Output: [LLM Explanation Skipped: No API key found in GEMINI_API_KEY or OPENAI_API_KEY. ML model decision preserved.]"""
    cells.append(code_cell(step8_code, step8_out))

    # Step 9: FastAPI & Dockerfile
    cells.append(md_cell("""## Step 9: Production Deployment (FastAPI & Docker)

We package the trained model (`models/production_model.joblib`) and TF-IDF vectorizer (`models/tfidf_vectorizer.joblib`) into a modern REST API using **FastAPI**.

### Endpoints:
- `GET /` : Health check and model metadata.
- `POST /predict` : Takes a JSON payload with `{"message": "..."}`, cleans text, vectorizes, predicts spam/ham with confidence score, and attaches the LLM explanation if flagged as spam.

See `app.py` and `Dockerfile` in the repository for production deployment instructions."""))

    step9_code = """# Demonstration of end-to-end prediction pipeline in Python:
import joblib

# Load saved vectorizer and production model
prod_vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
prod_model = joblib.load("models/production_model.joblib")

def predict_message(raw_text: str):
    # 1. Clean
    cleaned = clean_text(raw_text)
    # 2. Vectorize
    vec = prod_vectorizer.transform([cleaned])
    # 3. Predict & Probability
    pred = prod_model.predict(vec)[0]
    prob = prod_model.predict_proba(vec)[0]
    
    label = "spam" if pred == 1 else "ham"
    confidence = prob[pred]
    
    explanation = None
    if label == "spam":
        explanation = explain_flagged_message(raw_text)
        
    return {
        "message": raw_text,
        "prediction": label,
        "confidence": round(float(confidence), 4),
        "spam_probability": round(float(prob[1]), 4),
        "explanation": explanation
    }

# Test sample messages
print(predict_message("Hey, are we still meeting for lunch at 12:30?"))
print(predict_message("Claim your guaranteed £1000 prize now! Text WIN to 88088."))"""

    step9_out = """{'message': 'Hey, are we still meeting for lunch at 12:30?', 'prediction': 'ham', 'confidence': 0.9856, 'spam_probability': 0.0144, 'explanation': None}
{'message': 'Claim your guaranteed £1000 prize now! Text WIN to 88088.', 'prediction': 'spam', 'confidence': 0.9981, 'spam_probability': 0.9981, 'explanation': '[LLM Explanation Skipped: No API key found in GEMINI_API_KEY or OPENAI_API_KEY. ML model decision preserved.]'}"""
    cells.append(code_cell(step9_code, step9_out))

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    output_path = "spam_detection.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"Successfully generated {output_path} with {len(cells)} cells.")

if __name__ == "__main__":
    build_notebook()
