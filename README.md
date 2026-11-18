# 🛡️ SMS Spam & Phishing Detection System

An end-to-end, beginner-friendly Machine Learning project for detecting spam and phishing messages in Python. Designed for students preparing for technical interviews, this repository provides simple, modular code where every line and architectural decision can be clearly explained.

---

## 📌 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Dataset Overview](#-dataset-overview)
3. [End-to-End Approach & Architecture](#-end-to-end-approach--architecture)
4. [Real Evaluation Metrics & Comparison](#-real-evaluation-metrics--comparison)
5. [Why Not Rely on Accuracy?](#-why-not-rely-on-accuracy)
6. [Error Analysis (10 Failure Modes)](#-error-analysis-10-failure-modes)
7. [LLM Explanation Layer](#-llm-explanation-layer)
8. [How to Run](#-how-to-run)
9. [Project Limitations](#-project-limitations)
10. [Future Improvements](#-future-improvements)

---

## 🎯 Problem Statement
Unsolicited spam and phishing messages pose serious financial and security risks to mobile users. The primary engineering challenges are:
1. **Severe Class Imbalance:** Legitimate messages far outnumber spam messages (~87% ham vs. ~13% spam).
2. **Asymmetric Error Costs:** Marking a legitimate message as spam (**False Positive**) can cause users to miss critical security codes, flight alerts, or job offers. Letting spam into the inbox (**False Negative**) is merely annoying. Therefore, models must minimize the **False Positive Rate (FPR)**.
3. **Data Leakage Risk:** Feature extraction (e.g., TF-IDF) must be strictly isolated to training data to ensure unbiased real-world generalization.

---

## 📊 Dataset Overview
- **Dataset:** SMS Spam Collection (UCI Machine Learning Repository / Kaggle).
- **Format:** Tab-separated values (`\t`), 2 columns: `label` (`ham` or `spam`) and `message`.
- **Total Records:** 5,572 raw messages.
  - **Ham (Legitimate):** 4,825 (86.59%)
  - **Spam / Phishing:** 747 (13.41%)
- **Deduplication:** 403 duplicate messages exist (common greetings like *"Sorry, I'll call later"* and repeated spam blast templates). To prevent identical messages appearing across train and test sets, we deduplicate to **5,169 unique records** (4,516 ham, 653 spam).
- **Automated Downloader:** `download_data.py` automatically downloads and extracts the official UCI dataset into `data/SMSSpamCollection`.
  - *Manual Link:* [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) or [Kaggle SMS Spam Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

---

## 🏗️ End-to-End Approach & Architecture

```
Raw Text Message
      │
      ▼
1. Preprocessing (Lowercase, Remove Punctuation, Remove Stopwords, Keep Numbers)
      │
      ▼
2. Stratified Train/Test Split (80% Train, 20% Test, stratify=y)
      │
      ▼
3. TF-IDF Vectorization (fit on Train ONLY, transform Test)
      │
      ▼
4. Model Training & Comparison:
   ├── Multinomial Naive Bayes (High Precision, 0 FPR)
   ├── Logistic Regression (class_weight="balanced")
   └── Random Forest Classifier (100 Trees)
      │
      ▼
5. Decision Rule:
   ├── If HAM  ──> Deliver directly to Inbox
   └── If SPAM ──> Route to Spam + Invoke LLM Explanation Layer (1-line red flag)
```

### Why These Design Choices?
- **Lowercasing & Punctuation Stripping:** Ensures `"URGENT!"` and `"urgent"` are mapped to the same token.
- **Keeping Numbers:** Preserves digits (e.g., `87121`, `1000`, `0800`), which are critical indicators of shortcodes, prize lures, and toll-free spam lines.
- **Stratified Split Before Vectorization:** Prevents **data leakage**. If TF-IDF were fitted on the whole corpus, document frequency statistics from the test set would artificially boost evaluation scores.
- **TF-IDF over CountVectorizer:** Down-weights common words that appear across many messages and boosts discriminative keywords.

---

## 📈 Real Evaluation Metrics & Comparison

Evaluated on the held-out test split of **1,034 unseen messages** (903 Ham, 131 Spam):

| Model | Precision | Recall | F1-Score | Accuracy | FPR (False Positive Rate) | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **1.0000** | 0.8015 | 0.8898 | 0.9749 | **0.0000** | **903** | **0** | 26 | 105 |
| **Logistic Regression (Balanced)** | 0.8855 | **0.8855** | 0.8855 | 0.9710 | 0.0166 | 888 | 15 | **15** | **116** |
| **Random Forest** | 0.9826 | 0.8626 | **0.9187** | **0.9807** | 0.0022 | 901 | 2 | 18 | 113 |

### Metric Formulas:
- **Precision:** $\frac{TP}{TP + FP}$ — Out of all messages flagged as spam, what percentage was actually spam?
- **Recall:** $\frac{TP}{TP + FN}$ — Out of all true spam messages, what fraction was caught?
- **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ — Harmonic mean of Precision and Recall.
- **False Positive Rate (FPR):** $\frac{FP}{FP + TN}$ — Fraction of legitimate messages mistakenly filtered to spam.

### Key Production Trade-offs:
- **Multinomial Naive Bayes** achieved **0 False Positives (FPR = 0.0000)** and **100% Precision**. In production email/SMS systems, zero false positives ensures users never lose an important message. We select this as our default production model.
- **Logistic Regression (Balanced)** caught the most spam messages (Recall = **88.55%**), but at the cost of 15 false positives.
- **Random Forest** achieved the highest overall **F1-Score (0.9187)** with only 2 false positives.

---

## 🛑 Why Not Rely on Accuracy?

In binary classification with imbalanced data, **Accuracy is deeply misleading**:
1. **The "Do-Nothing" Baseline:** Since 87.3% of the dataset is ham, a trivial model that classifies *every single message* as "ham" without analyzing text would achieve **87.3% Accuracy**, yet have **0% Recall** on spam.
2. **Symmetric Error Treatment:** Accuracy treats a False Positive (blocking your mother's emergency SMS or your university admission notice) as equivalent to a False Negative (letting a discount coupon through). In practice, False Positives destroy product trust.
3. **Conclusion:** Production spam filters optimize for **Precision**, **Recall**, **F1-Score**, and **False Positive Rate (FPR)**, never Accuracy alone.

---

## 🔍 Error Analysis (10 Failure Modes)

Examining the errors made by Logistic Regression reveals clear NLP challenges:

| # | Type | Actual $\rightarrow$ Predicted | Snippet | Root Cause Note |
| :-: | :--- | :---: | :--- | :--- |
| 1 | False Positive | Ham $\rightarrow$ Spam | *"Your daily text from me – a favour this time..."* | Contains `'daily'` and `'text'`, common in SMS subscription spam. |
| 2 | False Negative | Spam $\rightarrow$ Ham | *"Want explicit SEX in 30 secs? Ring 02073162414..."* | Slang and unconventional phone formatting diluted TF-IDF weights. |
| 3 | False Negative | Spam $\rightarrow$ Ham | *"You will recieve your tone within the next 24hrs..."* | Typo (`'recieve'` instead of `'receive'`) bypassed vocabulary matching. |
| 4 | False Positive | Ham $\rightarrow$ Spam | *"what are your new years plans?..."* | Words `'plans'` and `'new years'` frequently appear in marketing blasts. |
| 5 | False Negative | Spam $\rightarrow$ Ham | *"ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE MINS..."* | Heavy SMS shorthand (`'INCLU'`, `'SED'`, `'L8'`) concealed intent. |
| 6 | False Positive | Ham $\rightarrow$ Spam | *"I am waiting machan. Call me once you free..."* | Polysemy of `'free'`: means "available time", not monetary prize. |
| 7 | False Positive | Ham $\rightarrow$ Spam | *"Are you free now?can i call now?..."* | Availability check flagged due to high TF-IDF spam weight on `'free'`. |
| 8 | False Positive | Ham $\rightarrow$ Spam | *"Sir, i am waiting for your call, once free please call me..."* | Business courtesy flagged due to high TF-IDF spam weight on `'free'`. |
| 9 | False Negative | Spam $\rightarrow$ Ham | *"Hi its LUCY Hubby at meetins all day Fri & I will B alone..."* | Social engineering scam mimicking a conversational personal note. |
| 10 | False Negative | Spam $\rightarrow$ Ham | *"Missed call alert. These numbers called but left no message..."* | Spoofed carrier system alert without explicit spam promotional keywords. |

---

## 🤖 LLM Explanation Layer

### Design Principle:
- **Separation of Concerns:** The machine learning model (MultinomialNB) makes the classification decision.
- **Explainability Only:** The LLM is invoked **only when a message is flagged as spam** to provide a single-sentence reason highlighting specific red flags (e.g., artificial urgency, prize lure, unverified toll-free number).
- **Zero Hallucination of Decision:** The LLM cannot override or classify messages. This guarantees low operational latency, predictable token costs, and rock-solid reliability.
- **Graceful Fallback:** If `GEMINI_API_KEY` or `OPENAI_API_KEY` is not set in the environment, the function skips gracefully without throwing exceptions.

---

## 🚀 How to Run

### 1. Prerequisites
Ensure Python 3.10+ is installed. Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/spam-detection.git
cd spam-detection
pip install -r requirements.txt
```

### 2. Download Data & Train the Models
```bash
python download_data.py
python train.py
```
This inspects the data, trains all 3 models, outputs comparison tables, runs error analysis, and serializes artifacts into `models/`.

### 3. Open the Jupyter Notebook
```bash
jupyter notebook spam_detection.ipynb
```
Follow the 9 structured steps with rich markdown explanations.

### 4. Run the FastAPI Service
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/`

**Test with cURL:**
```bash
# Legitimate message (Ham)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"message": "Hey, can you send me the lecture notes from this morning?"}'

# Phishing message (Spam)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"message": "URGENT: Your account has been suspended. Call 0800-123-456 immediately to verify."}'
```

### 5. Run with Docker
```bash
# Build the Docker image
docker build -t spam-detector .

# Run the container
docker run -p 8000:8000 -e GEMINI_API_KEY="your_api_key_here" spam-detector
```

---

## ⚠️ Project Limitations
1. **Bag-of-Words / TF-IDF Semantics:** TF-IDF treats words independently and cannot capture deep context or word polysemy (e.g., distinguishing *"Are you free tomorrow?"* from *"Free prize"*).
2. **Adversarial Perturbations:** Intentional misspellings (`"rec1eve"`, `"fr33"`) or character insertions bypass static n-gram vocabularies.
3. **Dataset Age:** The UCI SMS Spam Collection reflects older SMS messaging patterns (2000s–2010s). Modern phishing exploits shortlinks, QR codes, and multi-factor authentication (MFA) fatigue.

---

## 🔮 Future Improvements
1. **Character-Level N-grams or Subwords:** Capture misspelled spam keywords and obfuscated shortcodes.
2. **Transformer-Based Embeddings:** Fine-tune lightweight contextual models like `DistilBERT` or `MobileBERT` to resolve word polysemy and conversational nuance.
3. **Feature Engineering:** Add metadata features such as message length, presence of URLs, digit density, and capital letter ratio.
4. **Active Learning & Drift Monitoring:** Continuously log borderline predictions for human-in-the-loop annotation to combat concept drift.
