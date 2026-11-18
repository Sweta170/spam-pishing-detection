"""
model_utils.py
==============
Utility functions for text preprocessing, dataset loading, model training,
and metric evaluation for the Spam/Phishing Detection project.
Designed to be beginner-friendly with clear inline explanations for interview prep.
"""

import os
import re
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

def clean_text(text: str) -> str:
    """
    Cleans raw SMS/email text for machine learning:
    1. Converts text to lowercase (ensures 'FREE', 'Free', and 'free' are treated identically).
    2. Replaces punctuation with whitespace (keeps words separated, e.g., 'win!call' -> 'win call').
    3. Tokenizes and filters out English stopwords and single-character noise.
    
    Note: We intentionally keep numbers (e.g. '87121', '1000') because prize amounts
    and SMS shortcodes are strong signals of spam.
    """
    if not isinstance(text, str):
        return ""
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove punctuation by replacing with space
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    # Step 3: Split into words, remove stopwords and single-char remnants
    tokens = [
        word for word in text.split()
        if word not in ENGLISH_STOP_WORDS and len(word) > 1
    ]
    
    # Step 4: Rejoin into a single space-delimited string
    return " ".join(tokens)


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the SMS Spam Collection dataset from disk.
    - Separator: tab ('\t')
    - Columns: 'label' (ham/spam) and 'message' (raw text)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Run download_data.py first.")
    
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )
    
    # Map text labels to binary integers for ML algorithms
    # 0 = ham (legitimate message), 1 = spam (unsolicited / phishing message)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def prepare_data_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, drop_duplicates: bool = True):
    """
    Cleans text and splits into train and test sets using STRATIFIED sampling.
    
    Why Stratify?
    In this dataset, only ~13.4% of messages are spam. Stratification guarantees
    that both the training set and testing set retain this exact same 86.6% / 13.4%
    ratio. Without stratification, the test set could randomly receive too few spam samples.
    
    Why Split BEFORE TF-IDF Fitting?
    To prevent DATA LEAKAGE! If TF-IDF learns vocabulary, document frequencies, or IDF weights
    from the entire dataset, information from the test set would 'leak' into the training
    process, producing overly optimistic, non-generalizable evaluation results.
    """
    data = df.copy()
    if drop_duplicates:
        # Drop duplicates to prevent the exact same message appearing in both train and test
        data = data.drop_duplicates(subset=["message"], keep="first").reset_index(drop=True)
        
    data["clean_message"] = data["message"].apply(clean_text)
    
    X_train_text, X_test_text, y_train, y_test, raw_train, raw_test = train_test_split(
        data["clean_message"],
        data["label_num"],
        data["message"],
        test_size=test_size,
        random_state=random_state,
        stratify=data["label_num"]
    )
    
    return {
        "df": data,
        "X_train_text": X_train_text,
        "X_test_text": X_test_text,
        "y_train": y_train,
        "y_test": y_test,
        "raw_train": raw_train,
        "raw_test": raw_test,
    }


def fit_vectorizer_and_transform(X_train_text, X_test_text, max_features: int = 3000):
    """
    Fits TF-IDF on training messages ONLY, then transforms both train and test sets.
    
    TF-IDF (Term Frequency - Inverse Document Frequency):
    - TF: How often a word appears in a specific message.
    - IDF: log(Total Documents / Documents containing the word).
      Rare words across all documents get high weights (e.g. 'jackpot', 'claim').
      Common words across all documents get low weights.
    """
    tfidf = TfidfVectorizer(max_features=max_features)
    # fit_transform on train ONLY
    X_train_vec = tfidf.fit_transform(X_train_text)
    # transform on test ONLY (never fit!)
    X_test_vec = tfidf.transform(X_test_text)
    return tfidf, X_train_vec, X_test_vec


def train_and_evaluate_all_models(X_train_vec, y_train, X_test_vec, y_test):
    """
    Trains three distinct classification models and calculates comprehensive metrics:
    1. Multinomial Naive Bayes (fast, probabilistic baseline)
    2. Logistic Regression with class_weight='balanced' (penalizes spam misclassifications more)
    3. Random Forest (non-linear tree ensemble)
    """
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_vec, y_train)
        trained_models[name] = model
        
        # Predict on unseen test set
        y_pred = model.predict(X_test_vec)
        
        # Compute Confusion Matrix:
        # [[TN, FP],
        #  [FN, TP]]
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Precision = TP / (TP + FP) -> Of all messages we called spam, how many were actually spam?
        prec = precision_score(y_test, y_pred, zero_division=0)
        
        # Recall = TP / (TP + FN) -> Of all real spam messages, how many did we catch?
        rec = recall_score(y_test, y_pred, zero_division=0)
        
        # F1-Score = 2 * (P * R) / (P + R) -> Harmonic mean of precision and recall
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Accuracy = (TP + TN) / Total
        acc = accuracy_score(y_test, y_pred)
        
        # False Positive Rate (FPR) = FP / (FP + TN)
        # Fraction of legitimate (ham) messages mistakenly flagged as spam
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
    return trained_models, results_df
