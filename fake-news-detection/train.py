"""
train.py
Trains Naive Bayes, Logistic Regression and a Passive-Aggressive classifier,
picks the best by F1-score and saves artifacts into ./models.

Dataset resolution order:
  1. dataset/news_dataset.csv            (columns: text, label[, subject]; label 1 = fake)
  2. dataset/Fake.csv + dataset/True.csv (Kaggle "Fake and Real News" format)
  3. an auto-generated SYNTHETIC demo dataset (for testing the pipeline only)

Usage:  python train.py
"""

import json
import os
import random

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from utils.text_preprocessor import clean_text

DATASET_DIR = "dataset"
MODEL_DIR = "models"
DATASET_PATH = os.path.join(DATASET_DIR, "news_dataset.csv")
SEED = 42


def make_synthetic_dataset(n_per_class: int = 1000) -> pd.DataFrame:
    """Tiny template-based dataset. NOT real data: use it only to smoke-test the app."""
    rng = random.Random(SEED)
    real_subj = ["the senate", "city officials", "researchers", "the central bank", "local police", "the ministry"]
    real_verb = ["announced", "published a report on", "approved", "reviewed", "confirmed details of", "discussed"]
    real_obj = ["the annual budget", "a new transport plan", "quarterly economic data", "a climate study",
                "updated safety regulations", "the education policy"]
    fake_subj = ["anonymous insiders", "a secret source", "shocking leaked memos", "a banned doctor", "unnamed experts"]
    fake_verb = ["reveal", "expose", "admit", "claim they hid", "finally confirm"]
    fake_obj = ["the miracle cure they hate", "a global cover-up", "aliens controlling the government",
                "the truth the media won't show", "a hidden plot to control you"]
    fake_tail = [" Share before it is deleted!", " You won't believe what happens next.", " Wake up people!", ""]
    real_tail = [" Officials said more details will follow.", " The report is available online.", ""]

    rows = []
    for _ in range(n_per_class):
        rows.append((f"{rng.choice(real_subj)} {rng.choice(real_verb)} {rng.choice(real_obj)}.{rng.choice(real_tail)}",
                     0, rng.choice(["politicsNews", "worldnews"])))
        rows.append((f"{rng.choice(fake_subj)} {rng.choice(fake_verb)} {rng.choice(fake_obj)}!{rng.choice(fake_tail)}",
                     1, rng.choice(["News", "left-news"])))
    return pd.DataFrame(rows, columns=["text", "label", "subject"]).sample(frac=1, random_state=SEED).reset_index(drop=True)


def load_dataset() -> pd.DataFrame:
    os.makedirs(DATASET_DIR, exist_ok=True)
    fake_p, true_p = os.path.join(DATASET_DIR, "Fake.csv"), os.path.join(DATASET_DIR, "True.csv")

    if os.path.exists(DATASET_PATH):
        print(f"Using {DATASET_PATH}")
        return pd.read_csv(DATASET_PATH)

    if os.path.exists(fake_p) and os.path.exists(true_p):
        print("Merging Fake.csv + True.csv")
        fake, real = pd.read_csv(fake_p), pd.read_csv(true_p)
        fake["label"], real["label"] = 1, 0
        df = pd.concat([real, fake], ignore_index=True)
        if "title" in df.columns:  # title + body gives the model more signal
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        df.to_csv(DATASET_PATH, index=False)
        return df

    print("WARNING: no dataset found. Generating a SYNTHETIC demo dataset (results are not meaningful).")
    df = make_synthetic_dataset()
    df.to_csv(DATASET_PATH, index=False)
    return df


def main():
    df = load_dataset()
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)
    print(f"{len(df)} articles | fake share: {df['label'].mean():.1%}")

    print("Cleaning text...")
    df["clean"] = df["text"].astype(str).map(clean_text)
    df = df[df["clean"].str.len() > 0]

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=SEED, stratify=df["label"]
    )

    # Fit the vectorizer on training data only (avoids test-set leakage).
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2 if len(df) > 500 else 1)
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)

    candidates = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        # Equivalent to PassiveAggressiveClassifier (deprecated in scikit-learn 1.8).
        "Passive Aggressive": SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1",
                                            eta0=1.0, random_state=SEED),
    }

    results, fitted = {}, {}
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1_score": f1_score(y_test, pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        }
        fitted[name] = clf
        print(f"{name:22s} acc={results[name]['accuracy']:.4f}  f1={results[name]['f1_score']:.4f}")

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    print(f"Best model: {best_name}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(fitted[best_name], os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    with open(os.path.join(MODEL_DIR, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_model_name": best_name, "models": results}, f, indent=2)
    print("Saved artifacts to ./models")


if __name__ == "__main__":
    main()
