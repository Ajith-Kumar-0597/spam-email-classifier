"""
train.py
=========
Trains and evaluates two classifiers on email spam data:
  1. Multinomial Naive Bayes   (classic spam baseline)
  2. Logistic Regression       (stronger linear model)

Feature extraction:
  - Bag of Words  (CountVectorizer)
  - TF-IDF        (TfidfVectorizer)

Saves best model + vectorizer to models/
Saves all evaluation charts to charts/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, accuracy_score, f1_score
)
from preprocessor import preprocess_batch

os.makedirs("models",  exist_ok=True)
os.makedirs("charts",  exist_ok=True)

# ── Load & preprocess ─────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/emails.csv")
df.dropna(subset=["text", "label"], inplace=True)
df["label_num"] = (df["label"] == "spam").astype(int)

print(f"Total emails  : {len(df):,}")
print(f"Spam          : {df['label_num'].sum():,} ({df['label_num'].mean()*100:.1f}%)")
print(f"Ham           : {(df['label_num']==0).sum():,}\n")

print("Preprocessing text (cleaning, tokenising, stemming)...")
df["clean_text"] = preprocess_batch(df["text"].tolist())
print("[✓] Preprocessing done\n")

X = df["clean_text"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Feature extraction ────────────────────────────────────────────────────────
bow_vec   = CountVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)

X_train_bow   = bow_vec.fit_transform(X_train)
X_test_bow    = bow_vec.transform(X_test)

X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf  = tfidf_vec.transform(X_test)

print(f"BoW features   : {X_train_bow.shape[1]:,}")
print(f"TF-IDF features: {X_train_tfidf.shape[1]:,}\n")

# ── Train models ──────────────────────────────────────────────────────────────
models = {
    "Naive Bayes (BoW)"       : (MultinomialNB(alpha=0.1),      X_train_bow,   X_test_bow),
    "Naive Bayes (TF-IDF)"    : (MultinomialNB(alpha=0.1),      X_train_tfidf, X_test_tfidf),
    "Logistic Regression"     : (LogisticRegression(C=1.0, max_iter=1000, random_state=42),
                                  X_train_tfidf, X_test_tfidf),
}

results = {}
print("="*55)
for name, (model, Xtr, Xte) in models.items():
    model.fit(Xtr, y_train)
    preds  = model.predict(Xte)
    acc    = accuracy_score(y_test, preds)
    f1     = f1_score(y_test, preds)
    cv     = cross_val_score(model, Xtr, y_train, cv=5, scoring="f1").mean()

    results[name] = {
        "model": model, "Xtr": Xtr, "Xte": Xte,
        "preds": preds, "acc": acc, "f1": f1, "cv_f1": cv,
    }
    print(f"\n{name}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  CV F1     : {cv:.4f}")
    print(classification_report(y_test, preds, target_names=["Ham", "Spam"]))

# ── Save best model ───────────────────────────────────────────────────────────
best_name  = max(results, key=lambda n: results[n]["f1"])
best       = results[best_name]
joblib.dump(best["model"], "models/best_classifier.pkl")
joblib.dump(tfidf_vec,     "models/tfidf_vectorizer.pkl")
joblib.dump(bow_vec,       "models/bow_vectorizer.pkl")
print(f"\n[✓] Best model saved: {best_name}")

# ── Chart 1 – Confusion Matrices ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["preds"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    ax.set_title(f"{name}\nAcc: {res['acc']*100:.1f}%", fontweight="bold", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("charts/confusion_matrices.png", dpi=150)
plt.close()
print("[✓] Saved: charts/confusion_matrices.png")

# ── Chart 2 – ROC Curves ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#378ADD", "#1D9E75", "#E24B4A"]
for (name, res), color in zip(results.items(), colors):
    if hasattr(res["model"], "predict_proba"):
        proba = res["model"].predict_proba(res["Xte"])[:, 1]
    else:
        proba = res["model"].decision_function(res["Xte"])
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={roc_auc:.3f})")

ax.plot([0,1],[0,1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Spam Classifier Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/roc_curves.png", dpi=150)
plt.close()
print("[✓] Saved: charts/roc_curves.png")

# ── Chart 3 – Model Comparison Bar Chart ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
names    = list(results.keys())
acc_vals = [results[n]["acc"]*100 for n in names]
f1_vals  = [results[n]["f1"]*100 for n in names]
cv_vals  = [results[n]["cv_f1"]*100 for n in names]

x     = np.arange(len(names))
width = 0.25
ax.bar(x - width, acc_vals, width, label="Accuracy",   color="#378ADD", edgecolor="white")
ax.bar(x,         f1_vals,  width, label="F1 Score",   color="#1D9E75", edgecolor="white")
ax.bar(x + width, cv_vals,  width, label="CV F1 (5-fold)", color="#EF9F27", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(80, 100)
ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
ax.legend()
ax.spines[["top","right"]].set_visible(False)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.1,
            f"{rect.get_height():.1f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig("charts/model_comparison.png", dpi=150)
plt.close()
print("[✓] Saved: charts/model_comparison.png")

# ── Chart 4 – Top Spam vs Ham Words ──────────────────────────────────────────
lr_model   = results["Logistic Regression"]["model"]
features   = tfidf_vec.get_feature_names_out()
coef       = lr_model.coef_[0]

top_spam_idx = np.argsort(coef)[-20:]
top_ham_idx  = np.argsort(coef)[:20]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, idx, title, color in [
    (axes[0], top_spam_idx, "Top 20 Spam Indicator Words", "#E24B4A"),
    (axes[1], top_ham_idx,  "Top 20 Ham Indicator Words",  "#1D9E75"),
]:
    words  = features[idx]
    scores = coef[idx]
    ax.barh(words, np.abs(scores), color=color, edgecolor="white")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Coefficient magnitude")
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("charts/top_words.png", dpi=150)
plt.close()
print("[✓] Saved: charts/top_words.png")

# ── Chart 5 – Class Distribution ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
counts = df["label"].value_counts()
axes[0].bar(["Ham", "Spam"], [counts.get("ham",0), counts.get("spam",0)],
            color=["#1D9E75", "#E24B4A"], edgecolor="white")
axes[0].set_title("Email Class Distribution", fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].spines[["top","right"]].set_visible(False)

axes[1].pie([counts.get("ham",0), counts.get("spam",0)],
            labels=["Ham", "Spam"], autopct="%1.1f%%",
            colors=["#1D9E75", "#E24B4A"],
            wedgeprops=dict(width=0.6, edgecolor="white"))
axes[1].set_title("Spam vs Ham Split", fontweight="bold")
plt.tight_layout()
plt.savefig("charts/class_distribution.png", dpi=150)
plt.close()
print("[✓] Saved: charts/class_distribution.png")

print("\n✅ Training complete! Run predict.py to classify emails.")
