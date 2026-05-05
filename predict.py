"""
predict.py
===========
Interactive CLI — paste any email text and check if it is spam.
Run AFTER train.py has been executed.
"""

import joblib
import os
from preprocessor import preprocess

MODEL_DIR = "models"


def load_models():
    required = ["best_classifier.pkl", "tfidf_vectorizer.pkl"]
    for f in required:
        if not os.path.exists(f"{MODEL_DIR}/{f}"):
            print(f"[!] Model not found: {MODEL_DIR}/{f}")
            print("    Please run  python train.py  first.")
            exit(1)
    model  = joblib.load(f"{MODEL_DIR}/best_classifier.pkl")
    tfidf  = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.pkl")
    return model, tfidf


def classify_email(text: str, model, tfidf) -> dict:
    """Classify a single email. Returns label + confidence."""
    cleaned   = preprocess(text)
    features  = tfidf.transform([cleaned])

    pred      = model.predict(features)[0]
    label     = "SPAM" if pred == 1 else "HAM (Legitimate)"

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(features)[0]
        spam_prob  = proba[1]
        ham_prob   = proba[0]
    else:
        score     = model.decision_function(features)[0]
        spam_prob = 1 / (1 + 2.718 ** (-score))
        ham_prob  = 1 - spam_prob

    return {
        "label"    : label,
        "is_spam"  : bool(pred),
        "spam_prob": spam_prob,
        "ham_prob" : ham_prob,
    }


def print_result(result: dict, text: str):
    bar_len  = int(result["spam_prob"] * 20)
    bar      = "█" * bar_len + "░" * (20 - bar_len)
    verdict  = "🚨 SPAM" if result["is_spam"] else "✅ HAM (Legitimate)"

    print("\n" + "="*52)
    print("  CLASSIFICATION RESULT")
    print("="*52)
    print(f"  Verdict     : {verdict}")
    print(f"  Spam prob   : {result['spam_prob']*100:.1f}%  |{bar}|")
    print(f"  Ham prob    : {result['ham_prob']*100:.1f}%")

    # Explain why
    print("\n  Why this verdict:")
    text_lower = text.lower()
    spam_signals = []
    if any(w in text_lower for w in ["free", "click here", "win", "won", "prize"]):
        spam_signals.append("Contains common spam trigger words (free, win, click here)")
    if any(w in text_lower for w in ["$", "dollar", "money", "cash", "loan"]):
        spam_signals.append("Contains financial bait language")
    if any(w in text_lower for w in ["http", "www", ".com", "url"]):
        spam_signals.append("Contains suspicious URLs")
    if any(w in text_lower for w in ["urgent", "limited", "act now", "immediately"]):
        spam_signals.append("Uses urgency/pressure language")
    if text.upper() == text and len(text) > 20:
        spam_signals.append("Excessive use of CAPITAL LETTERS")

    if spam_signals and result["is_spam"]:
        for s in spam_signals:
            print(f"    • {s}")
    elif not result["is_spam"]:
        print("    • No major spam signals detected")
        print("    • Conversational tone typical of legitimate emails")

    print("="*52)


def main():
    print("\n" + "="*52)
    print("  SPAM EMAIL CLASSIFIER")
    print("  Powered by TF-IDF + Naive Bayes / Logistic Regression")
    print("="*52)

    model, tfidf = load_models()
    print("\n[✓] Models loaded. Ready to classify emails.\n")

    while True:
        print("Paste your email text below (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)

        text = " ".join(lines).strip()
        if not text:
            print("No text entered. Try again.")
            continue

        result = classify_email(text, model, tfidf)
        print_result(result, text)

        again = input("\nClassify another email? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye!\n")
            break


if __name__ == "__main__":
    main()
