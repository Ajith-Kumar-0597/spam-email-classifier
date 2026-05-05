# Spam Email Classifier

A machine learning project that detects spam emails using NLP text preprocessing, TF-IDF feature extraction, and multiple classifiers — with a CLI to test any email in real time.

---

## Why I built this

Spam detection is one of the classic NLP problems, but I wanted to go beyond just training a model and calling it done. I focused on understanding why certain words make an email look like spam, how different feature extraction methods (BoW vs TF-IDF) affect performance, and why a model that scores 99% accuracy might still be missing important spam.

---

## What it does

- Cleans and preprocesses email text — removes URLs, special characters, stopwords, and stems words
- Extracts features using Bag of Words and TF-IDF (with bigrams)
- Trains and compares three models: Naive Bayes (BoW), Naive Bayes (TF-IDF), and Logistic Regression
- Visualises top spam/ham indicator words, confusion matrices, and ROC curves
- Interactive CLI — paste any email and get an instant spam verdict with confidence score

---

## How the NLP pipeline works

```
Raw Email Text
      │
      ▼
  Lowercase + remove URLs, emails, phone numbers
      │
      ▼
  Remove special characters and numbers
      │
      ▼
  Tokenise → Remove stopwords → Stem (PorterStemmer)
      │
      ▼
  TF-IDF Vectorisation (5000 features, unigrams + bigrams)
      │
      ▼
  Naive Bayes / Logistic Regression Classifier
      │
      ▼
  Spam or Ham prediction + confidence score
```

---

## Why TF-IDF over Bag of Words

BoW counts raw word frequencies — so common words like "the" or "is" get high scores even though they carry no spam signal. TF-IDF downweights words that appear everywhere and upweights words that are rare but specific to certain emails (like "lottery", "click here", "free offer"). This gives the model much better signal for spam detection.

---

## Models used

| Model | Feature | Notes |
|---|---|---|
| Multinomial Naive Bayes | BoW | Fast baseline, works well with sparse text data |
| Multinomial Naive Bayes | TF-IDF | Better than raw counts for text classification |
| Logistic Regression | TF-IDF | Best overall — also gives interpretable feature weights |

---

## Project structure

```
spam-email-classifier/
│
├── data/
│   └── emails.csv
├── models/
│   ├── best_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── bow_vectorizer.pkl
├── charts/
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── model_comparison.png
│   ├── top_words.png
│   └── class_distribution.png
├── notebooks/
│   └── spam_classifier_analysis.ipynb
├── preprocessor.py
├── generate_dataset.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Generate dataset (or use real Kaggle dataset — see note below)
python generate_dataset.py

# Train models
python train.py

# Classify emails interactively
python predict.py
```

**Using the real SpamAssassin dataset from Kaggle:**
```bash
kaggle datasets download -d beatoa/spamassassin-public-corpus --unzip -p data/
```

---

## Results

Logistic Regression with TF-IDF performed best across accuracy, F1, and cross-validation scores. The most informative spam words were: `free`, `click`, `win`, `money`, `urgent`, `prize`, `offer`. The most informative ham words were conversational terms like `meeting`, `attached`, `please`, `review`, `team`.

False positives (ham marked as spam) were rare — which matters because missing a legitimate email is usually worse than letting a spam through.

---

## Things I learned

- Why stemming matters — `winning`, `winner`, `wins` all become `win`, reducing feature sparsity
- The difference between BoW and TF-IDF and when each works better
- Why precision matters more than recall in spam filtering — you don't want legitimate emails blocked
- How to read feature coefficients from Logistic Regression to understand what the model actually learned
- Why 5-fold cross validation gives a more honest picture of model performance than a single train/test split

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-blue)

---

## Author

**Ajith Kumar**
[LinkedIn](https://www.linkedin.com/in/ajith-python-dev)
