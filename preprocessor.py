"""
preprocessor.py
================
Text cleaning and preprocessing pipeline for email classification.

Steps:
1. Lowercase
2. Remove URLs, email addresses, phone numbers
3. Remove special characters and numbers
4. Tokenise
5. Remove stopwords
6. Stem words (PorterStemmer)
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from typing import List

# Download required NLTK data (only runs once)
def download_nltk_data():
    packages = ["punkt", "stopwords", "punkt_tab"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

download_nltk_data()

stemmer        = PorterStemmer()
STOP_WORDS     = set(stopwords.words("english"))

# Keep some spam-signal words that are normally stopwords
KEEP_WORDS     = {"free", "not", "no", "won", "win", "now", "click", "here", "buy"}
STOP_WORDS     = STOP_WORDS - KEEP_WORDS


def clean_text(text: str) -> str:
    """Remove noise from raw email text."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+",        " urltoken ",    text)  # URLs
    text = re.sub(r"\S+@\S+",               " emailtoken ",  text)  # emails
    text = re.sub(r"\b\d[\d\s\-().]{6,}\d\b"," phonetoken ", text)  # phones
    text = re.sub(r"\$[\d,]+",              " moneytoken ",  text)  # prices
    text = re.sub(r"[^a-z\s]",              " ",             text)  # special chars
    text = re.sub(r"\s+",                   " ",             text).strip()
    return text


def tokenise(text: str) -> List[str]:
    """Split cleaned text into tokens."""
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def stem(tokens: List[str]) -> List[str]:
    return [stemmer.stem(t) for t in tokens]


def preprocess(text: str, do_stem: bool = True) -> str:
    """
    Full preprocessing pipeline.
    Returns a single cleaned string ready for vectorisation.
    """
    text   = clean_text(text)
    tokens = tokenise(text)
    tokens = remove_stopwords(tokens)
    if do_stem:
        tokens = stem(tokens)
    return " ".join(tokens)


def preprocess_batch(texts, do_stem: bool = True) -> List[str]:
    """Process a list of email texts."""
    return [preprocess(t, do_stem) for t in texts]
