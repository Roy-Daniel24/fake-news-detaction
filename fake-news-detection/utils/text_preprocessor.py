"""Text cleaning shared by training (train.py) and inference (app.py)."""

import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = PorterStemmer()
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(text) -> str:
    """Lowercase, strip URLs/symbols/numbers, drop stopwords, stem.

    Returns an empty string if nothing meaningful is left.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    tokens = [
        _stemmer.stem(w)
        for w in text.split()
        if len(w) > 2 and w not in ENGLISH_STOP_WORDS
    ]
    return " ".join(tokens)
