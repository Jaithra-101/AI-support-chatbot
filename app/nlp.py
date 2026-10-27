"""NLTK text helpers shared by the intent matcher and the document chunker."""
import re

import nltk
from nltk.stem import WordNetLemmatizer

from .config import BASE_DIR

# Keep NLTK data inside the project so it works the same on every machine and in Docker
NLTK_DIR = BASE_DIR / "nltk_data"
nltk.data.path.insert(0, str(NLTK_DIR))

_NLTK_PACKAGES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "wordnet": "corpora/wordnet",
    "stopwords": "corpora/stopwords",
    "vader_lexicon": "sentiment/vader_lexicon",
}
# Words that change meaning, so we never strip them as stopwords
_KEEP = {"not", "no", "how", "when", "where", "who", "what", "why", "can", "cancel"}


def ensure_nltk_data():
    for pkg, path in _NLTK_PACKAGES.items():
        if not ((NLTK_DIR / path).exists() or (NLTK_DIR / f"{path}.zip").exists()):
            nltk.download(pkg, download_dir=str(NLTK_DIR), quiet=True)


ensure_nltk_data()
_lemmatizer = WordNetLemmatizer()
_stopwords = set(nltk.corpus.stopwords.words("english")) - _KEEP


def preprocess(text: str) -> str:
    """Lowercase, tokenize, drop punctuation/stopwords and lemmatize."""
    tokens = nltk.word_tokenize(text.lower())
    words = [t for t in tokens if re.fullmatch(r"[a-z0-9']+", t) and t not in _stopwords]
    return " ".join(_lemmatizer.lemmatize(w) for w in words)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
