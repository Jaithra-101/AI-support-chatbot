"""Trained intent classifier on the Bitext customer-support dataset.

26,872 real-style customer messages across 27 intents (track_order,
cancel_order, get_refund, payment_issue, ...). We learn a text classifier:

    message -> TF-IDF features (word 1-2 grams + character 2-5 grams)
            -> Logistic Regression -> probability for each of the 27 intents

Character n-grams make it robust to typos ("cancel my oorder"), and the
probability tells the router how sure the model is.

Train / evaluate from the command line:
    python -m app.intent_model
"""
import csv
import json
import re
import time
import urllib.request

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import BASE_DIR, DATA_DIR

DATASET_PATH = DATA_DIR / "bitext_customer_support.csv"
EXTRA_PATH = DATA_DIR / "extra_training.csv"  # our own examples for wordings Bitext misses
EXTRA_WEIGHT = 20                              # so ~60 examples still matter next to 27k
EVAL_PATH = DATA_DIR / "eval_questions.csv"    # hand-written, never trained on
DATASET_URL = ("https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset"
               "/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "intent_classifier.joblib"
METRICS_PATH = MODEL_DIR / "intent_metrics.json"

_PLACEHOLDER = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")
_NUMBER = re.compile(r"#?\b\d{3,}\b")


def normalize(text: str) -> str:
    """Lowercase and map order numbers / dataset placeholders to one token.

    The dataset writes "{{Order Number}}" where real users type "10234", so
    both become "numtoken" and the model treats them the same.
    """
    def placeholder(m: re.Match) -> str:
        name = m.group(1).lower()
        return " numtoken " if "number" in name else f" {name} "

    text = _PLACEHOLDER.sub(placeholder, text)
    text = _NUMBER.sub(" numtoken ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def _read(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_dataset() -> tuple[list[str], list[str], list[float]]:
    """Return (texts, intents, sample weights)."""
    if not DATASET_PATH.exists():
        print(f"Downloading dataset to {DATASET_PATH} ...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    rows = [(r, 1.0) for r in _read(DATASET_PATH)]
    if EXTRA_PATH.exists():
        rows += [(r, EXTRA_WEIGHT) for r in _read(EXTRA_PATH)]
    return ([normalize(r["instruction"]) for r, _ in rows],
            [r["intent"] for r, _ in rows],
            [w for _, w in rows])


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(C=10, max_iter=3000))])


def train(verbose: bool = True) -> dict:
    """Evaluate twice, then refit on all data and save.

    1. Held-out 20% of the dataset. Scores ~99% but is optimistic: Bitext is
       template-generated, so test sentences closely resemble training ones.
    2. Our hand-written eval_questions.csv, never trained on. This is the
       honest "real users" number.
    """
    texts, labels, weights = load_dataset()
    x_train, x_test, y_train, y_test, w_train, _ = train_test_split(
        texts, labels, weights, test_size=0.2, stratify=labels, random_state=42)

    start = time.time()
    model = build_pipeline().fit(x_train, y_train, clf__sample_weight=w_train)
    predicted = model.predict(x_test)
    report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
    metrics = {
        "examples": len(texts),
        "intents": len(set(labels)),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "accuracy": round(accuracy_score(y_test, predicted), 4),
        "macro_f1": round(f1_score(y_test, predicted, average="macro"), 4),
        "per_intent_f1": {k: round(v["f1-score"], 4) for k, v in report.items() if k in set(labels)},
    }
    if verbose:
        print(classification_report(y_test, predicted, digits=3, zero_division=0))
        print(f"Dataset test split: accuracy {metrics['accuracy']:.2%} | macro F1 {metrics['macro_f1']:.2%} "
              f"| trained in {time.time() - start:.0f}s")

    final = build_pipeline().fit(texts, labels, clf__sample_weight=weights)  # deployed model sees everything

    if EVAL_PATH.exists():
        rows = _read(EVAL_PATH)
        guesses = final.predict([normalize(r["question"]) for r in rows])
        misses = [(r["question"], r["intent"], g) for r, g in zip(rows, guesses) if g != r["intent"]]
        metrics["real_world_accuracy"] = round(1 - len(misses) / len(rows), 4)
        metrics["real_world_size"] = len(rows)
        if verbose:
            print(f"Hand-written questions: accuracy {metrics['real_world_accuracy']:.2%} on {len(rows)}")
            for q, want, got in misses:
                print(f"   miss: {q!r} expected {want}, got {got}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


class IntentClassifier:
    """Loads the trained model (training it first if it doesn't exist yet)."""

    def __init__(self):
        if not MODEL_PATH.exists():
            print("No trained intent model found - training one now (about a minute)...")
            train(verbose=False)
        self.model = joblib.load(MODEL_PATH)
        self.metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

    def predict(self, text: str) -> tuple[str, float]:
        """Return (intent, probability)."""
        probs = self.model.predict_proba([normalize(text)])[0]
        best = int(probs.argmax())
        return self.model.classes_[best], float(probs[best])


if __name__ == "__main__":
    train()
