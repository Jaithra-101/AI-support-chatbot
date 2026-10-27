"""Self-learning loop: turn unanswered questions into new FAQs.

1. Collect every on-topic question the bot couldn't answer.
2. Embed them and group by meaning (agglomerative clustering), so
   "do you price match", "will you match a lower price" and "price matching?"
   become ONE knowledge gap with a count.
3. An admin writes the answer once; all the questions in that group become the
   example phrasings of a new FAQ intent, and the bot answers them from then on.
"""
import re
from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from . import db
from .config import GAP_CLUSTER_DISTANCE


def find_gaps(embed) -> list[dict]:
    """Return [{label, count, partial, questions}] biggest gap first.

    partial = how many times the bot only quoted a "closest passage" instead of a real answer.
    """
    rows = db.unresolved_questions()
    asked = Counter(q.strip() for q, _ in rows)
    partial = Counter(q.strip() for q, route in rows if route == "passage")
    questions = list(asked)
    if not questions:
        return []

    vectors = embed(questions)
    if len(questions) == 1:
        labels = [0]
    else:
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=GAP_CLUSTER_DISTANCE,
        ).fit_predict(vectors)
    gaps = []
    for cluster in set(labels):
        idx = [i for i, label in enumerate(labels) if label == cluster]
        # The question nearest the cluster's centre is the best summary of it
        centre = vectors[idx].mean(axis=0)
        label = questions[idx[int(np.argmax(vectors[idx] @ centre))]]
        gaps.append({
            "label": label,
            "count": sum(asked[questions[i]] for i in idx),
            "partial": sum(partial[questions[i]] for i in idx),
            "questions": sorted((questions[i] for i in idx), key=lambda q: -asked[q]),
        })
    return sorted(gaps, key=lambda g: -g["count"])


def suggest_intent_name(text: str) -> str:
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3][:3]
    return "_".join(words) or "new_intent"
