"""Fast path: classic TF-IDF intent matching over FAQ example phrasings."""
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import db
from .nlp import preprocess


class IntentMatcher:
    def __init__(self):
        self.reload()

    def reload(self):
        """Rebuild the TF-IDF index from the FAQ table (call after FAQs change)."""
        self.labels, patterns, self.answers = [], [], {}
        for faq in db.list_faqs():
            self.answers[faq["intent"]] = faq["answer"]
            for p in faq["patterns"]:
                self.labels.append(faq["intent"])
                patterns.append(preprocess(p))
        self.vectorizer = None
        if patterns:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self.matrix = self.vectorizer.fit_transform(patterns)

    def _with_synonyms(self, tokens: list[str]) -> list[str]:
        """Swap unknown words for a WordNet synonym the FAQs do know.

        e.g. "when do you shut" -> "when do you close"
        """
        vocab = self.vectorizer.vocabulary_
        out = []
        for t in tokens:
            if t not in vocab:
                synonyms = {l.name().lower() for s in wordnet.synsets(t) for l in s.lemmas()}
                t = next((s for s in sorted(synonyms) if s in vocab), t)
            out.append(t)
        return out

    def rank(self, text: str) -> list[tuple[str, float]]:
        """Return [(intent, score)] best first, one entry per intent.

        Score = cosine similarity x share of the question's words the FAQ
        vocabulary knows. TF-IDF silently ignores unknown words, so without the
        coverage factor "what about sale items?" would match "what are you"
        perfectly on the word "what" alone.
        """
        if self.vectorizer is None:
            return []
        tokens = self._with_synonyms(preprocess(text).split())
        if not tokens:
            return []
        query = self.vectorizer.transform([" ".join(tokens)])
        if query.nnz == 0:  # no known vocabulary at all
            return []
        coverage = sum(t in self.vectorizer.vocabulary_ for t in tokens) / len(tokens)
        best: dict[str, float] = {}
        for label, sim in zip(self.labels, cosine_similarity(query, self.matrix)[0]):
            # squared, so a half-understood question ("what about the battery?") can't reach 0.5
            best[label] = max(best.get(label, 0.0), float(sim) * coverage ** 2)
        return sorted(best.items(), key=lambda x: -x[1])

    def match(self, text: str) -> tuple[str | None, float]:
        ranked = self.rank(text)
        return ranked[0] if ranked else (None, 0.0)

    def vocabulary(self) -> set[str]:
        return set(self.vectorizer.vocabulary_) if self.vectorizer is not None else set()

    def example(self, intent: str) -> str:
        """First example phrasing of an intent, used for clickable suggestions."""
        return next(f["patterns"][0] for f in db.list_faqs() if f["intent"] == intent)
