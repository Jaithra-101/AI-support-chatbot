"""Slow path: retrieval-augmented answers over uploaded documents, with citations.

1. Documents are split into sentence-aligned chunks and embedded with a
   sentence-transformer; vectors live in SQLite next to the text.
2. A question is embedded and compared with every chunk (cosine similarity).
3. An extractive QA model reads the top chunks and picks the answer span, so
   every answer is a quote from a real document - no hallucinated text.
"""
import re

import numpy as np

from . import db
from .config import (CHUNK_WORDS, DOMAIN_TOPICS, EMBED_MODEL, FAQ_MATCH_THRESHOLD, HEADING_MATCH_THRESHOLD,
                     QA_MODEL, QA_THRESHOLD, RETRIEVAL_THRESHOLD, TOP_K)
from .nlp import split_sentences

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")


def chunk_page(text: str, max_words: int = CHUNK_WORDS) -> list[tuple[str | None, str]]:
    """Split a page into (section heading, chunk text) pairs.

    Chunks never cross a Markdown heading and overlap by one sentence so an
    answer that spans a chunk boundary is still retrievable.
    """
    sections, heading, lines = [], None, []
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            if lines:
                sections.append((heading, " ".join(lines)))
            heading, lines = m.group(1).strip(), []
        elif line.strip():
            lines.append(line.strip())
    if lines:
        sections.append((heading, " ".join(lines)))

    chunks = []
    for heading, body in sections:
        current, words = [], 0
        for sent in split_sentences(body):
            n = len(sent.split())
            if current and words + n > max_words:
                chunks.append((heading, " ".join(current)))
                current, words = current[-1:], len(current[-1].split())
            current.append(sent)
            words += n
        if current:
            chunks.append((heading, " ".join(current)))
    return chunks


def display_section(section: str | None) -> str | None:
    """The first phrasing of a heading, for showing to people."""
    return section.split("|")[0].strip() if section else section


def _with_section(section: str | None, text: str) -> str:
    return f"{section}. {text}" if section else text


class DocumentStore:
    def __init__(self):
        self._embedder = None
        self._qa = None
        self._domain = None
        self.reload()

    # Models load lazily so the server starts fast and FAQ-only use stays light.
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBED_MODEL)
        return self._embedder

    def qa(self, question: str, context: str, max_answer_tokens: int = 30) -> dict:
        """Extractive QA: find the span of `context` that best answers `question`.

        The model scores every token as a possible answer start and end; the
        best valid (start <= end, inside the context) pair is the answer.
        Done by hand because transformers v5 dropped the QA pipeline.
        """
        import torch
        if self._qa is None:
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
            self._qa = (AutoTokenizer.from_pretrained(QA_MODEL),
                        AutoModelForQuestionAnswering.from_pretrained(QA_MODEL).eval())
        tokenizer, model = self._qa

        enc = tokenizer(question, context, truncation="only_second", max_length=384,
                        return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        in_context = [sid == 1 for sid in enc.sequence_ids(0)]
        with torch.no_grad():
            out = model(**enc)
        start_p = out.start_logits[0].softmax(-1).tolist()
        end_p = out.end_logits[0].softmax(-1).tolist()

        best = (0.0, 0, 0)
        for s in (i for i, ok in enumerate(in_context) if ok):
            for e in range(s, min(s + max_answer_tokens, len(in_context))):
                if not in_context[e]:
                    break
                score = start_p[s] * end_p[e]
                if score > best[0]:
                    best = (score, s, e)
        score, s, e = best
        # SQuAD2 models point at the [CLS] token when the context doesn't contain
        # the answer. If that "no answer" option beats the best span, say so.
        if start_p[0] * end_p[0] > score:
            return {"score": 0.0, "answer": ""}
        return {"score": score, "answer": context[offsets[s][0]:offsets[e][1]]}

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.embedder.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def reload(self):
        """Load every chunk vector into memory for fast similarity search."""
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT c.id, c.page, c.section, c.text, c.embedding, d.name "
                "FROM chunks c JOIN documents d ON d.id = c.document_id ORDER BY c.id"
            ).fetchall()
        self.meta = [{k: r[k] for k in ("id", "page", "section", "text", "name")} for r in rows]
        self.matrix = (
            np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]) if rows else None
        )
        self.sections = sorted({m["section"] for m in self.meta if m["section"]})
        # A heading may list alternative phrasings of the same question:
        #   ## What if my order arrived damaged or broken? | Can I return a product that arrived damaged?
        self._variants = [(s, v.strip()) for s in self.sections for v in s.split("|") if v.strip()]
        self._domain = None  # section headings may have changed

    def _anchors(self) -> np.ndarray:
        """Embeddings of every heading variant followed by DOMAIN_TOPICS (cached)."""
        if self._domain is None:
            self._domain = self.embed([v for _, v in self._variants] + DOMAIN_TOPICS)
        return self._domain

    def _topic_anchors(self) -> np.ndarray:
        """Anchors for the off-topic check. FAQ-style question headings are left out:
        "Do you price match?" would make "who won the cricket match?" look on-topic."""
        n = len(self._variants)
        topics = [i for i, (_, v) in enumerate(self._variants) if not v.endswith("?")]
        return self._anchors()[topics + list(range(n, n + len(DOMAIN_TOPICS)))]

    def domain_score(self, question: str) -> float:
        """How close the question is to anything this business handles (0..1).

        Compares against DOMAIN_TOPICS plus every document section heading, so
        "do you price match?" scores high (on-topic, just not answered yet)
        while "who won the cricket match?" scores low (off-topic).
        """
        return float((self._topic_anchors() @ self.embed([question])[0]).max())

    def matching_sections(self, question: str) -> list[tuple[str, float]]:
        """Section headings ranked by similarity to the question (best variant counts)."""
        if not self._variants:
            return []
        scores = self._anchors()[: len(self._variants)] @ self.embed([question])[0]
        best: dict[str, float] = {}
        for (section, _), score in zip(self._variants, scores.tolist()):
            best[section] = max(best.get(section, -1.0), score)
        return sorted(best.items(), key=lambda x: -x[1])

    def add_document(self, name: str, pages: list[tuple[int, str]], checksum: str | None = None) -> dict:
        records = [
            (page, section, text)
            for page, page_text in pages
            for section, text in chunk_page(page_text)
        ]
        if not records:
            raise ValueError("No readable text found in this document.")
        vectors = self.embed([_with_section(s, t) for _, s, t in records])
        with db.get_conn() as conn:
            doc_id = conn.execute("INSERT INTO documents (name, checksum) VALUES (?, ?)", (name, checksum)).lastrowid
            conn.executemany(
                "INSERT INTO chunks (document_id, page, section, position, text, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (doc_id, page, section, i, text, vec.tobytes())
                    for i, ((page, section, text), vec) in enumerate(zip(records, vectors))
                ],
            )
        self.reload()
        return {"id": doc_id, "name": name, "chunks": len(records)}

    def delete_document(self, doc_id: int):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.reload()

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        if self.matrix is None:
            return []
        scores = self.matrix @ self.embed([query])[0]
        top = np.argsort(-scores)[:k]
        return [{**self.meta[i], "score": float(scores[i])} for i in top]

    def best_sentence(self, question: str, hits: list[dict]) -> tuple[str, dict, float]:
        """The sentence closest in meaning to the question, across all retrieved chunks.

        If the runner-up sentence is from the same chunk and also relevant, both are
        quoted in their original order ("...30 days... Sale items ... 15 days").
        """
        candidates = [(i, s, h) for h in hits for i, s in enumerate(split_sentences(h["text"]))]
        scores = self.embed([s for _, s, _ in candidates]) @ self.embed([question])[0]
        order = np.argsort(-scores)
        best = order[0]
        _, sentence, hit = candidates[best]
        if len(order) > 1:
            second = order[1]
            if candidates[second][2] is hit and scores[second] >= 0.5:
                pair = sorted([candidates[best], candidates[second]], key=lambda c: c[0])
                sentence = " ".join(s for _, s, _ in pair)
        return sentence, hit, float(scores[best])

    def _source(self, h: dict, score: float, used: bool) -> dict:
        return {"document": h["name"], "page": h["page"], "section": display_section(h["section"]),
                "similarity": round(score, 3), "snippet": h["text"][:300], "used": used}

    def faq_answer(self, question: str, any_heading: bool = False) -> dict | None:
        """Help-centre FAQ match: the question is close to one of our FAQ headings.

        Comparing question to question ("do you sell gift cards?" vs "Can I buy a
        gift card?") is far more reliable than extractive QA for yes/no questions.
        With any_heading, a very close plain heading counts too ("what is ShopEase
        Plus?" -> "ShopEase Plus Membership"). That is only safe after order actions
        have been ruled out, because "cancel my order" is also close to "Order Cancellation".
        """
        matches = [(s, score) for s, score in self.matching_sections(question)
                   if score >= (FAQ_MATCH_THRESHOLD if display_section(s).endswith("?") else
                                HEADING_MATCH_THRESHOLD if any_heading else 2)]
        if not matches:
            return None
        section, score = matches[0]
        chunk = next(m for m in self.meta if m["section"] == section)
        answer = " ".join(split_sentences(chunk["text"])[:2])
        return {"kind": "rag", "answer": answer, "highlight": None, "match": score,
                "confidence": round(score, 3), "sources": [self._source(chunk, score, True)]}

    def answer(self, question: str, hits: list[dict]) -> dict | None:
        """Answer from the help centre, or None when it doesn't cover the question.

        kind="rag":     an FAQ heading matches, or the QA model found a confident answer span.
        kind="passage": the right section was found but the reader is unsure,
                        so we quote the most relevant sentence instead.
        """
        faq = self.faq_answer(question, any_heading=True)
        if faq:
            return faq
        hits = [h for h in hits if h["score"] >= RETRIEVAL_THRESHOLD]
        if not hits:
            return None

        best = None
        for hit in hits:
            result = self.qa(question=question, context=_with_section(hit["section"], hit["text"]))
            # Blend reader confidence with retrieval similarity so a confident
            # span from an off-topic chunk doesn't win.
            score = result["score"] * hit["score"]
            if best is None or score > best[0]:
                best = (score, result, hit)

        score, result, hit = best
        span = result["answer"].strip()
        match = None
        if result["score"] >= QA_THRESHOLD and span:
            kind, confidence = "rag", result["score"]
            sentence = next((s for s in split_sentences(hit["text"]) if span in s), span)
        else:
            kind, span = "passage", None
            sentence, hit, match = self.best_sentence(question, hits)
            confidence = hit["score"]

        return {
            "kind": kind,
            "answer": sentence,
            "highlight": span,
            "match": match,  # passage only: how closely the quoted sentence matches the question
            "confidence": round(float(confidence), 3),
            "sources": [self._source(h, h["score"], h is hit) for h in hits],
        }
