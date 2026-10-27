"""Router: decides how every message is answered.

    0. fix typos against the bot's own vocabulary; score customer frustration
    1. continue    finishing an order action (waiting for an order number / yes / address)
    2. faq         small talk and admin-taught answers (TF-IDF over FAQ phrasings)
    3. action      track / cancel / refund / invoice / address on the order database
                   (trained 27-intent classifier, or an order number in the message)
    4. product     facts, specs, Q&A and search from the product catalog
    5. intent      other trained intents: policy topics from documents, or a fixed reply
    6. rag         QA model extracts a confident answer from the help centre
    7. passage     right area found but no exact answer -> honest "closest thing" + ticket
    8. offtopic    unrelated to the business -> redirect
    9. clarify     near misses -> "did you mean...?" buttons
   10. fallback    on-topic but unknown -> log the gap, offer a ticket
    +  escalation  a frustrated customer is offered a person, whatever the route
"""
import json
import re
from collections import defaultdict, deque

from . import actions, db
from .config import (CLASSIFIER_THRESHOLD, DATA_DIR, DOC_SUGGEST_THRESHOLD, DOMAIN_THRESHOLD,
                     FAQ_SUGGEST_THRESHOLD, HISTORY_TURNS, INTENT_THRESHOLD, PASSAGE_MIN_MATCH)
from .frustration import FrustrationDetector
from .intent import IntentMatcher
from .intent_model import IntentClassifier
from .products import PRODUCT_QUESTION, PRONOUN, ProductCatalog
from .rag import DocumentStore, display_section
from .spelling import SpellCorrector

# Never offered as "did you mean" suggestions
SMALL_TALK = {"greeting", "goodbye", "thanks", "bot_identity"}
# Trained intents whose reply apologises: only use them when the customer actually sounds unhappy
UNHAPPY_INTENTS = {"complaint"}
# Questions that lean on the previous turn ("what about sale items?", "does it ship free?")
_FOLLOW_UP = re.compile(r"\b(it|its|that|this|those|these|they|them|there|what about|how about)\b|^\s*and\b", re.I)
_SELL = re.compile(r"\bdo you (have|sell|stock|carry)\b", re.I)
_SELL_ONLY = re.compile(r"\bdo you (sell|stock|carry)\b", re.I)  # clearly about products, unlike "do you have"
STRONG_FAQ = 0.75  # a help-centre FAQ this close beats the intent classifier
HUMAN_OFFER = ("Would you like to talk to a person? Leave your email below and an agent will reply within a "
               "few hours, or call 1800-123-4567 (Mon–Sat, 9 AM–8 PM).")


class Chatbot:
    def __init__(self):
        self.intents = IntentMatcher()
        self.classifier = IntentClassifier()
        self.intent_specs = {k: v for k, v in json.loads(
            (DATA_DIR / "intent_responses.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}
        self.docs = DocumentStore()
        self.products = ProductCatalog(self.docs.embed, self.docs.qa)
        self.frustration = FrustrationDetector(self.docs.embed)
        self.speller = SpellCorrector()
        self.history = defaultdict(lambda: deque(maxlen=HISTORY_TURNS))
        self.pending: dict[str, dict] = {}          # session -> action waiting for more info
        self.last_products: dict[str, list] = {}    # session -> products discussed last ("does it...?")
        self.refresh()

    def refresh(self):
        """Rebuild indexes after FAQs or documents change."""
        self.intents.reload()
        self.docs.reload()
        patterns = [p for f in db.list_faqs() for p in f["patterns"]]
        self.speller.rebuild(patterns + [m["text"] for m in self.docs.meta] + self.products.vocabulary())

    # ---- helpers ---------------------------------------------------------------

    def _retrieval_query(self, session_id: str, message: str) -> str:
        """Add the previous question to vague follow-ups so retrieval keeps context."""
        past = self.history[session_id]
        if past and (len(message.split()) <= 3 or _FOLLOW_UP.search(message)):
            return f"{past[-1]['user']} {message}"
        return message

    def _suggestions(self, text: str, ranked: list[tuple[str, float]]) -> list[dict]:
        """Near-miss FAQs and clearly related help-centre sections, as clickable questions."""
        out = []
        for intent, score in ranked[:2]:
            if score >= FAQ_SUGGEST_THRESHOLD and intent not in SMALL_TALK:
                out.append({"label": intent.replace("_", " ").capitalize(),
                            "query": self.intents.example(intent)})
        for section, score in self.docs.matching_sections(text)[:2]:
            if score >= DOC_SUGGEST_THRESHOLD:
                label = display_section(section)
                query = label if label.endswith("?") else f"Tell me about {label.lower()}"
                out.append({"label": label, "query": query})
        return out[:3]

    def _in_domain(self, text: str) -> bool:
        return self.docs.matrix is None or self.docs.domain_score(text) >= DOMAIN_THRESHOLD

    def _action(self, session_id: str, action: str, number: int | None, **kwargs) -> dict:
        result = actions.run(action, number, **kwargs)
        if result["pending"]:
            self.pending[session_id] = result["pending"]
        return {"route": "action", "reply": result["reply"], "suggestions": result["suggestions"],
                "intent": action, "confidence": 1.0, "order_id": number}

    def _from_docs(self, session_id: str, text: str, exact_only: bool = False) -> dict | None:
        hits = self.docs.search(self._retrieval_query(session_id, text))
        found = self.docs.answer(text, hits)
        if not found or (exact_only and found["kind"] != "rag"):
            return None
        if found["kind"] == "passage" and found["match"] < PASSAGE_MIN_MATCH:
            return None  # nothing relevant enough to quote: let the ladder clarify or log a gap
        if found["kind"] == "rag":
            return {"route": "rag", "reply": found["answer"], "highlight": found["highlight"],
                    "confidence": found["confidence"], "sources": found["sources"]}
        # No exact answer span: quote the closest sentence, and be honest about how close it is.
        if found["match"] >= 0.5:
            reply = f"Here's what our help centre says:\n\n\"{found['answer']}\""
        else:
            reply = (f"I couldn't find an exact answer to that. The closest thing in our help centre says:\n\n"
                     f"\"{found['answer']}\"\n\nIf that doesn't answer your question, leave your email "
                     f"and our team will get back to you.")
        return {"route": "passage", "confidence": found["confidence"], "sources": found["sources"],
                "offer_ticket": found["match"] < 0.5, "reply": reply}

    # ---- routing ---------------------------------------------------------------

    def _continue(self, session_id: str, message: str, number: int | None) -> dict | None:
        """Finish an action that was waiting for the user. None = user changed topic."""
        pending = self.pending.pop(session_id, None)
        if not pending:
            return None
        action, awaiting = pending["action"], pending["awaiting"]
        if actions.NO.match(message):
            order = f" order #{pending['order_id']}" if pending.get("order_id") else " that"
            return {"route": "action", "intent": action, "confidence": 1.0,
                    "reply": f"Okay, I've left{order} as it is. Anything else I can help with?"}
        if awaiting == "order_id" and number:
            return self._action(session_id, action, number)
        if awaiting == "confirm" and actions.YES.match(message):
            return self._action(session_id, action, pending["order_id"], confirmed=True)
        if awaiting == "address" and len(message.split()) >= 3 and "?" not in message:
            return self._action(session_id, action, pending["order_id"], new_address=message.strip())
        return None

    def _products(self, session_id: str, text: str, products: list[dict]) -> dict | None:
        if self.products.is_search(text) and (_SELL.search(text) or not products or len(products) > 1
                                              or re.search(r"\d", text)):
            found = self.products.search(text)
            if not products and found.get("weak"):
                if _SELL_ONLY.search(text):
                    return {"route": "product", "reply": "Sorry, we don't sell that at the moment.",
                            "products": [], "confidence": found["confidence"]}
                # "do you have a referral program?" is not a product search: let the rest of the ladder decide
                return None
            if not products and _SELL.search(text) and found["products"] and found["confidence"] < 0.45:
                found["reply"] = "We don't seem to sell exactly that. These are the closest items we have:"
            found.pop("weak", None)
            return {"route": "product", **found}
        if not products:
            return None
        self.last_products[session_id] = products
        found = self.products.answer(text, products)
        if not found:
            return None
        if found.pop("unknown", False):
            return {"route": "fallback", "offer_ticket": True, **found}
        return {"route": "product", **found}

    def _route(self, session_id: str, text: str, number: int | None, mood: dict) -> dict:
        ranked = self.intents.rank(text)
        faq, faq_score = ranked[0] if ranked else (None, 0.0)
        if faq_score >= INTENT_THRESHOLD:
            return {"route": "faq", "reply": self.intents.answers[faq], "intent": faq,
                    "confidence": round(faq_score, 3)}

        intent, prob = self.classifier.predict(text)
        spec = self.intent_specs.get(intent, {})
        if number is not None:
            # An order number was mentioned: trust a confident classifier, else clear keywords,
            # else just look the order up.
            action = spec.get("action") if prob >= 0.8 else None
            return self._action(session_id, action or actions.keyword_action(text) or spec.get("action", "track"),
                                number)
        if actions.YES.fullmatch(text.strip(" .!")) or actions.NO.fullmatch(text.strip(" .!")):
            return {"route": "clarify", "confidence": 0.0,
                    "reply": "Sorry, I'm not sure what you're answering. What would you like help with?",
                    "suggestions": [{"label": "Track an order", "query": "track my order"},
                                    {"label": "Cancel an order", "query": "cancel my order"},
                                    {"label": "Get a refund", "query": "I want a refund"}]}

        faq_doc = self.docs.faq_answer(text)
        if faq_doc and faq_doc["confidence"] >= STRONG_FAQ:
            # "my order arrived broken" ~ "What if my product arrived damaged?": a near-exact help-centre FAQ
            return {"route": "rag", "reply": faq_doc["answer"], "confidence": faq_doc["confidence"],
                    "sources": faq_doc["sources"]}

        products = self.products.find(text)
        if not products and self.last_products.get(session_id) and PRONOUN.search(text) \
                and (self.products.attributes(text) or PRODUCT_QUESTION.search(text)):
            products = self.last_products[session_id]  # "does it have GPS?" -> the product we were discussing
        spec_question = bool(products and self.products.attributes(text))

        # Order actions, unless this is clearly a spec question ("does it cancel noise?")
        if not spec_question and prob >= CLASSIFIER_THRESHOLD and spec.get("handler") == "action" \
                and self._in_domain(text):
            return self._action(session_id, spec["action"], None)

        if not products and _SELL.search(text) and faq_doc:  # "do you sell gift cards?" -> help-centre FAQ
            return {"route": "rag", "reply": faq_doc["answer"], "confidence": faq_doc["confidence"],
                    "sources": faq_doc["sources"]}
        found = self._products(session_id, text, products)
        if found:
            return found

        sounds_unhappy = mood["negative"] or mood["angry_words"] or mood["shouting"]
        # "what if the product was damaged by me?" can look like a complaint to the classifier,
        # but nobody is upset, so an apology would be wrong: answer it from the help centre instead
        trust_intent = intent not in UNHAPPY_INTENTS or sounds_unhappy
        if prob >= CLASSIFIER_THRESHOLD and spec and spec["handler"] != "action" and trust_intent \
                and self._in_domain(text):
            if spec["handler"] == "docs":
                found = self._from_docs(session_id, text)
                if found:
                    return {**found, "intent": intent}
            return {"route": "intent", "reply": spec["reply"], "intent": intent,
                    "confidence": round(prob, 3), "offer_ticket": spec.get("offer_ticket", False)}

        found = self._from_docs(session_id, text)
        if found:
            return found
        if not self._in_domain(text):
            return {"route": "offtopic", "confidence": 0.0,
                    "reply": "That's outside what I can help with. I'm here for your orders, refunds, deliveries, "
                             "payments, our products and your account."}
        if suggestions := self._suggestions(text, ranked):
            ask = "Did you mean this?" if len(suggestions) == 1 else \
                "I'm not quite sure what you mean. Are you asking about one of these?"
            return {"route": "clarify", "reply": ask, "suggestions": suggestions,
                    "confidence": round(max(faq_score, prob), 3)}
        return {"route": "fallback", "offer_ticket": True, "confidence": round(max(faq_score, prob), 3),
                "reply": "I don't have an answer for that yet, so I've noted it for our team. "
                         "Want a person to follow up? Leave your email below."}

    def respond(self, session_id: str, message: str) -> dict:
        text = self.speller.correct(message)
        number = actions.find_order_number(message)
        escalate, signals = self.frustration.update(session_id, message)

        result = self._continue(session_id, message, number) or self._route(session_id, text, number, signals)
        reply = {"intent": None, "highlight": None, "sources": [], "suggestions": [], "products": [],
                 "offer_ticket": False, "order_id": None, "escalated": False,
                 "corrected": text if text != message else None, **result}

        if escalate:
            reply["escalated"] = reply["offer_ticket"] = True
            if reply["route"] in ("passage", "fallback", "offtopic", "clarify"):
                # mostly venting, not a question we can answer: just apologise and offer a person
                reply.update(reply="I'm really sorry this has been frustrating. " + HUMAN_OFFER,
                             sources=[], suggestions=[], products=[])
            elif reply["intent"] not in ("complaint", "contact_human_agent"):  # those already apologise
                reply["reply"] = f"I'm sorry this has been frustrating. {reply['reply']}\n\n{HUMAN_OFFER}"
        reply["sentiment"] = signals["sentiment"]

        self.frustration.record_route(session_id, reply["route"])
        self.history[session_id].append({"user": text, "bot": reply["reply"]})
        db.log_interaction(session_id, message, reply["reply"], reply["route"], reply["intent"],
                           reply["confidence"], reply["sources"], reply["escalated"])
        return reply
