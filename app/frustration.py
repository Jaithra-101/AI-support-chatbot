"""Frustration detection: notice an upset customer and offer a human early.

Each message gets a score from several signals:
    +1 / +2  negative sentiment (NLTK VADER compound <= -0.4 / -0.7)
    +1       angry words ("useless", "worst", "scam", ...)
    +1       SHOUTING in capitals, or "!!!" / "?!"
    +1       repeating a question they already asked
    +1       the bot failed to answer twice in a row

The session level decays (half each turn) so one grumpy word doesn't trigger
it, but a run of bad turns does. At level >= 2 the bot apologises, offers a
person and a support ticket, and the turn is logged as escalated.
"""
import re
from collections import defaultdict, deque

from nltk.sentiment import SentimentIntensityAnalyzer

from . import nlp  # noqa: F401  (makes sure the VADER lexicon is downloaded)

ANGRY = re.compile(r"\b(useless|worst|ridiculous|pathetic|terrible|horrible|awful|scam|fraud|cheat(ed|ing)?|"
                   r"waste|angry|furious|frustrat\w*|annoy\w*|disgust\w*|unacceptable|stupid|hate|rubbish|"
                   r"fed up|not helpful|doesn'?t help|no help|already (told|said|asked)|again\?|ugh+|"
                   r"nothing works|not working|so hard|seriously)\b", re.I)
FAILED_ROUTES = {"fallback", "clarify", "passage", "offtopic"}
WEIGHTS = {"negative": 1.0, "angry_words": 1.5, "shouting": 1.5, "repeated": 1.0, "bot_failing": 1.0}
DECAY = 0.6
ESCALATE_AT = 2.0


class FrustrationDetector:
    def __init__(self, embed):
        self.embed = embed
        try:
            self.vader = SentimentIntensityAnalyzer()
        except LookupError:  # lexicon couldn't be downloaded: the other signals still work
            print("Warning: NLTK vader_lexicon missing, frustration detection runs without sentiment scores.")
            self.vader = None
        self.level = defaultdict(float)
        self.asked = defaultdict(lambda: deque(maxlen=4))   # recent questions per session
        self.routes = defaultdict(lambda: deque(maxlen=2))  # recent bot routes per session
        self.offered = defaultdict(lambda: 99)              # turns since we last offered a human

    def signals(self, session_id: str, message: str) -> dict:
        compound = self.vader.polarity_scores(message)["compound"] if self.vader else 0.0
        letters = [c for c in message if c.isalpha()]
        shouting = len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.6
        repeated = False
        past = list(self.asked[session_id])
        if past and len(message.split()) >= 3:
            vectors = self.embed(past + [message])
            repeated = bool((vectors[:-1] @ vectors[-1]).max() >= 0.85)
        return {
            "sentiment": compound,
            # capped at 1: words like "damaged" are negative without the customer being upset
            "negative": compound <= -0.4,
            "angry_words": bool(ANGRY.search(message)),
            "shouting": shouting or bool(re.search(r"!{3,}|\?!|!\?", message)),
            "repeated": repeated,
            "bot_failing": len(self.routes[session_id]) == 2 and set(self.routes[session_id]) <= FAILED_ROUTES,
        }

    def update(self, session_id: str, message: str) -> tuple[bool, dict]:
        """Score this message; return (should_escalate, signals)."""
        s = self.signals(session_id, message)
        score = sum(weight for name, weight in WEIGHTS.items() if s[name])
        self.level[session_id] = self.level[session_id] * DECAY + score
        self.asked[session_id].append(message)
        self.offered[session_id] += 1
        escalate = self.level[session_id] >= ESCALATE_AT and self.offered[session_id] > 2
        if escalate:
            self.offered[session_id] = 0  # don't repeat the offer every turn
            self.level[session_id] = 0.0
        s["level"] = round(self.level[session_id], 2)
        return escalate, s

    def record_route(self, session_id: str, route: str):
        self.routes[session_id].append(route)
