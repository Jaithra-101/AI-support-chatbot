"""Central settings. Tweak thresholds here to change how queries are routed."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = Path(os.getenv("CHATBOT_DB", BASE_DIR / "chatbot.db"))

# Models (downloaded from Hugging Face on first use, then cached)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "deepset/roberta-base-squad2"

# Routing thresholds
INTENT_THRESHOLD = 0.5      # TF-IDF cosine needed to answer from the FAQ table
CLASSIFIER_THRESHOLD = 0.6  # trained-classifier probability needed to trust its intent
RETRIEVAL_THRESHOLD = 0.30  # embedding cosine needed for a chunk to count as relevant
QA_THRESHOLD = 0.15         # extractive-QA confidence needed to trust an answer span
PASSAGE_MIN_MATCH = 0.35    # below this the "closest sentence" isn't related enough to quote
FAQ_MATCH_THRESHOLD = 0.68  # question vs help-centre FAQ heading similarity for a direct answer
HEADING_MATCH_THRESHOLD = 0.8  # same for plain section headings ("what is ShopEase Plus?" -> "ShopEase Plus Membership")

# Fallback ladder (used when neither FAQ nor RAG is confident)
FAQ_SUGGEST_THRESHOLD = 0.25   # FAQ near-miss worth offering as "did you mean"
DOC_SUGGEST_THRESHOLD = 0.45   # question vs section *heading* similarity worth offering
DOMAIN_THRESHOLD = 0.25        # below this the question is off-topic for the business
GAP_CLUSTER_DISTANCE = 0.5     # cosine distance for grouping unanswered questions

# What this bot is about. Used to tell "off-topic" apart from "on-topic but
# not covered yet". Document section headings are added automatically.
DOMAIN_TOPICS = [
    "online shopping orders and delivery",
    "payments, prices, discounts and offers",
    "returns, refunds and exchanges",
    "product warranty and repairs",
    "my account, login and customer support",
    "shipping and tracking a parcel",
    "cancelling or changing an order",
    "product availability and stock",
    "product features, specifications, price and colours",
    "offers, rewards, referrals, coupons and memberships",
    "store opening hours and closing time",
    "contacting customer care by phone or email",
]

# Retrieval
TOP_K = 3
CHUNK_WORDS = 120           # target chunk size, sentence-aligned
HISTORY_TURNS = 3           # turns kept per session for follow-up questions
