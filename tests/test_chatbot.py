"""End-to-end tests through the HTTP API. The first run downloads the models (~550 MB)."""
import os
import tempfile

os.environ["CHATBOT_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag import chunk_page


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def ask(client, message, session_id="test"):
    r = client.post("/api/chat", json={"message": message, "session_id": session_id})
    assert r.status_code == 200
    return r.json()


def test_chunker_respects_headings():
    chunks = chunk_page("# A\nFirst sentence. Second sentence.\n## B\nThird sentence.")
    assert [c[0] for c in chunks] == ["A", "B"]


def test_faq_route(client):
    data = ask(client, "what time do you open?")
    assert data["route"] == "faq"
    assert data["intent"] == "opening_hours"


def test_rag_route_with_citation(client):
    data = ask(client, "How many days do I have to return an item?")
    assert data["route"] == "rag"
    assert "30 days" in data["reply"]
    cited = [s for s in data["sources"] if s["used"]]
    assert cited[0]["document"] in ("help_centre_faq.md", "returns_and_refunds.md")


def test_extractive_answer_is_highlighted(client):
    data = ask(client, "How long does express shipping take?", session_id="express")
    assert data["route"] == "rag" and "1 to 2 business days" in data["reply"]
    assert any(s["document"] == "shipping_and_delivery.md" for s in data["sources"] if s["used"])


def test_damaged_items_policy(client):
    data = ask(client, "is it possible to return damaged products, if it was damaged while arriving??", session_id="dmg")
    assert data["route"] == "rag" and "48 hours" in data["reply"]
    assert not data["escalated"]  # "damaged" sounds negative, but the customer isn't upset


def test_follow_up_uses_context(client):
    ask(client, "How many days to return an item?", session_id="ctx")
    data = ask(client, "what about sale items?", session_id="ctx")
    assert data["route"] in ("rag", "passage")
    assert "15" in data["reply"] or "store credit" in data["reply"]


def test_typos_are_corrected(client):
    data = ask(client, "refnd polcy for sale itms", session_id="typo")
    assert data["corrected"].startswith("refund")
    assert data["route"] in ("rag", "passage")
    assert "15 days" in data["reply"]


def test_unknown_brand_names_are_not_corrected(client):
    data = ask(client, "does the smartwatch support spotify", session_id="spotify")
    assert data["corrected"] is None  # not turned into "notify"
    assert data["route"] == "fallback" and "rather not guess" in data["reply"]


def test_passage_for_yes_no_question(client):
    data = ask(client, "Can I pay with UPI?", session_id="upi")
    assert data["route"] in ("rag", "passage")
    assert "UPI" in data["reply"]


def test_clarify_offers_suggestions(client):
    data = ask(client, "When do you shut?", session_id="shut")
    assert data["route"] == "clarify"
    assert data["suggestions"][0]["label"] == "Opening hours"


def test_off_topic_is_redirected(client):
    data = ask(client, "who won the cricket match?", session_id="cricket")
    assert data["route"] == "offtopic"


def test_gap_clustering_and_teaching(client):
    questions = ["do you use eco friendly packaging", "is your packaging plastic free?", "is the packaging recyclable"]
    for q in questions:
        assert ask(client, q, session_id="gap")["route"] in ("fallback", "passage")  # not (fully) answered

    gaps = client.get("/api/gaps").json()
    eco = next(g for g in gaps if any("packaging" in q for q in g["questions"]))
    assert eco["count"] == 3  # three phrasings grouped into one gap

    r = client.post("/api/gaps/resolve", json={
        "intent": "eco_packaging", "patterns": eco["questions"], "answer": "Yes, our packaging is 100% recyclable.",
    })
    assert r.status_code == 200
    # A phrasing the bot has never seen is now answered from the new FAQ
    data = ask(client, "is your packaging eco friendly", session_id="gap2")
    assert data["route"] == "faq" and data["intent"] == "eco_packaging"
    assert all(not any("packaging" in q for q in g["questions"]) for g in client.get("/api/gaps").json())


def test_tickets(client):
    bad = client.post("/api/tickets", json={"question": "x", "email": "not-an-email"})
    assert bad.status_code == 422
    t = client.post("/api/tickets", json={"question": "loyalty program?", "email": "a@b.co"}).json()
    assert client.get("/api/stats").json()["open_tickets"] >= 1
    client.post(f"/api/tickets/{t['id']}/close")
    closed = next(x for x in client.get("/api/tickets").json() if x["id"] == t["id"])
    assert closed["status"] == "closed"


# ---- trained intents and order actions ---------------------------------------

def test_model_metrics(client):
    m = client.get("/api/model").json()
    assert m["intents"] == 27
    assert m["accuracy"] > 0.95
    assert m["real_world_accuracy"] >= 0.7


def test_track_order_with_number(client):
    data = ask(client, "where's my order 10027", session_id="track")
    assert data["route"] == "action" and data["intent"] == "track"
    assert "shipped" in data["reply"] and "#10027" in data["reply"]


def test_slot_filling_asks_for_order_number(client):
    data = ask(client, "wheres my package", session_id="slot")
    assert data["route"] == "action" and "order number" in data["reply"]
    data = ask(client, "10040", session_id="slot")
    assert "out for delivery" in data["reply"]


def test_change_address_flow(client):
    ask(client, "I want to change my delivery address", session_id="addr")
    data = ask(client, "10023", session_id="addr")
    assert "new delivery address" in data["reply"]
    data = ask(client, "9 Lake View Road, Pune 411001", session_id="addr")
    assert "Pune" in data["reply"]


def test_cancel_needs_confirmation(client):
    data = ask(client, "cancel order 10023", session_id="cancel")
    assert data["route"] == "action" and data["suggestions"][0]["query"] == "yes"
    data = ask(client, "yes", session_id="cancel")
    assert "is cancelled" in data["reply"] and "refund" in data["reply"].lower()
    assert "was cancelled" in ask(client, "track order 10023", session_id="cancel2")["reply"]


def test_cannot_cancel_shipped_order(client):
    data = ask(client, "cancel order 10027", session_id="shipped")
    assert "already shipped" in data["reply"]


def test_refund_within_and_outside_window(client):
    data = ask(client, "I want a refund for order 10031", session_id="refund")
    assert "return window" in data["reply"]
    assert "pick it up" in ask(client, "yes", session_id="refund")["reply"]
    assert "isn't eligible" in ask(client, "refund for 10005", session_id="refund2")["reply"]


def test_invoice(client):
    data = ask(client, "can I get the invoice for 10031", session_id="inv")
    assert data["intent"] == "invoice" and "Total: ₹" in data["reply"]


def test_unknown_order_number(client):
    assert "couldn't find order #99999" in ask(client, "where is order 99999", session_id="x1")["reply"]


def test_classifier_template_offers_ticket(client):
    data = ask(client, "my card was charged twice", session_id="pay")
    assert data["route"] == "intent" and data["intent"] == "payment_issue" and data["offer_ticket"]


def test_bare_yes_without_context(client):
    assert ask(client, "yes", session_id="lonely")["route"] == "clarify"


# ---- product catalog -----------------------------------------------------------

def test_product_price_and_followups(client):
    data = ask(client, "How much are the wireless earbuds?", session_id="prod")
    assert data["route"] == "product" and "₹2,499" in data["reply"]
    assert data["products"][0]["name"] == "Wireless Earbuds"
    assert "White" in ask(client, "do they come in white?", session_id="prod")["reply"]  # "they" = earbuds
    assert "IPX5" in ask(client, "are they waterproof?", session_id="prod")["reply"]


def test_product_stock_and_sizes(client):
    assert "out of stock" in ask(client, "is the phone case in stock?", session_id="stock")["reply"]
    data = ask(client, "running shoes available in UK 9?", session_id="size")
    assert "are available in UK 9" in data["reply"]


def test_product_comparison(client):
    data = ask(client, "earbuds vs headphones battery", session_id="cmp")
    assert "6 hours" in data["reply"] and "50 hours" in data["reply"]
    assert len(data["products"]) == 2


def test_product_qa_and_unknown_spec(client):
    assert "not reply" in ask(client, "can I reply to messages from the smartwatch?", session_id="qa")["reply"]
    data = ask(client, "Does the bluetooth speaker support Alexa?", session_id="alexa")
    assert data["route"] == "fallback" and data["offer_ticket"]  # honest, never invents a feature


def test_product_search(client):
    data = ask(client, "show me audio products under 3000", session_id="search")
    assert data["route"] == "product" and data["products"]
    assert all(p["category"] == "Audio" and p["price"] <= 3000 for p in data["products"])
    assert "don't" in ask(client, "do you sell televisions?", session_id="tv")["reply"]


# ---- honesty and escalation -------------------------------------------------------

def test_frustration_escalates_to_a_human(client):
    ask(client, "where is my order", session_id="angry")
    data = ask(client, "this is useless!!!", session_id="angry")
    assert data["escalated"] and data["offer_ticket"]
    assert "talk to a person" in data["reply"]
    assert client.get("/api/stats").json()["escalations"] >= 1


def test_calm_customer_is_not_escalated(client):
    for q in ["where is order 10027", "thanks", "how much is the yoga mat"]:
        assert not ask(client, q, session_id="calm")["escalated"]


def test_interactions_are_logged(client):
    logs = client.get("/api/logs?limit=200").json()
    assert {"faq", "rag", "clarify", "offtopic", "fallback", "action", "intent", "product"} <= {l["route"] for l in logs}
