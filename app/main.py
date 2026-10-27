"""FastAPI entry point. Run with:  uvicorn app.main:app --reload"""
import hashlib
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db, store
from .chatbot import Chatbot
from .config import BASE_DIR, DATA_DIR
from .gaps import find_gaps, suggest_intent_name
from .loaders import load_pages

bot: Chatbot | None = None


def seed():
    """Load sample FAQs and documents the first time the database is created."""
    if not db.list_faqs():
        for faq in json.loads((DATA_DIR / "faqs.json").read_text(encoding="utf-8")):
            db.upsert_faq(faq["intent"], faq["patterns"], faq["answer"])
    with db.get_conn() as conn:
        indexed = {r["name"]: (r["id"], r["checksum"]) for r in conn.execute("SELECT id, name, checksum FROM documents")}
    if "shopease_policies.md" in indexed:  # replaced by the help-centre documents in this version
        bot.docs.delete_document(indexed.pop("shopease_policies.md")[0])
    for path in sorted((DATA_DIR / "docs").glob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        checksum = hashlib.sha1(data).hexdigest()
        if path.name in indexed:
            doc_id, old = indexed[path.name]
            if old == checksum:
                continue
            bot.docs.delete_document(doc_id)  # the file was edited: index the new version
        bot.docs.add_document(path.name, load_pages(path.name, data), checksum)
    bot.refresh()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global bot
    db.init_db()
    store.init_store()
    bot = Chatbot()
    seed()
    yield


app = FastAPI(title="Hybrid FAQ + RAG Chatbot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = None


class FaqIn(BaseModel):
    intent: str = Field(min_length=1)
    patterns: list[str] = Field(min_length=1)
    answer: str = Field(min_length=1)


class TicketIn(BaseModel):
    question: str = Field(min_length=1)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    session_id: str | None = None


# ---- pages ----------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/admin")
def admin():
    return FileResponse(BASE_DIR / "static" / "admin.html")


# ---- chat -----------------------------------------------------------------

@app.post("/api/chat")
def chat(body: ChatIn):
    session_id = body.session_id or uuid.uuid4().hex
    return {"session_id": session_id, **bot.respond(session_id, body.message.strip())}


@app.post("/api/tickets")
def create_ticket(body: TicketIn):
    return {"id": db.create_ticket(body.session_id, body.question, body.email)}


# ---- knowledge base ---------------------------------------------------------

@app.get("/api/documents")
def documents():
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT d.id, d.name, d.uploaded_at, COUNT(c.id) AS chunks "
            "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id GROUP BY d.id ORDER BY d.id"
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    try:
        pages = load_pages(file.filename, await file.read())
        result = bot.docs.add_document(file.filename, pages)
    except ValueError as e:
        raise HTTPException(400, str(e))
    bot.refresh()
    return result


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int):
    bot.docs.delete_document(doc_id)
    bot.refresh()
    return {"deleted": doc_id}


@app.get("/api/faqs")
def faqs():
    return db.list_faqs()


@app.post("/api/faqs")
def add_faq(body: FaqIn):
    db.upsert_faq(body.intent, body.patterns, body.answer)
    bot.refresh()
    return {"saved": body.intent}


# ---- admin: self-learning ---------------------------------------------------

@app.get("/api/gaps")
def gaps():
    return [{**g, "suggested_intent": suggest_intent_name(g["label"])} for g in find_gaps(bot.docs.embed)]


@app.post("/api/gaps/resolve")
def resolve_gap(body: FaqIn):
    """Teach the bot: the gap's questions become a new FAQ with the admin's answer."""
    existing = next((f for f in db.list_faqs() if f["intent"] == body.intent), None)
    patterns = list(dict.fromkeys((existing["patterns"] if existing else []) + body.patterns))
    db.upsert_faq(body.intent, patterns, body.answer)
    db.mark_resolved(body.patterns)
    bot.refresh()
    return {"saved": body.intent, "patterns": len(patterns)}


@app.get("/api/tickets")
def tickets():
    return db.list_tickets()


@app.post("/api/tickets/{ticket_id}/close")
def close_ticket(ticket_id: int):
    db.close_ticket(ticket_id)
    return {"closed": ticket_id}


@app.get("/api/orders")
def orders(limit: int = 15):
    """Demo orders, so you know which order numbers to try in the chat."""
    return store.recent_orders(limit)


@app.get("/api/products")
def products():
    return [bot.products.card(p) for p in bot.products.products]


@app.get("/api/evaluation")
def evaluation():
    """Results of the last `python -m app.evaluate` run."""
    path = BASE_DIR / "models" / "evaluation.json"
    return json.loads(path.read_text()) if path.exists() else {}


@app.get("/api/model")
def model():
    """Training metrics of the intent classifier."""
    return bot.classifier.metrics


@app.get("/api/logs")
def logs(limit: int = 50):
    return db.recent_logs(limit)


@app.get("/api/stats")
def stats():
    return db.stats()
