# ShopEase Support Bot: an intelligent customer-support and FAQ chatbot

A customer-support chatbot for an online store that can:

1. **Understand** what customers want: a classifier trained on ~27,000 support messages (27 intents), plus typo correction and synonyms
2. **Act**: track, cancel and refund orders, send invoices and change addresses on a real (demo) database, asking for missing details and confirming before changing anything
3. **Answer from knowledge**: a 40-question help centre and six policy documents (with citations), plus a 28-product catalog with prices, stock, specs and Q&As
4. **Remember context**: "does **it** have GPS?", "what about sale items?", order numbers given in a later message
5. **Know its limits**: says "I don't know" instead of guessing, offers a person when unsure, detects **frustrated customers** and escalates them
6. **Learn**: unanswered questions are clustered into knowledge gaps; an admin answers once and the bot knows it from then on

**Stack:** Python · NLTK (preprocessing, WordNet, VADER sentiment) · scikit-learn · Hugging Face Transformers / sentence-transformers · FastAPI · SQLite · vanilla HTML/JS

## Example conversation
```
User: How much are the wireless earbuds?
Bot:  The Wireless Earbuds cost ₹2,499 (including GST).               [Product catalog]
User: do they come in white?
Bot:  Yes, the Wireless Earbuds come in White. All colours: Black, White.
User: Does the bluetooth speaker support Alexa?
Bot:  I don't have that detail for the Bluetooth Speaker in our product
      information, and I'd rather not guess. Leave your email and our
      product team will confirm.                                       [email box]
User: I want to cancel my order
Bot:  I can help cancel it. What's your order number?                   [Order system]
User: 10023
Bot:  Order #10023 (...) hasn't shipped yet. Do you want me to cancel it? [Yes] [No]
User: yes
Bot:  Done. Order #10023 is cancelled. A refund of ₹... to your UPI will arrive in 5–7 business days.
User: this is useless!!!
Bot:  I'm really sorry this has been frustrating. Would you like to talk to a person? ...  [Escalated]
```

## How a message is answered

```
user message
   │
   ▼
0. Spell-correct (bot's own vocabulary)  +  score frustration (VADER sentiment, angry words, CAPS, repeats)
   │
   ▼
1. Continue   finishing an action? (waiting for order no. / yes-no / new address)
2. FAQ        small talk + admin-taught answers (TF-IDF + WordNet synonyms)
3. Help-centre FAQ match  question ≈ an FAQ heading (embeddings, ≥ 0.75) ─► direct answer + citation
4. Action     trained classifier says track/cancel/refund/invoice/address, or an order number is given ─► database
5. Product    product mentioned (or "it") ─► exact fact / spec / Q&A / product-page reading / search
6. Intent     other trained intents: policy topics from documents, or a fixed reply (+ ticket)
7. RAG        embed → top-3 chunks → extractive QA (with "no answer" detection) ─► sentence + citation
8. Passage    right area, no exact answer ─► "closest thing our help centre says" + ticket
9. Off-topic  unrelated to the business ─► polite redirect
10. Clarify   near misses ─► "Did you mean…?" buttons
11. Gap       on-topic but unknown ─► logged + support ticket ─► admin clusters & teaches
 +  Escalation  a frustrated customer is offered a person, whatever the route
```

## What makes it reliable

| Principle | How |
|---|---|
| **Facts are never generated** | Prices, stock, specs and order data come straight from the database. Policy answers are quoted from documents. No text-generating model is used. |
| **Honest when unsure** | The QA model's "no answer" score is used; loose matches are labelled "the closest thing…"; unknown specs get "I'd rather not guess" + ticket |
| **The right tool per question** | Exact lookups for facts, question-to-question matching for FAQs, extractive QA for document details, classifier for intents, rules only where ML is weak (order numbers) |
| **Measured** | Unit/API tests + an end-to-end evaluation set + a held-out set never used for tuning |

## Evaluation (be honest about these in a report)

| Test | Result | What it means |
|---|---|---|
| Intent classifier, Bitext's own 20% test split | **99.8%** | Inflated: Bitext is template-generated, test ≈ train |
| Intent classifier, 40 hand-written questions | **80%** | Realistic for the classifier alone |
| Whole bot, 103-message evaluation set (`data/eval_set.csv`) | **100%** | The bot was improved *while looking at* these, so this is optimistic |
| Whole bot, 44-message **held-out** set (`data/eval_holdout.csv`) | **75%** | Never used for tuning → the realistic estimate for new customers |

Main error types on the held-out set: classifier confusions on unusual wordings ("can I use two coupons together" → account), product-name collisions ("DSLR cameras" → our Webcam's alias "camera"), and slot filling treating a typed address as a new question. These are good "future work" points.

```bash
python -m app.evaluate              # 103-message set, saved for the admin dashboard
python -m app.evaluate --holdout    # held-out set
python -m app.intent_model          # retrain + evaluate the classifier (~20 s)
```

## The knowledge
| File | What | Edit it to… |
|---|---|---|
| `data/docs/*.md` | Help centre: returns, damaged items, shipping, payments, orders & account, warranty, 40 FAQs | change policies (the bot re-indexes new files at start-up, or upload in admin) |
| `data/products.json` | 28 products: price, stock, colours, sizes, specs, Q&As, aliases | add products or specs |
| `data/intent_responses.json` | How each trained intent is handled + its reply | change replies |
| `data/faqs.json` | Small talk (hi, thanks, hours, "are you a bot?") | add greetings |

FAQ headings can list alternative phrasings separated by `|`:
```
## What if my order arrived damaged or broken? | Can I return a product that arrived damaged?
```

## Demo orders
40 fake orders (10001–10040) are seeded on first start. Fixed ones for demos:

| Order | State | Try |
|---|---|---|
| 10023 | processing | cancel it, change its address |
| 10027 | shipped | track it, try to cancel it |
| 10040 | out for delivery | track it |
| 10031 | delivered 10 days ago | return it for a refund, get the invoice |
| 10005 | delivered 50+ days ago | refund is refused (outside the 30-day window) |
| 10012 | returned | refund status |

> A real store would check that the customer is logged in before showing or changing an order. This demo trusts the order number.

## Run it locally
Use a **short folder path** (Windows' 260-character limit breaks the PyTorch install in deep folders).

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
- Chat: http://127.0.0.1:8000 (the sidebar has sample questions and the demo orders)
- Admin dashboard: http://127.0.0.1:8000/admin
- API docs: http://127.0.0.1:8000/docs

On first start the app downloads NLTK data into `nltk_data/` and two Hugging Face models (~550 MB, cached), trains the intent classifier if `models/` is empty (~20 s), and seeds the help centre, products and orders. Delete `chatbot.db` to reset all data.

## Deploy (free) to Hugging Face Spaces
1. Create a Space → SDK **Docker**.
2. Push this folder. The `Dockerfile` bakes in the models, NLTK data and the trained classifier, and listens on port 7860.

Or anywhere with Docker: `docker build -t chatbot .` then `docker run -p 7860:7860 chatbot`. Spaces storage resets on restart; point `CHATBOT_DB` at persistent storage for a permanent deployment.

## Tests
```bash
pytest -q
```
32 end-to-end tests: every route, order flows, product facts / follow-ups / comparisons / search / unknown specs, frustration escalation (and *not* escalating calm customers), knowledge-gap clustering and teaching, tickets.

## Project layout
```
app/
  chatbot.py       the router: decides how every message is answered
  intent_model.py  trains / loads the 27-intent classifier
  actions.py       order actions: track, cancel, refund, invoice, address
  store.py         demo store database: customers, products, orders, refunds
  products.py      product catalog: facts, specs, Q&A, search, honest unknowns
  rag.py           help-centre search: FAQ matching, chunking, embeddings, extractive QA, domain detector
  frustration.py   frustration detection and escalation (VADER + support-specific signals)
  intent.py        TF-IDF FAQ matcher (small talk + taught answers)
  spelling.py      domain-aware spell correction
  gaps.py          clustering unanswered questions into knowledge gaps
  evaluate.py      end-to-end evaluation
  nlp.py           NLTK preprocessing
  db.py            SQLite: faqs, documents, chunks, logs, tickets
  loaders.py · config.py · main.py
data/            help centre, products, intents, training & evaluation data
static/          chat page, admin dashboard
tests/           end-to-end tests
```
