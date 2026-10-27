"""End-to-end evaluation of the whole bot on realistic customer conversations.

data/eval_set.csv lists questions (grouped into conversations, so follow-ups
and slot filling are tested too), the route(s) that count as handling it
correctly, a phrase the reply must contain, and optionally whether the turn
should be escalated. A turn passes only if all of these hold.

Run it (uses a fresh temporary database, so your real data is untouched):
    python -m app.evaluate
"""
import os
import tempfile

os.environ["CHATBOT_DB"] = os.path.join(tempfile.mkdtemp(), "evaluation.db")  # before importing app modules

import csv  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402

from . import db, store  # noqa: E402
from .config import BASE_DIR, DATA_DIR  # noqa: E402

EVAL_PATH = DATA_DIR / "eval_set.csv"
RESULT_PATH = BASE_DIR / "models" / "evaluation.json"


def run(path=EVAL_PATH, verbose: bool = True, save: bool = True) -> dict:
    from .chatbot import Chatbot
    from .main import seed
    from . import main

    db.init_db()
    store.init_store()
    main.bot = bot = Chatbot()
    seed()

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_route = defaultdict(lambda: {"total": 0, "correct": 0})
    failures = []
    for row in rows:
        reply = bot.respond(f"eval-{row['conversation']}", row["question"])
        allowed = row["expected_route"].split("|")
        must = row["must_contain"].strip()
        checks = {
            "route": reply["route"] in allowed,
            "content": not must or must.lower() in reply["reply"].lower(),
            "escalation": not row["escalated"] or reply["escalated"] == (row["escalated"] == "yes"),
        }
        ok = all(checks.values())
        bucket = by_route[allowed[0]]
        bucket["total"] += 1
        bucket["correct"] += ok
        if not ok:
            failures.append({"question": row["question"], "expected": row["expected_route"], "got": reply["route"],
                             "failed": [k for k, v in checks.items() if not v], "reply": reply["reply"][:120]})

    correct = sum(b["correct"] for b in by_route.values())
    result = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4),
        "by_route": {r: {**b, "accuracy": round(b["correct"] / b["total"], 4)} for r, b in by_route.items()},
        "failures": failures,
    }
    if save:
        RESULT_PATH.parent.mkdir(exist_ok=True)
        RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"\n{'Expected route':<16}{'Questions':>10}{'Correct':>9}{'Accuracy':>10}")
        for route, b in sorted(by_route.items(), key=lambda x: -x[1]["total"]):
            print(f"{route:<16}{b['total']:>10}{b['correct']:>9}{b['correct'] / b['total']:>10.0%}")
        print(f"{'TOTAL':<16}{len(rows):>10}{correct:>9}{correct / len(rows):>10.1%}\n")
        for f in failures:
            print(f"FAIL {f['question']!r}: expected {f['expected']}, got {f['got']} ({', '.join(f['failed'])})\n"
                  f"     {f['reply']!r}")
    return result


HOLDOUT_PATH = DATA_DIR / "eval_holdout.csv"

if __name__ == "__main__":
    import sys

    if "--holdout" in sys.argv:
        # Questions never used while tuning the bot: the honest real-world estimate.
        held = run(HOLDOUT_PATH, save=False)
        if RESULT_PATH.exists():
            saved = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
            saved["holdout"] = {k: held[k] for k in ("total", "correct", "accuracy", "by_route")}
            RESULT_PATH.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        run()
