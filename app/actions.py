"""Order actions: the bot doing real customer-service work on the store database.

Each action needs an order number. If the user didn't give one, the bot asks
for it and remembers what it was doing (slot filling):

    user: I want to cancel my order          -> intent cancel_order, no number
    bot:  Sure - what's your order number?   -> pending = {action: cancel, awaiting: order_id}
    user: 10023                               -> number found, action resumes
    bot:  ...hasn't shipped yet. Cancel it?  -> pending = {awaiting: confirm}
    user: yes                                 -> order cancelled, refund created

Note: a real store would verify the customer is logged in before showing or
changing an order. This demo trusts the order number.
"""
import re
from datetime import date

from . import store

ORDER_NUMBER = re.compile(r"#?\b(\d{5})\b")
YES = re.compile(r"^\s*(yes|yeah|yep|y|sure|ok|okay|confirm|go ahead|do it|please do)\b", re.I)
NO = re.compile(r"^\s*(no|nope|n|cancel that|never ?mind|don't|keep it|stop)\b", re.I)

RETURN_DAYS, SALE_RETURN_DAYS = 30, 15

ASK_NUMBER = {
    "track": "Sure, I can check that. What's your order number? It's the 5-digit number in your confirmation email (e.g. 10023).",
    "cancel": "I can help cancel it. What's your order number? (5 digits, e.g. 10023)",
    "refund": "Let me look into that refund. What's your order number? (5 digits, e.g. 10031)",
    "invoice": "Sure. Which order do you need the invoice for? Please share the 5-digit order number.",
    "address": "I can update the delivery address if the order hasn't shipped. What's the order number?",
}


# Safety net when an order number is given but the classifier is unsure.
# (In the dataset "order <number>" mostly appears in cancel requests, so the
# model leans towards cancel_order whenever it sees a number.)
KEYWORDS = [
    ("cancel", re.compile(r"\b(cancel|stop|call off)\b", re.I)),
    ("refund", re.compile(r"\b(refund|return|money back|send (it )?back)\b", re.I)),
    ("invoice", re.compile(r"\b(invoice|bill|receipt)\b", re.I)),
    ("address", re.compile(r"\baddress\b", re.I)),
    ("track", re.compile(r"\b(where|track|status|arriv|deliver|ship)", re.I)),
]


def find_order_number(text: str) -> int | None:
    m = ORDER_NUMBER.search(text)
    return int(m.group(1)) if m else None


def keyword_action(text: str) -> str | None:
    return next((action for action, pattern in KEYWORDS if pattern.search(text)), None)


def _rupees(amount: int) -> str:
    return f"₹{amount:,}"


def _day(iso: str | None) -> str:
    return date.fromisoformat(iso).strftime("%a, %d %b") if iso else "-"


def _items(order: dict) -> str:
    return ", ".join(f"{i['name']}" + (f" ×{i['quantity']}" if i["quantity"] > 1 else "") for i in order["items"])


def _chip(label: str, query: str) -> dict:
    return {"label": label, "query": query}


def _reply(text: str, suggestions=None, pending=None, done=None) -> dict:
    return {"reply": text, "suggestions": suggestions or [], "pending": pending, "action_done": done}


def _refund_status(order: dict) -> str:
    r = order["refund"]
    if r["status"] == "completed":
        return f"Your refund of {_rupees(r['amount'])} to {r['method']} was completed. It should be in your account."
    if r["status"] == "requested":
        return (f"Your return is booked. A courier will pick it up within 3 business days, and "
                f"{_rupees(r['amount'])} will be refunded to {r['method']} after inspection "
                f"(expected by {_day(r['expected_by'])}).")
    return (f"Your refund of {_rupees(r['amount'])} to {r['method']} is being processed and should "
            f"reach you by {_day(r['expected_by'])}.")


# ---- the actions -------------------------------------------------------------

def track(order: dict) -> dict:
    n, s = order["id"], order["status"]
    head = f"Order #{n} ({_items(order)})"
    if s == "processing":
        return _reply(f"{head} is being packed and will be dispatched soon.\nExpected delivery: {_day(order['eta'])}.",
                      [_chip("Cancel this order", f"cancel order {n}"), _chip("Change address", f"change address for order {n}")])
    if s == "shipped":
        return _reply(f"{head} shipped on {_day(order['shipped_at'])} with {order['courier']} "
                      f"(tracking {order['tracking_number']}).\nExpected delivery: {_day(order['eta'])}.")
    if s == "out_for_delivery":
        return _reply(f"Good news! {head} is out for delivery and should reach you today. "
                      f"{order['courier']} tracking: {order['tracking_number']}.")
    if s == "delivered":
        return _reply(f"{head} was delivered on {_day(order['delivered_at'])} to {order['address']}.",
                      [_chip("Return for a refund", f"refund for order {n}"), _chip("Get invoice", f"invoice for order {n}")])
    if s == "cancelled":
        extra = f" {_refund_status(order)}" if order["refund"] else ""
        return _reply(f"{head} was cancelled.{extra}")
    # returned / return_requested
    extra = f" {_refund_status(order)}" if order["refund"] else ""
    return _reply(f"{head} was returned.{extra}")


def cancel(order: dict, confirmed: bool = False) -> dict:
    n, s = order["id"], order["status"]
    if s == "processing":
        if not confirmed:
            return _reply(
                f"Order #{n} ({_items(order)}, {_rupees(order['total'])}) hasn't shipped yet, so it can be "
                f"cancelled free of charge. Do you want me to cancel it?",
                [_chip("Yes, cancel it", "yes"), _chip("No, keep it", "no")],
                pending={"action": "cancel", "order_id": n, "awaiting": "confirm"})
        refund = store.cancel(n)
        money = (f" A refund of {_rupees(refund['amount'])} to your {refund['method']} will arrive within "
                 f"5–7 business days." if refund else " Nothing was charged since it was cash on delivery.")
        return _reply(f"Done. Order #{n} is cancelled.{money}", done="cancelled")
    if s in ("shipped", "out_for_delivery"):
        return _reply(f"Order #{n} has already shipped, so it can't be cancelled now. You can refuse the delivery, "
                      f"or request a return once it arrives.", [_chip("Track it", f"track order {n}")])
    if s == "cancelled":
        return _reply(f"Order #{n} is already cancelled." + (f" {_refund_status(order)}" if order["refund"] else ""))
    return _reply(f"Order #{n} was already delivered, so it can't be cancelled, but you may be able to return it.",
                  [_chip("Return for a refund", f"refund for order {n}")])


def refund(order: dict, confirmed: bool = False) -> dict:
    n, s = order["id"], order["status"]
    if order["refund"]:
        return _reply(f"Order #{n}: {_refund_status(order)}")
    if s == "processing":
        return _reply(f"Order #{n} hasn't shipped yet. Cancelling it gets you a full refund right away.",
                      [_chip("Cancel this order", f"cancel order {n}")])
    if s in ("shipped", "out_for_delivery"):
        return _reply(f"Order #{n} is still on its way. Once it's delivered you can request a return here.")
    if s == "cancelled":
        return _reply(f"Order #{n} was cancelled and paid by cash on delivery, so there's nothing to refund.")
    if s == "delivered":
        window = SALE_RETURN_DAYS if order["on_sale"] else RETURN_DAYS
        age = (date.today() - date.fromisoformat(order["delivered_at"])).days
        if age > window:
            return _reply(f"Order #{n} was delivered {age} days ago, which is outside our {window}-day return window, "
                          f"so it isn't eligible for a refund. If the product is faulty, it may be covered by warranty.",
                          [_chip("Warranty details", "what does the warranty cover?")])
        if not confirmed:
            credit = " Since it was bought in a sale, the refund will be store credit." if order["on_sale"] else ""
            return _reply(
                f"Order #{n} ({_items(order)}) was delivered {age} days ago, so it's within the {window}-day "
                f"return window.{credit} Shall I book a return pickup?",
                [_chip("Yes, book the return", "yes"), _chip("Not now", "no")],
                pending={"action": "refund", "order_id": n, "awaiting": "confirm"})
        store.request_return(n)
        return _reply(f"Done! {_refund_status(store.get_order(n))}", done="return_requested")
    return _reply(f"Order #{n} is being returned." + (f" {_refund_status(order)}" if order["refund"] else ""))


def invoice(order: dict) -> dict:
    lines = "\n".join(f"• {i['name']} ×{i['quantity']}: {_rupees(i['price'] * i['quantity'])}" for i in order["items"])
    return _reply(f"Invoice for order #{order['id']} (placed {_day(order['placed_at'])})\n{lines}\n"
                  f"Total: {_rupees(order['total'])}, paid by {order['payment_method']}\n"
                  f"Billed to {order['customer']}, {order['address']}")


def address(order: dict, new_address: str | None = None) -> dict:
    n = order["id"]
    if order["status"] == "cancelled":
        return _reply(f"Order #{n} was cancelled, so there's no delivery address to change.")
    if order["status"] in ("delivered", "returned", "return_requested"):
        return _reply(f"Order #{n} was already delivered on {_day(order['delivered_at'])}.")
    if order["status"] != "processing":
        return _reply(f"Order #{n} has already left our warehouse, so the address can't be changed. "
                      f"Contact the courier ({order['courier'] or 'see your tracking email'}) to reroute it.")
    if not new_address:
        return _reply(f"Order #{n} is currently going to {order['address']}. What's the new delivery address?",
                      pending={"action": "address", "order_id": n, "awaiting": "address"})
    store.update_address(n, new_address)
    return _reply(f"Updated. Order #{n} will now be delivered to {new_address}.", done="address_changed")


ACTIONS = {"track": track, "cancel": cancel, "refund": refund, "invoice": invoice, "address": address}


def run(action: str, number: int | None, **kwargs) -> dict:
    """Run an action, or ask for the order number if we don't have one."""
    if number is None:
        return _reply(ASK_NUMBER[action], pending={"action": action, "awaiting": "order_id"})
    order = store.get_order(number)
    if not order:
        return _reply(f"I couldn't find order #{number}. Please check the 5-digit number in your confirmation "
                      f"email (demo orders are 10001–10040).",
                      pending={"action": action, "awaiting": "order_id"})
    return ACTIONS[action](order, **kwargs)
