"""Mock ShopEase store: customers, products, orders and refunds in SQLite.

Seeded once with ~40 realistic orders so the bot can answer
"where is order 10023?" or cancel / refund real records. A few order
numbers are fixed so demos and tests are predictable:

    10023  processing        -> can be cancelled / address changed
    10027  shipped           -> on the way, can't cancel
    10040  out for delivery  -> arriving today
    10031  delivered 10 days ago -> eligible for a return
    10005  delivered 50 days ago -> return window closed
    10012  returned, refund in progress
"""
import json
import random
from datetime import date, timedelta

from . import db
from .config import DATA_DIR

STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL,
    email TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS products (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    category TEXT NOT NULL,
    price    INTEGER NOT NULL,        -- rupees
    stock    INTEGER NOT NULL         -- live stock; descriptions and specs live in data/products.json
);
CREATE TABLE IF NOT EXISTS orders (
    id              INTEGER PRIMARY KEY,   -- the 5-digit order number customers see
    customer_id     INTEGER REFERENCES customers(id),
    status          TEXT NOT NULL,         -- processing | shipped | out_for_delivery | delivered
                                           -- | return_requested | returned | cancelled
    placed_at       TEXT NOT NULL,
    shipped_at      TEXT,
    delivered_at    TEXT,
    eta             TEXT,
    total           INTEGER NOT NULL,
    payment_method  TEXT NOT NULL,
    address         TEXT NOT NULL,
    courier         TEXT,
    tracking_number TEXT,
    on_sale         INTEGER DEFAULT 0      -- sale items: 15-day returns, store credit only
);
CREATE TABLE IF NOT EXISTS order_items (
    order_id   INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity   INTEGER NOT NULL,
    price      INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS refunds (
    id           INTEGER PRIMARY KEY,
    order_id     INTEGER REFERENCES orders(id),
    amount       INTEGER NOT NULL,
    method       TEXT NOT NULL,          -- original payment method or store credit
    status       TEXT NOT NULL,          -- requested | processing | completed
    requested_at TEXT NOT NULL,
    expected_by  TEXT
);
"""

CATALOG = json.loads((DATA_DIR / "products.json").read_text(encoding="utf-8"))
PRODUCTS = [(p["name"], p["price"]) for p in CATALOG]
STORE_TABLES = ("refunds", "order_items", "orders", "products", "customers")
CUSTOMERS = [
    ("Aarav Sharma", "aarav@example.com"), ("Diya Patel", "diya@example.com"),
    ("Rohan Iyer", "rohan@example.com"), ("Meera Nair", "meera@example.com"),
    ("Kabir Singh", "kabir@example.com"), ("Ananya Rao", "ananya@example.com"),
    ("Vikram Das", "vikram@example.com"), ("Sara Khan", "sara@example.com"),
]
ADDRESSES = ["12 MG Road, Bengaluru 560001", "45 Park Street, Kolkata 700016",
             "8 Marine Drive, Mumbai 400020", "221 Anna Salai, Chennai 600002",
             "17 Connaught Place, New Delhi 110001", "3 Banjara Hills, Hyderabad 500034"]
PAYMENTS = ["UPI", "Credit card", "Debit card", "Net banking", "Cash on delivery"]
COURIERS = ["BlueDart", "Delhivery", "Ekart"]

# order number -> (status, days since placed); everything else is random
FIXED = {10023: ("processing", 1), 10027: ("shipped", 3), 10040: ("out_for_delivery", 4),
         10031: ("delivered", 14), 10005: ("delivered", 55), 10012: ("returned", 20)}


def init_store():
    with db.get_conn() as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(products)")}
        if columns and "stock" not in columns:  # database from an older version: rebuild the demo store
            for table in STORE_TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.executescript(STORE_SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0:
            _seed(conn)


def product_stock(product_id: int) -> int:
    with db.get_conn() as conn:
        row = conn.execute("SELECT stock FROM products WHERE id = ?", (product_id,)).fetchone()
    return row[0] if row else 0


def _seed(conn):
    rng = random.Random(7)
    today = date.today()
    conn.executemany("INSERT INTO products (name, category, price, stock) VALUES (?, ?, ?, ?)",
                     [(p["name"], p["category"], p["price"], p["stock"]) for p in CATALOG])
    conn.executemany("INSERT INTO customers (name, email) VALUES (?, ?)", CUSTOMERS)

    for number in range(10001, 10041):
        status, age = FIXED.get(number) or rng.choice([
            ("processing", rng.randint(0, 1)), ("shipped", rng.randint(2, 4)),
            ("delivered", rng.randint(6, 40)), ("delivered", rng.randint(6, 40)),
            ("cancelled", rng.randint(3, 20)), ("returned", rng.randint(15, 35)),
        ])
        placed = today - timedelta(days=age)
        items = rng.sample(range(1, len(PRODUCTS) + 1), rng.randint(1, 3))
        lines = [(pid, rng.choice([1, 1, 1, 2]), PRODUCTS[pid - 1][1]) for pid in items]
        total = sum(q * p for _, q, p in lines)
        payment = rng.choice(PAYMENTS)
        if number in FIXED:
            payment = "UPI"  # predictable refunds for the demo orders
        shipped = delivered = eta = courier = tracking = None
        if status in ("shipped", "out_for_delivery", "delivered", "returned"):
            shipped = placed + timedelta(days=1)
            courier = rng.choice(COURIERS)
            tracking = f"SE{rng.randint(10**7, 10**8 - 1)}"
        if status == "processing":
            eta = placed + timedelta(days=5)
        elif status == "shipped":
            eta = today + timedelta(days=rng.randint(1, 3))
        elif status == "out_for_delivery":
            eta = today
        if status in ("delivered", "returned"):
            delivered = placed + timedelta(days=4)

        conn.execute(
            "INSERT INTO orders (id, customer_id, status, placed_at, shipped_at, delivered_at, eta, total, "
            "payment_method, address, courier, tracking_number, on_sale) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (number, rng.randint(1, len(CUSTOMERS)), status, placed.isoformat(),
             shipped and shipped.isoformat(), delivered and delivered.isoformat(), eta and eta.isoformat(),
             total, payment, rng.choice(ADDRESSES), courier, tracking, int(rng.random() < 0.15)),
        )
        conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?)", [(number, *line) for line in lines])

        if status in ("cancelled", "returned") and payment != "Cash on delivery":
            requested = placed + timedelta(days=1 if status == "cancelled" else 8)
            done = number != 10012 and rng.random() < 0.7
            # refunds still in progress always have a date in the future
            expected = requested + timedelta(days=7) if done else today + timedelta(days=rng.randint(2, 5))
            conn.execute(
                "INSERT INTO refunds (order_id, amount, method, status, requested_at, expected_by) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (number, total, payment, "completed" if done else "processing", requested.isoformat(),
                 expected.isoformat()),
            )


def get_order(number: int) -> dict | None:
    with db.get_conn() as conn:
        order = conn.execute(
            "SELECT o.*, c.name AS customer FROM orders o JOIN customers c ON c.id = o.customer_id "
            "WHERE o.id = ?", (number,)).fetchone()
        if not order:
            return None
        items = conn.execute(
            "SELECT p.name, i.quantity, i.price FROM order_items i JOIN products p ON p.id = i.product_id "
            "WHERE i.order_id = ?", (number,)).fetchall()
        refund = conn.execute("SELECT * FROM refunds WHERE order_id = ? ORDER BY id DESC", (number,)).fetchone()
    return {**dict(order), "items": [dict(i) for i in items], "refund": dict(refund) if refund else None}


def cancel(number: int) -> dict | None:
    """Cancel a not-yet-shipped order. Returns the refund created, if any."""
    order = get_order(number)
    with db.get_conn() as conn:
        conn.execute("UPDATE orders SET status = 'cancelled', eta = NULL WHERE id = ?", (number,))
        if order["payment_method"] == "Cash on delivery":
            return None
        return _create_refund(conn, order, order["payment_method"], "processing")


def request_return(number: int) -> dict:
    order = get_order(number)
    method = "Store credit" if order["on_sale"] else order["payment_method"]
    if method == "Cash on delivery":
        method = "Bank transfer"
    with db.get_conn() as conn:
        conn.execute("UPDATE orders SET status = 'return_requested' WHERE id = ?", (number,))
        return _create_refund(conn, order, method, "requested")


def update_address(number: int, address: str):
    with db.get_conn() as conn:
        conn.execute("UPDATE orders SET address = ? WHERE id = ?", (address, number))


def _create_refund(conn, order: dict, method: str, status: str) -> dict:
    today = date.today()
    refund = {"order_id": order["id"], "amount": order["total"], "method": method, "status": status,
              "requested_at": today.isoformat(), "expected_by": (today + timedelta(days=10)).isoformat()}
    conn.execute("INSERT INTO refunds (order_id, amount, method, status, requested_at, expected_by) "
                 "VALUES (:order_id, :amount, :method, :status, :requested_at, :expected_by)", refund)
    return refund


def recent_orders(limit: int = 15) -> list[dict]:
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT o.id, o.status, o.placed_at, o.total, c.name AS customer FROM orders o "
            "JOIN customers c ON c.id = o.customer_id ORDER BY o.id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
