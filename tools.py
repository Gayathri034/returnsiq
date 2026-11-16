"""Predefined, parameterized SQL tools over returns.db.

The agent can only call these functions; it never writes its own SQL.
Every number in a data answer comes straight from these queries, so the
language model cannot invent figures.
"""
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent.parent / "returns-analytics" / "returns.db"

# Restocking-fee rule, kept in one place and matching docs/04_restocking_fee.md
FEE_RATE_THRESHOLD = 0.20
FEE_MIN_ORDERS = 3


class ReturnsDB:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Build it with the E-commerce Returns Analytics project "
                "(python load_data.py ...), or pass --db with its location."
            )
        self.con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)  # read-only

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self.con.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def countries(self) -> list[str]:
        return [r["country"] for r in self._rows("SELECT DISTINCT country FROM transactions")]

    def sample_customer(self) -> str:
        return self._rows("SELECT customer_id FROM transactions WHERE customer_id IS NOT NULL LIMIT 1")[0]["customer_id"]


SALES = "SUM(CASE WHEN is_return = 0 THEN line_value ELSE 0 END)"
RETURNS = "-SUM(CASE WHEN is_return = 1 THEN line_value ELSE 0 END)"


def overall_return_rate(db: ReturnsDB) -> dict:
    r = db._rows(f"SELECT {SALES} AS sales, {RETURNS} AS returned FROM transactions")[0]
    rate = 100 * r["returned"] / r["sales"]
    return {"rows": [r], "summary": (
        f"The overall return rate is {rate:.2f}% ({r['returned']:,.0f} GBP returned "
        f"out of {r['sales']:,.0f} GBP in sales).")}


def monthly_return_rate(db: ReturnsDB, last_n: int = 6) -> dict:
    rows = db._rows(f"""SELECT month, ROUND(100.0 * {RETURNS} / NULLIF({SALES}, 0), 2) AS return_rate_pct
                        FROM transactions GROUP BY month ORDER BY month DESC LIMIT ?""", (last_n,))
    rows.reverse()
    trend = ", ".join(f"{r['month']}: {r['return_rate_pct']}%" for r in rows)
    return {"rows": rows, "summary": f"Monthly return rate for the last {len(rows)} months: {trend}."}


def top_returned_products(db: ReturnsDB, n: int = 5) -> dict:
    n = max(1, min(int(n), 20))
    rows = db._rows(f"""SELECT MAX(description) AS product, ROUND({RETURNS}, 2) AS returned_value
                        FROM transactions GROUP BY stock_code ORDER BY returned_value DESC LIMIT ?""", (n,))
    items = "; ".join(f"{i}. {r['product']} ({r['returned_value']:,.0f} GBP)" for i, r in enumerate(rows, 1))
    return {"rows": rows, "summary": f"Top {n} products by returned value: {items}."}


def country_return_rate(db: ReturnsDB, country: str) -> dict:
    rows = db._rows(f"""SELECT country, {SALES} AS sales, {RETURNS} AS returned
                        FROM transactions WHERE LOWER(country) = LOWER(?) GROUP BY country""", (country,))
    if not rows or not rows[0]["sales"]:
        return {"rows": [], "summary": f"No sales data found for '{country}'."}
    r = rows[0]
    return {"rows": rows, "summary": (
        f"The return rate in {r['country']} is {100 * r['returned'] / r['sales']:.2f}% "
        f"({r['returned']:,.0f} GBP returned out of {r['sales']:,.0f} GBP).")}


def check_restocking_fee(db: ReturnsDB, customer_id: str) -> dict:
    """Combines live data with the written policy: would this customer pay the fee?"""
    rows = db._rows(f"""SELECT customer_id,
                               COUNT(DISTINCT CASE WHEN is_return = 0 THEN invoice END) AS orders,
                               {SALES} AS sales, {RETURNS} AS returned
                        FROM transactions WHERE customer_id = ? GROUP BY customer_id""", (str(customer_id),))
    if not rows or not rows[0]["sales"]:
        return {"rows": [], "summary": f"No purchase history found for customer {customer_id}."}
    r = rows[0]
    rate = r["returned"] / r["sales"]
    applies = rate >= FEE_RATE_THRESHOLD and r["orders"] >= FEE_MIN_ORDERS
    reason = ("their return rate is at or above 20% with at least 3 orders" if applies else
              "they have fewer than 3 orders" if r["orders"] < FEE_MIN_ORDERS else
              "their return rate is below 20%")
    return {"rows": rows, "summary": (
        f"Customer {customer_id} has {r['orders']} orders and a return rate of {100 * rate:.1f}%. "
        f"The 10% restocking fee {'WOULD' if applies else 'would NOT'} apply, because {reason} "
        f"[04_restocking_fee#who-the-fee-applies-to]. Damaged, defective, or wrong items are always exempt.")}


TOOLS = {
    "overall_return_rate": (overall_return_rate, "Overall return rate across all data. No arguments."),
    "monthly_return_rate": (monthly_return_rate, "Return rate by month for recent months. Optional arg: last_n (int)."),
    "top_returned_products": (top_returned_products, "Products with the most returned value. Optional arg: n (int)."),
    "country_return_rate": (country_return_rate, "Return rate for one country. Required arg: country (str)."),
    "check_restocking_fee": (check_restocking_fee, "Whether one customer would pay the restocking fee. Required arg: customer_id (str)."),
}
