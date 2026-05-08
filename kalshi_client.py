"""Kalshi Trading API v2 client with RSA authentication."""
import base64
import os
import textwrap
import time
import uuid
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

_KALSHI_BASE_URL = "https://api.kalshi.com"
_API_PREFIX = "/trade-api/v2"
_KEY_ID = os.getenv("KALSHI_API_KEY", "")
_PRIV_RAW = os.getenv("KALSHI_PRIV", "")


def _load_private_key():
    if not _PRIV_RAW:
        raise EnvironmentError("KALSHI_PRIV not set in .env")
    raw_b64 = "".join(_PRIV_RAW.split())
    lines = textwrap.wrap(raw_b64, 64)
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    for header in ("RSA PRIVATE KEY", "PRIVATE KEY"):
        try:
            pem = (f"-----BEGIN {header}-----\n"
                   + "\n".join(lines)
                   + f"\n-----END {header}-----")
            return serialization.load_pem_private_key(
                pem.encode(), password=None, backend=default_backend()
            )
        except Exception:
            continue
    raise ValueError("Could not load Kalshi private key — check KALSHI_PRIV in .env")


_PRIVATE_KEY = _load_private_key()


def _signed_headers(method: str, path: str) -> dict:
    """RSA-sign request: message = timestamp + METHOD + /trade-api/v2/path"""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + _API_PREFIX + path).encode()
    sig = _PRIVATE_KEY.sign(msg, padding.PKCS1v15(), hashes.SHA256())
    return {
        "KALSHI-ACCESS-KEY": _KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }


def _url(path: str) -> str:
    return _KALSHI_BASE_URL + _API_PREFIX + path


def _get(path: str, params: dict = None) -> dict:
    r = requests.get(_url(path), headers=_signed_headers("GET", path),
                     params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _post(path: str, body: dict) -> dict:
    r = requests.post(_url(path), headers=_signed_headers("POST", path),
                      json=body, timeout=10)
    r.raise_for_status()
    return r.json()


def _delete(path: str) -> dict:
    r = requests.delete(_url(path), headers=_signed_headers("DELETE", path),
                        timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_balance() -> dict:
    """Returns {'balance': cents, 'payout': cents, ...}"""
    return _get("/portfolio/balance")


def get_markets(series_ticker: str = "KXBTCD", status: str = "open") -> list:
    data = _get("/markets", {"series_ticker": series_ticker,
                              "status": status, "limit": 50})
    return data.get("markets", [])


def search_markets(keyword: str, status: str = "open") -> list:
    """Fallback: search all open markets for BTC by keyword in title/ticker."""
    data = _get("/markets", {"status": status, "limit": 100})
    all_markets = data.get("markets", [])
    kw = keyword.lower()
    return [m for m in all_markets
            if kw in m.get("title", "").lower() or kw in m.get("ticker", "").lower()]


def get_positions() -> list:
    data = _get("/portfolio/positions")
    return data.get("market_positions", [])


def get_orders(status: str = "resting") -> list:
    data = _get("/portfolio/orders", {"status": status, "limit": 100})
    return data.get("orders", [])


def get_order(order_id: str) -> dict:
    return _get(f"/orders/{order_id}")


def place_order(
    ticker: str,
    action: str,
    side: str,
    count: int,
    order_type: str,
    price_cents: Optional[int] = None,
) -> dict:
    """
    action: "buy" | "sell"
    side:   "yes" | "no"
    type:   "limit" | "market"
    price_cents: required for limit orders (1-99)
    """
    body = {
        "ticker": ticker,
        "client_order_id": str(uuid.uuid4()),
        "action": action,
        "type": order_type,
        "side": side,
        "count": count,
    }
    if order_type == "limit" and price_cents is not None:
        body["yes_price" if side == "yes" else "no_price"] = price_cents
    return _post("/orders", body)


def cancel_order(order_id: str) -> dict:
    return _delete(f"/orders/{order_id}")


def cancel_all_resting() -> list:
    results = []
    for order in get_orders(status="resting"):
        try:
            results.append(cancel_order(order["order_id"]))
        except Exception:
            pass
    return results
