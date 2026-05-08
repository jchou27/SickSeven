"""
BRTI-approximate composite BTC price feed.

Averages the live mid-price from four CME CF Bitcoin Real-Time Index (BRTI)
constituent exchanges: Kraken, Bitstamp, Coinbase, Gemini.
All endpoints are free with no API key required.

Why: Kalshi settles BTC contracts using the BRTI (60-second average in the
final minute before expiry). Using the actual BRTI requires a paid CF
Benchmarks license. Averaging these four constituent exchange prices is the
closest free approximation.

Usage:
    from price_feed import get_composite_price

    result = get_composite_price()
    # {
    #   "price":        95142.50,   # mean mid-price across responding exchanges
    #   "sources":      {"kraken": 95140, "bitstamp": 95143, ...},
    #   "source_count": 4,
    #   "spread_pct":   0.003,      # (max-min)/mid * 100 — quality indicator
    # }
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

log = logging.getLogger(__name__)

_TIMEOUT = 5


def _fetch_kraken() -> float:
    r = requests.get(
        "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    result = r.json()["result"]
    t = next(iter(result.values()))
    return (float(t["a"][0]) + float(t["b"][0])) / 2


def _fetch_bitstamp() -> float:
    r = requests.get(
        "https://www.bitstamp.net/api/v2/ticker/btcusd/",
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    d = r.json()
    return (float(d["ask"]) + float(d["bid"])) / 2


def _fetch_coinbase() -> float:
    r = requests.get(
        "https://api.coinbase.com/v2/prices/BTC-USD/spot",
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return float(r.json()["data"]["amount"])


def _fetch_gemini() -> float:
    r = requests.get(
        "https://api.gemini.com/v1/pubticker/btcusd",
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    d = r.json()
    return (float(d["ask"]) + float(d["bid"])) / 2


_FETCHERS: dict = {
    "kraken":   _fetch_kraken,
    "bitstamp": _fetch_bitstamp,
    "coinbase": _fetch_coinbase,
    "gemini":   _fetch_gemini,
}


def get_composite_price() -> dict:
    """
    Fetch BTC mid-price from all four exchanges in parallel and return the mean.

    Falls back gracefully — returns whatever subset responds (minimum 1).
    Raises RuntimeError only if every exchange fails.

    The 'spread_pct' field indicates cross-exchange divergence: values above
    ~0.05% suggest unusual fragmentation or a stale feed on one exchange.
    """
    prices: dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn): name for name, fn in _FETCHERS.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                prices[name] = future.result()
            except Exception as e:
                log.debug(f"price_feed: {name} failed — {e}")

    if not prices:
        raise RuntimeError("All BRTI-constituent exchange fetches failed")

    vals      = list(prices.values())
    mid       = sum(vals) / len(vals)
    spread    = ((max(vals) - min(vals)) / mid * 100) if len(vals) > 1 else 0.0

    return {
        "price":        round(mid, 2),
        "sources":      {k: round(v, 2) for k, v in prices.items()},
        "source_count": len(prices),
        "spread_pct":   round(spread, 4),
    }
