"""Signal generation and order decision logic."""
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Indicators (pure pandas)
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def compute_indicators(prices: pd.Series) -> dict:
    sma20 = prices.rolling(20).mean()
    sma50 = prices.rolling(50).mean()
    ema20 = prices.ewm(span=20, adjust=False).mean()
    rsi = compute_rsi(prices)
    return {
        "price":  float(prices.iloc[-1]),
        "sma20":  float(sma20.iloc[-1]),
        "sma50":  float(sma50.iloc[-1]),
        "ema20":  float(ema20.iloc[-1]),
        "rsi":    float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50.0,
    }


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signal(ind: dict) -> dict:
    """
    Score-based multi-factor signal.
    Each factor contributes ±1 or ±2 to a bull_score.
    Score  >= 3 → STRONG BUY
    Score  1-2  → BUY
    Score  0    → HOLD
    Score -1–-2 → SELL
    Score <= -3 → STRONG SELL
    """
    rsi = ind["rsi"]
    ma_bull  = ind["sma20"] > ind["sma50"]   # golden/death cross
    ema_bull = ind["price"] > ind["ema20"]    # price above short-term EMA

    score = 0
    score += 2 if rsi < 30 else (1 if rsi < 40 else 0)
    score -= 2 if rsi > 70 else (1 if rsi > 60 else 0)
    score += 1 if ma_bull  else -1
    score += 1 if ema_bull else -1

    if score >= 3:
        label, direction, strength = "STRONG BUY",  "up",      1.0
    elif score >= 1:
        label, direction, strength = "BUY",          "up",      0.5
    elif score <= -3:
        label, direction, strength = "STRONG SELL",  "down",    1.0
    elif score <= -1:
        label, direction, strength = "SELL",          "down",    0.5
    else:
        label, direction, strength = "HOLD",          "neutral", 0.0

    return {
        "label":      label,
        "direction":  direction,
        "strength":   strength,
        "bull_score": score,
        "rsi":        rsi,
    }


# ---------------------------------------------------------------------------
# Market selection
# ---------------------------------------------------------------------------

def select_market(markets: list, direction: str) -> Optional[dict]:
    """
    Find the best Kalshi BTC market to trade for the given direction.

    Prefer markets where:
    - The relevant-side ask is between 15 and 85 cents (not too extreme)
    - Volume is highest (most liquid, tightest spread)
    - Odds closest to 50 (most informative signal)
    """
    if not markets:
        return None

    scored = []
    for m in markets:
        yes_ask = m.get("yes_ask")
        no_ask  = m.get("no_ask")
        volume  = m.get("volume") or 0
        if yes_ask is None or no_ask is None:
            continue
        target = yes_ask if direction == "up" else no_ask
        if 15 <= target <= 85:
            proximity_to_even = 1.0 - abs(target - 50) / 50.0
            scored.append((proximity_to_even * (1 + volume), m))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # Relax constraint: just pick highest volume
    return max(markets, key=lambda m: m.get("volume") or 0)


# ---------------------------------------------------------------------------
# Order sizing
# ---------------------------------------------------------------------------

def compute_order(signal: dict, market: dict, max_contracts: int) -> Optional[dict]:
    """
    Convert a signal + market into order parameters.
    Returns None when no trade should be placed (HOLD or missing price).
    """
    if signal["direction"] == "neutral" or signal["strength"] == 0:
        return None

    count = max(1, round(max_contracts * signal["strength"]))

    side = "yes" if signal["direction"] == "up" else "no"
    ask  = market.get("yes_ask" if side == "yes" else "no_ask")
    if ask is None:
        return None

    # Limit 1 cent inside worst case to get a quick fill without chasing
    limit_price = min(99, max(1, ask + 1))

    return {
        "ticker":      market["ticker"],
        "action":      "buy",
        "side":        side,
        "count":       count,
        "order_type":  "limit",
        "price_cents": limit_price,
        "cost_usd":    round(count * limit_price / 100, 2),
        "rationale":   (f"{signal['label']} | RSI={signal['rsi']:.1f} "
                        f"score={signal['bull_score']:+d}"),
    }
