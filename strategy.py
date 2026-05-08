"""
Signal generation, indicator computation, and order decision logic.

Imported by trader.py, dashboard.py, monitor.py, and probability_model.py.
All indicator logic lives here — never duplicate it elsewhere.
"""
import math
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core indicator functions
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    # When loss == 0 (all-gain run), RS is infinite → RSI = 100 (not NaN)
    rs = np.where(loss == 0, np.inf, gain / loss)
    return pd.Series(100 - (100 / (1 + rs)), index=series.index)


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast    = series.ewm(span=fast,          adjust=False).mean()
    ema_slow    = series.ewm(span=slow,          adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, mid, lower) Bollinger Bands."""
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ---------------------------------------------------------------------------
# Indicator snapshot (latest values — used by trader.py)
# ---------------------------------------------------------------------------

def compute_indicators(prices: pd.Series) -> dict:
    """
    Return a dict of the most recent value for every indicator.
    Pass this to generate_signal().
    """
    sma20  = prices.rolling(20).mean()
    sma50  = prices.rolling(50).mean()
    ema20  = prices.ewm(span=20,  adjust=False).mean()
    ema200 = prices.ewm(span=200, adjust=False).mean()

    rsi14            = compute_rsi(prices, 14)
    macd_l, macd_s, macd_h = compute_macd(prices)
    bb_upper, bb_mid, bb_lower = compute_bollinger(prices)

    # Bollinger %B: 0 = price at lower band, 1 = price at upper band
    bb_width = bb_upper - bb_lower
    bb_pct_b = (prices - bb_lower) / bb_width.replace(0, np.nan)

    # Volatility proxy: rolling std of log returns over 14 periods
    log_ret = np.log(prices / prices.shift(1))
    valid_std = log_ret.rolling(14).std().dropna()
    atr_pct = float(valid_std.iloc[-1]) if not valid_std.empty else 0.003
    if math.isnan(atr_pct) or atr_pct <= 0:
        atr_pct = 0.003

    def _safe(s: pd.Series, fallback: float = 0.0) -> float:
        arr = s.to_numpy()
        if len(arr) == 0:
            return fallback
        v = float(arr[-1])
        return fallback if math.isnan(v) else v

    return {
        "price":       _safe(prices),
        "sma20":       _safe(sma20),
        "sma50":       _safe(sma50),
        "ema20":       _safe(ema20),
        "ema200":      _safe(ema200),
        "rsi":         _safe(rsi14, 50.0),
        "macd":        _safe(macd_l),
        "macd_signal": _safe(macd_s),
        "macd_hist":   _safe(macd_h),
        "bb_upper":    _safe(bb_upper),
        "bb_lower":    _safe(bb_lower),
        "bb_pct_b":    _safe(bb_pct_b, 0.5),
        "atr_pct":     atr_pct,
    }


# ---------------------------------------------------------------------------
# Full series for charting (used by dashboard.py)
# ---------------------------------------------------------------------------

def add_chart_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all indicator columns to a DataFrame that has a 'price' column.
    The DataFrame is returned with new columns added in-place on a copy.
    """
    df = df.copy()
    p = df["price"]

    df["sma20"]  = p.rolling(20).mean()
    df["sma50"]  = p.rolling(50).mean()
    df["ema20"]  = p.ewm(span=20,  adjust=False).mean()
    df["ema200"] = p.ewm(span=200, adjust=False).mean()

    df["rsi14"] = compute_rsi(p, 14)

    macd_l, macd_s, macd_h = compute_macd(p)
    df["macd"]        = macd_l
    df["macd_signal"] = macd_s
    df["macd_hist"]   = macd_h

    bb_upper, bb_mid, bb_lower = compute_bollinger(p)
    df["bb_upper"] = bb_upper
    df["bb_mid"]   = bb_mid
    df["bb_lower"] = bb_lower

    return df


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signal(ind: dict) -> dict:
    """
    Five-factor signal engine.  Returns label, direction, strength, and
    the raw bull_score so callers can reason about how close to a boundary we are.

    Factors and their max contribution (each can add OR subtract):
      RSI          : ±2  (oversold/overbought)
      MACD         : ±2  (line vs signal, and position vs zero)
      MA crossover : ±1  (SMA20 vs SMA50 golden/death cross)
      Long trend   : ±1  (price vs EMA200)
      Bollinger    : ±1  (price at band extremes)
    ─────────────────────
    Total range    : ±7

    Score → signal:
      ≥ 4  : STRONG BUY
      2–3  : BUY
      -1–1 : HOLD
      -2–-3: SELL
      ≤ -4 : STRONG SELL
    """
    rsi      = ind["rsi"]
    macd     = ind["macd"]
    macd_sig = ind["macd_signal"]
    price    = ind["price"]
    sma20    = ind["sma20"]
    sma50    = ind["sma50"]
    ema200   = ind["ema200"]
    bb_pct_b = ind["bb_pct_b"]  # 0 = at lower band, 1 = at upper band

    score = 0

    # 1. RSI (±2)
    if rsi < 30:    score += 2
    elif rsi < 40:  score += 1
    elif rsi > 70:  score -= 2
    elif rsi > 60:  score -= 1

    # 2. MACD (±2)
    #    First point: is MACD above or below its signal line?
    #    Second point (bonus): is the MACD line itself above or below zero?
    if macd > macd_sig:
        score += 1
        if macd > 0:   score += 1   # confirmed: above zero
    else:
        score -= 1
        if macd < 0:   score -= 1   # confirmed: below zero

    # 3. Medium-term MA crossover (±1)
    score += 1 if sma20 > sma50 else -1

    # 4. Long-term trend filter — EMA200 (±1)
    #    Only trade with the dominant trend.
    score += 1 if price > ema200 else -1

    # 5. Bollinger Band extremes (±1)
    #    Near lower band = oversold pressure.  Near upper = extended.
    if bb_pct_b <= 0.05:    score += 1
    elif bb_pct_b >= 0.95:  score -= 1

    # Map to label
    if score >= 4:
        label, direction, strength = "STRONG BUY",  "up",      1.0
    elif score >= 2:
        label, direction, strength = "BUY",          "up",      0.5
    elif score <= -4:
        label, direction, strength = "STRONG SELL",  "down",    1.0
    elif score <= -2:
        label, direction, strength = "SELL",          "down",    0.5
    else:
        label, direction, strength = "HOLD",          "neutral", 0.0

    return {
        "label":      label,
        "direction":  direction,
        "strength":   strength,
        "bull_score": score,
        "rsi":        rsi,
        "macd":       macd,
        "macd_hist":  ind["macd_hist"],
        "atr_pct":    ind["atr_pct"],
    }


# ---------------------------------------------------------------------------
# Market selection
# ---------------------------------------------------------------------------

def select_market(markets: list, direction: str) -> Optional[dict]:
    """
    Pick the best Kalshi BTC market for the given direction.

    Scoring criteria (in priority order):
    1. Relevant-side ask must be 15–85 cents (not near-certain; maximum edge range)
    2. Prefer markets closest to 50 cents (maximum uncertainty = maximum edge value)
    3. Among similar odds, prefer highest volume (tightest spread, easiest exit)

    Uses log(1 + volume) to prevent a single high-volume market from completely
    dominating markets with reasonable odds and moderate volume.
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
            proximity = 1.0 - abs(target - 50) / 50.0   # 0→edge, 1→fair
            vol_score = math.log1p(volume)
            scored.append((proximity * 0.6 + vol_score * 0.4, m))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # Relax: any market, highest volume
    return max(markets, key=lambda m: m.get("volume") or 0)


# ---------------------------------------------------------------------------
# Order sizing
# ---------------------------------------------------------------------------

def compute_order(
    signal: dict,
    market: dict,
    max_contracts: int,
    atr_pct: float = 0.003,
) -> Optional[dict]:
    """
    Convert signal + market into order parameters.
    Returns None when no trade should be placed (HOLD or missing price).

    Position sizing:
    - Base: max_contracts × signal.strength  (1.0 for strong, 0.5 for regular)
    - Volatility adjustment: scale down when BTC is unusually volatile.
      Reference volatility = 0.3% per hour.  At 2× vol, halve the position.
      This ensures roughly constant dollar-risk per trade across vol regimes.
    """
    if signal["direction"] == "neutral" or signal["strength"] == 0:
        return None

    side = "yes" if signal["direction"] == "up" else "no"
    ask  = market.get("yes_ask" if side == "yes" else "no_ask")
    if ask is None:
        return None

    # Volatility-adjusted count
    base_vol    = 0.003
    vol_factor  = min(1.0, base_vol / max(atr_pct, base_vol * 0.1))
    raw_count   = max_contracts * signal["strength"] * vol_factor
    count       = max(1, round(raw_count))

    # Limit 1 cent above ask — quick fill without chasing the book
    limit_price = min(99, max(1, ask + 1))
    cost_usd    = round(count * limit_price / 100, 2)

    return {
        "ticker":      market["ticker"],
        "action":      "buy",
        "side":        side,
        "count":       count,
        "order_type":  "limit",
        "price_cents": limit_price,
        "cost_usd":    cost_usd,
        "rationale":   (
            f"{signal['label']} | RSI={signal['rsi']:.1f} "
            f"MACD={'▲' if signal['macd'] > signal['macd_hist'] + signal['macd'] else '▼'} "
            f"score={signal['bull_score']:+d} | "
            f"vol_adj={vol_factor:.2f} → {count} contracts @ {limit_price}¢"
        ),
    }


# ---------------------------------------------------------------------------
# Short-term indicators  (1m / 5m / 15m OHLCV data)
# ---------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Rolling windowed VWAP over `window` periods.
    Uses typical price = (high + low + close) / 3.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol     = df["volume"].replace(0, float("nan"))
    vwap    = (typical * vol).rolling(window).sum() / vol.rolling(window).sum()
    return vwap


def compute_short_term_indicators(df: pd.DataFrame, timeframe: str = "1m") -> dict:
    """
    Compute short-term indicators for OHLCV data.

    df must have columns: open, high, low, close, volume
    timeframe: "1m" | "5m" | "15m"  — adjusts indicator periods accordingly.

    Indicator periods by timeframe:
      1m:  EMA5/10/21, RSI7,  MACD(5,13,3),  BB(14), VWAP(60)
      5m:  EMA9/21/55, RSI10, MACD(12,26,9), BB(20), VWAP(48)
      15m: EMA9/21/55, RSI14, MACD(12,26,9), BB(20), VWAP(24)
    """
    periods = {
        "1m":  dict(ema_fast=5,  ema_mid=10, ema_slow=21, rsi=7,  macd=(5,13,3),  bb=14, vwap=60),
        "5m":  dict(ema_fast=9,  ema_mid=21, ema_slow=55, rsi=10, macd=(12,26,9), bb=20, vwap=48),
        "15m": dict(ema_fast=9,  ema_mid=21, ema_slow=55, rsi=14, macd=(12,26,9), bb=20, vwap=24),
    }
    p = periods.get(timeframe, periods["5m"])
    close = df["close"]

    ema_f  = close.ewm(span=p["ema_fast"],  adjust=False).mean()
    ema_m  = close.ewm(span=p["ema_mid"],   adjust=False).mean()
    ema_s  = close.ewm(span=p["ema_slow"],  adjust=False).mean()
    rsi_s  = compute_rsi(close, p["rsi"])
    mf, ms, mh = compute_macd(close, *p["macd"])
    bb_u, bb_mid, bb_l = compute_bollinger(close, p["bb"])
    bb_w   = (bb_u - bb_l).replace(0, float("nan"))
    bb_pct = (close - bb_l) / bb_w
    vwap   = compute_vwap(df, p["vwap"])

    log_ret = np.log(close / close.shift(1))
    valid_std = log_ret.rolling(14).std().dropna()
    atr_pct = float(valid_std.iloc[-1]) if not valid_std.empty else 0.003
    if math.isnan(atr_pct) or atr_pct <= 0:
        atr_pct = 0.003

    def _s(series: pd.Series, fallback: float = 0.0) -> float:
        arr = series.to_numpy()
        if len(arr) == 0:
            return fallback
        v = float(arr[-1])
        return fallback if math.isnan(v) else v

    price = _s(close)
    return {
        "price":      price,
        "ema_fast":   _s(ema_f),
        "ema_mid":    _s(ema_m),
        "ema_slow":   _s(ema_s),
        "rsi":        _s(rsi_s, 50.0),
        "macd":       _s(mf),
        "macd_signal":_s(ms),
        "macd_hist":  _s(mh),
        "bb_upper":   _s(bb_u),
        "bb_lower":   _s(bb_l),
        "bb_pct_b":   _s(bb_pct, 0.5),
        "vwap":       _s(vwap, price),
        "atr_pct":    atr_pct,
        "timeframe":  timeframe,
    }


def generate_short_term_signal(ind: dict) -> dict:
    """
    Five-factor signal for short-term (1m/5m/15m) data.

    Factors:
      RSI         : ±2  (same as long-term but faster periods)
      MACD        : ±2  (line vs signal + zero line bonus)
      EMA cross   : ±1  (fast EMA vs slow EMA)
      VWAP        : ±1  (price vs rolling VWAP)
      Bollinger   : ±1  (price at band extremes)
    """
    rsi      = ind["rsi"]
    macd     = ind["macd"]
    macd_sig = ind["macd_signal"]
    price    = ind["price"]
    ema_fast = ind["ema_fast"]
    ema_slow = ind["ema_slow"]
    vwap     = ind["vwap"]
    bb_pct_b = ind["bb_pct_b"]

    score = 0

    # RSI (±2)
    if rsi < 30:    score += 2
    elif rsi < 40:  score += 1
    elif rsi > 70:  score -= 2
    elif rsi > 60:  score -= 1

    # MACD (±2)
    if macd > macd_sig:
        score += 1
        if macd > 0: score += 1
    else:
        score -= 1
        if macd < 0: score -= 1

    # Fast EMA crossover (±1) — replaces SMA50/EMA200 for short-term
    score += 1 if ema_fast > ema_slow else -1

    # VWAP (±1) — institutional reference price
    score += 1 if price > vwap else -1

    # Bollinger (±1)
    if bb_pct_b <= 0.05:    score += 1
    elif bb_pct_b >= 0.95:  score -= 1

    if score >= 4:
        label, direction, strength = "STRONG BUY",  "up",      1.0
    elif score >= 2:
        label, direction, strength = "BUY",          "up",      0.5
    elif score <= -4:
        label, direction, strength = "STRONG SELL",  "down",    1.0
    elif score <= -2:
        label, direction, strength = "SELL",          "down",    0.5
    else:
        label, direction, strength = "HOLD",          "neutral", 0.0

    return {
        "label":      label,
        "direction":  direction,
        "strength":   strength,
        "bull_score": score,
        "rsi":        rsi,
        "macd":       macd,
        "macd_hist":  ind["macd_hist"],
        "atr_pct":    ind["atr_pct"],
        "timeframe":  ind.get("timeframe", "?"),
    }


def add_short_term_chart_indicators(df: pd.DataFrame, timeframe: str = "5m") -> pd.DataFrame:
    """Add all short-term indicator columns to an OHLCV DataFrame."""
    p = {
        "1m":  dict(ema_fast=5,  ema_mid=10, ema_slow=21, rsi=7,  macd=(5,13,3),  bb=14, vwap=60),
        "5m":  dict(ema_fast=9,  ema_mid=21, ema_slow=55, rsi=10, macd=(12,26,9), bb=20, vwap=48),
        "15m": dict(ema_fast=9,  ema_mid=21, ema_slow=55, rsi=14, macd=(12,26,9), bb=20, vwap=24),
    }.get(timeframe, {
        "ema_fast":5, "ema_mid":10, "ema_slow":21, "rsi":7, "macd":(5,13,3), "bb":14, "vwap":60
    })

    df = df.copy()
    c  = df["close"]
    df["ema_fast"]    = c.ewm(span=p["ema_fast"],  adjust=False).mean()
    df["ema_mid"]     = c.ewm(span=p["ema_mid"],   adjust=False).mean()
    df["ema_slow"]    = c.ewm(span=p["ema_slow"],  adjust=False).mean()
    df["rsi"]         = compute_rsi(c, p["rsi"])
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(c, *p["macd"])
    df["bb_upper"], df["bb_mid"], df["bb_lower"]   = compute_bollinger(c, p["bb"])
    df["vwap"]        = compute_vwap(df, p["vwap"])
    return df
