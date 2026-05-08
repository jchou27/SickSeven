import base64
import json
import os
import textwrap
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

load_dotenv()

GECKO_API_KEY  = os.getenv("GECKO_API", "")
KALSHI_KEY_ID  = os.getenv("KALSHI_API_KEY", "")
KALSHI_PRIV_RAW = os.getenv("KALSHI_PRIV", "")
GECKO_BASE     = "https://api.coingecko.com/api/v3"
KALSHI_BASE_URL = "https://api.kalshi.com"
KALSHI_API_PFX  = "/trade-api/v2"
STATE_FILE     = Path("trading_state.json")
CONFIG_FILE    = Path("trading_config.json")

DEFAULT_CONFIG = {
    "enabled":           False,
    "series_ticker":     "KXBTCD",
    "max_contracts":     5,
    "max_open_risk_usd": 50.0,
    "only_on_change":    True,
    "loop_interval_sec": 60,
    "dry_run":           True,
}

# ---------------------------------------------------------------------------
# Kalshi auth (needed for live market data in dashboard)
# ---------------------------------------------------------------------------

def _load_kalshi_key():
    if not KALSHI_PRIV_RAW:
        return None
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        raw_b64 = "".join(KALSHI_PRIV_RAW.split())
        lines = textwrap.wrap(raw_b64, 64)
        for hdr in ("RSA PRIVATE KEY", "PRIVATE KEY"):
            try:
                pem = f"-----BEGIN {hdr}-----\n" + "\n".join(lines) + f"\n-----END {hdr}-----"
                return serialization.load_pem_private_key(pem.encode(), password=None, backend=default_backend())
            except Exception:
                pass
    except Exception as e:
        st.warning(f"Kalshi key load error: {e}")
    return None

_KAL_KEY = _load_kalshi_key()


def kalshi_signed(method: str, path: str) -> dict:
    if not _KAL_KEY or not KALSHI_KEY_ID:
        return {}
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + KALSHI_API_PFX + path).encode()
    sig = _KAL_KEY.sign(msg, padding.PKCS1v15(), hashes.SHA256())
    return {
        "KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def fetch_btc_price() -> dict:
    url = (f"{GECKO_BASE}/simple/price?ids=bitcoin&vs_currencies=usd"
           f"&include_24hr_change=true&include_24hr_vol=true"
           f"&x_cg_demo_api_key={GECKO_API_KEY}")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()["bitcoin"]


@st.cache_data(ttl=120)
def fetch_ohlc(days: int = 7) -> pd.DataFrame:
    url = (f"{GECKO_BASE}/coins/bitcoin/ohlc?vs_currency=usd"
           f"&days={days}&x_cg_demo_api_key={GECKO_API_KEY}")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df


@st.cache_data(ttl=120)
def fetch_price_history(days: int = 30) -> pd.DataFrame:
    url = (f"{GECKO_BASE}/coins/bitcoin/market_chart?vs_currency=usd"
           f"&days={days}&interval=hourly&x_cg_demo_api_key={GECKO_API_KEY}")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["prices"], columns=["ts", "price"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df


@st.cache_data(ttl=60)
def fetch_kalshi_markets() -> list:
    path = "/markets"
    headers = kalshi_signed("GET", path)
    if not headers:
        return []
    for params in ("?series_ticker=KXBTCD&limit=20&status=open",
                   "?limit=50&status=open"):
        try:
            r = requests.get(
                KALSHI_BASE_URL + KALSHI_API_PFX + "/markets" + params,
                headers=headers, timeout=10,
            )
            if r.status_code == 200:
                markets = r.json().get("markets", [])
                if "series_ticker=KXBTCD" in params:
                    if markets:
                        return markets
                else:
                    btc = [m for m in markets
                           if "btc" in m.get("ticker", "").lower()
                           or "bitcoin" in m.get("title", "").lower()]
                    if btc:
                        return btc
        except requests.RequestException:
            pass
    return []


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    gain = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma20"] = df["price"].rolling(20).mean()
    df["sma50"] = df["price"].rolling(50).mean()
    df["ema20"] = df["price"].ewm(span=20, adjust=False).mean()
    df["rsi14"] = compute_rsi(df["price"])
    return df


def get_signal(df: pd.DataFrame) -> tuple[str, str, int]:
    """Returns (label, direction, bull_score)."""
    last = df.dropna(subset=["sma20", "sma50", "rsi14"]).iloc[-1]
    rsi  = last["rsi14"]
    ma_bull  = last["sma20"] > last["sma50"]
    ema_bull = last["price"] > last["ema20"]
    score = 0
    score += 2 if rsi < 30 else (1 if rsi < 40 else 0)
    score -= 2 if rsi > 70 else (1 if rsi > 60 else 0)
    score += 1 if ma_bull  else -1
    score += 1 if ema_bull else -1
    if score >= 3:   return "STRONG BUY",  "up",      score
    if score >= 1:   return "BUY",          "up",      score
    if score <= -3:  return "STRONG SELL",  "down",    score
    if score <= -1:  return "SELL",          "down",    score
    return "HOLD", "neutral", score


SIGNAL_COLORS = {
    "STRONG BUY":  "#00C853",
    "BUY":         "#69F0AE",
    "HOLD":        "#FFD600",
    "SELL":        "#FF5252",
    "STRONG SELL": "#D50000",
}


# ---------------------------------------------------------------------------
# Config / state helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text()))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def load_trader_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="BTC Dashboard", layout="wide", page_icon="₿")
st_autorefresh(interval=30_000, key="btc_refresh")

st.title("₿  BTC/USD Live Dashboard")
st.caption(f"Auto-refreshes every 30 s  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------------------------------------------------------
# Fetch all data
# ---------------------------------------------------------------------------

with st.spinner("Loading market data…"):
    price_data    = fetch_btc_price()
    ohlc_df       = fetch_ohlc(days=7)
    hist_df       = fetch_price_history(days=30)
    kalshi_mkts   = fetch_kalshi_markets()
    trader_state  = load_trader_state()
    config        = load_config()

hist_df     = add_indicators(hist_df)
sig_label, sig_dir, sig_score = get_signal(hist_df)
sig_color   = SIGNAL_COLORS.get(sig_label, "#FFD600")
rsi_series  = hist_df["rsi14"].dropna()
current_rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else None

price     = price_data["usd"]
chg_24h   = price_data.get("usd_24h_change", 0.0)
vol_24h   = price_data.get("usd_24h_vol", 0.0)

# ---------------------------------------------------------------------------
# Metric row
# ---------------------------------------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("BTC Price", f"${price:,.2f}", f"{chg_24h:+.2f}%")
c2.metric("24h Volume", f"${vol_24h / 1e9:.2f}B")

if current_rsi is not None:
    rsi_lbl = "Overbought" if current_rsi > 70 else ("Oversold" if current_rsi < 30 else "Neutral")
    c3.metric("RSI (14)", f"{current_rsi:.1f}", rsi_lbl)

c4.markdown(
    f"<div style='text-align:center;padding:12px 6px;border-radius:8px;"
    f"background:{sig_color}18;border:2px solid {sig_color}'>"
    f"<div style='font-size:0.75em;color:#aaa;margin-bottom:4px'>Strategy Signal</div>"
    f"<div style='font-size:1.5em;font-weight:700;color:{sig_color}'>{sig_label}</div>"
    f"<div style='font-size:0.75em;color:#aaa'>score {sig_score:+d}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

if kalshi_mkts:
    m0 = kalshi_mkts[0]
    yes_p = m0.get("yes_ask") or m0.get("last_price")
    c5.metric(
        f"Kalshi: {m0.get('ticker','BTC')[:16]}",
        f"{yes_p}¢" if isinstance(yes_p, (int, float)) else "N/A",
    )
else:
    c5.metric("Kalshi", "N/A", "no markets")

st.divider()

# ---------------------------------------------------------------------------
# Price chart + RSI
# ---------------------------------------------------------------------------

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
    vertical_spacing=0.04,
    subplot_titles=("BTC/USD — 7-day OHLC + 30-day MAs", "RSI (14)"),
)

fig.add_trace(go.Candlestick(
    x=ohlc_df["ts"], open=ohlc_df["open"], high=ohlc_df["high"],
    low=ohlc_df["low"], close=ohlc_df["close"], name="OHLC",
    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
), row=1, col=1)

for col_name, label, color, dash in [
    ("sma20", "SMA 20", "#FFA726", "solid"),
    ("sma50", "SMA 50", "#42A5F5", "solid"),
    ("ema20", "EMA 20", "#AB47BC", "dash"),
]:
    fig.add_trace(go.Scatter(
        x=hist_df["ts"], y=hist_df[col_name], name=label,
        line=dict(color=color, width=1.5, dash=dash),
    ), row=1, col=1)

fig.add_trace(go.Scatter(
    x=hist_df["ts"], y=hist_df["rsi14"], name="RSI 14",
    line=dict(color="#66BB6A", width=2),
    fill="tozeroy", fillcolor="rgba(102,187,106,0.05)",
), row=2, col=1)

for y_val, clr in [(70, "red"), (30, "green")]:
    fig.add_hline(y=y_val, line_dash="dash", line_color=clr, line_width=1, row=2, col=1)
fig.add_hrect(y0=70, y1=100, fillcolor="red",   opacity=0.07, row=2, col=1, line_width=0)
fig.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.07, row=2, col=1, line_width=0)
fig.add_hline(y=50, line_dash="dot", line_color="grey", line_width=1, row=2, col=1)

fig.update_layout(
    height=720, xaxis_rangeslider_visible=False, template="plotly_dark",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=60, b=10),
)
fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Kalshi markets table
# ---------------------------------------------------------------------------

st.subheader("Kalshi BTC Markets")
if kalshi_mkts:
    rows = []
    for m in kalshi_mkts:
        ct = (m.get("close_time") or "")[:10]
        rows.append({
            "Ticker":        m.get("ticker", ""),
            "Title":         m.get("title", ""),
            "Yes Ask (¢)":   m.get("yes_ask", "-"),
            "No Ask (¢)":    m.get("no_ask", "-"),
            "Last (¢)":      m.get("last_price", "-"),
            "Volume":        f"{m.get('volume', 0):,}",
            "Closes":        ct or "-",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No open Kalshi BTC markets — check credentials or markets are closed.")

st.divider()

# ---------------------------------------------------------------------------
# Trading dashboard
# ---------------------------------------------------------------------------

st.header("Automated Trading")

# Trader daemon status banner
is_running = trader_state.get("is_running", False)
is_enabled = trader_state.get("enabled", False)
is_dry_run = trader_state.get("dry_run", True)

if is_running and is_enabled and not is_dry_run:
    st.error("🔴  LIVE TRADING ACTIVE — real orders are being placed on Kalshi")
elif is_running and is_enabled and is_dry_run:
    st.warning("🟡  Trader running in DRY RUN mode — signals logged, no real orders sent")
elif is_running:
    st.info("⚪  Trader daemon is running but trading is disabled")
else:
    st.info("⚫  Trader daemon is not running.  Start it with:  `python trader.py`")

# Portfolio snapshot from trader state
tc1, tc2, tc3, tc4 = st.columns(4)
bal_cents = trader_state.get("balance_cents")
tc1.metric("Kalshi Balance", f"${bal_cents/100:.2f}" if bal_cents is not None else "—")

positions = trader_state.get("active_positions", [])
total_unreal = sum((p.get("unrealized_pnl") or 0) for p in positions)
tc2.metric("Open Positions", len(positions))
tc3.metric("Unrealized P&L", f"${total_unreal/100:.2f}" if positions else "—")

last_sig_time = trader_state.get("last_signal_time", "")
tc4.metric("Last Signal Change",
           last_sig_time[:16].replace("T", " ") if last_sig_time else "—")

# Active positions detail
if positions:
    st.subheader("Open Positions")
    pos_rows = []
    for p in positions:
        pos_rows.append({
            "Ticker":          p.get("ticker", ""),
            "Position":        p.get("position", 0),
            "Avg Cost (¢)":    round((p.get("total_cost") or 0) / max(abs(p.get("position", 1)), 1), 1),
            "Unrealized P&L":  f"${(p.get('unrealized_pnl') or 0)/100:.2f}",
            "Realized P&L":    f"${(p.get('realized_pnl') or 0)/100:.2f}",
        })
    st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

# Recent orders
recent_orders = trader_state.get("recent_orders", [])
if recent_orders:
    st.subheader("Recent Orders")
    ord_rows = []
    for o in recent_orders[:20]:
        result = o.get("result")
        if isinstance(result, dict):
            status = result.get("order", {}).get("status", "sent")
        else:
            status = str(result) if result else "—"
        ord_rows.append({
            "Time":      (o.get("timestamp") or "")[:16].replace("T", " "),
            "Signal":    o.get("signal", ""),
            "Ticker":    o.get("ticker", ""),
            "Side":      o.get("side", "").upper(),
            "Count":     o.get("count", 0),
            "Price (¢)": o.get("price_cents", "—"),
            "Cost":      f"${o.get('cost_usd', 0):.2f}",
            "Status":    status,
            "Dry Run":   "✓" if o.get("dry_run") else "✗",
        })
    st.dataframe(pd.DataFrame(ord_rows), use_container_width=True, hide_index=True)

# Errors
errs = trader_state.get("errors", [])
if errs:
    with st.expander(f"⚠ {len(errs)} error(s) from last cycle"):
        for e in errs:
            st.text(e)

st.divider()

# ---------------------------------------------------------------------------
# Trading configuration panel
# ---------------------------------------------------------------------------

st.subheader("Trading Configuration")
st.caption("Changes are written to `trading_config.json` and picked up by the trader daemon on its next cycle.")

with st.form("trading_config_form"):
    col_a, col_b = st.columns(2)

    with col_a:
        dry_run = st.checkbox(
            "Dry Run (log signals but place NO real orders)",
            value=config.get("dry_run", True),
        )
        enabled = st.checkbox(
            "Enable live trading",
            value=config.get("enabled", False),
            disabled=dry_run,
            help="Only available when Dry Run is unchecked",
        )
        only_on_change = st.checkbox(
            "Only trade on signal change",
            value=config.get("only_on_change", True),
        )
        series_ticker = st.text_input(
            "Kalshi series ticker",
            value=config.get("series_ticker", "KXBTCD"),
        )

    with col_b:
        max_contracts = st.number_input(
            "Max contracts per trade",
            min_value=1, max_value=100,
            value=int(config.get("max_contracts", 5)),
        )
        max_risk = st.number_input(
            "Max open risk (USD)",
            min_value=1.0, max_value=10_000.0, step=5.0,
            value=float(config.get("max_open_risk_usd", 50.0)),
        )
        loop_interval = st.number_input(
            "Cycle interval (seconds)",
            min_value=30, max_value=3600,
            value=int(config.get("loop_interval_sec", 60)),
        )

    if enabled and not dry_run:
        st.warning(
            "You are enabling LIVE trading. Real orders will be placed on Kalshi. "
            "Make sure your max risk and contract limits are set correctly before saving."
        )

    submitted = st.form_submit_button("Save Configuration")
    if submitted:
        new_cfg = {
            "enabled":           enabled and not dry_run,
            "dry_run":           dry_run,
            "series_ticker":     series_ticker,
            "max_contracts":     int(max_contracts),
            "max_open_risk_usd": float(max_risk),
            "only_on_change":    only_on_change,
            "loop_interval_sec": int(loop_interval),
        }
        save_config(new_cfg)
        st.success("Configuration saved. The trader daemon will pick it up on the next cycle.")
        st.cache_data.clear()
