# BTC / Kalshi Automated Trading System

A real-time Bitcoin dashboard and autonomous trading engine that generates
technical signals from CoinGecko market data and executes binary-option orders
on Kalshi.

---

## Project Structure

```
sickseven/
├── .env                  # API keys and secrets (never commit this)
├── requirements.txt      # Python dependencies
├── kalshi_client.py      # Kalshi API wrapper
├── strategy.py           # Indicators, signal engine, order logic
├── trader.py             # Autonomous trading daemon
├── dashboard.py          # Streamlit live dashboard
├── trading_config.json   # Runtime config (auto-created on first run)
├── trading_state.json    # Live state written by trader daemon
└── trader.log            # Rolling log of all trader activity
```

---

## File Descriptions

### `.env`
Holds all secrets. Never share or commit this file.

| Variable | What it is |
|---|---|
| `GECKO_API` | CoinGecko demo API key for price data |
| `KALSHI_API_KEY` | Your Kalshi key UUID (the "who you are") |
| `KALSHI_PRIV` | RSA private key used to sign every Kalshi request |

Kalshi uses RSA signature authentication — every API request is signed with
your private key, not a simple bearer token. The `KALSHI_API_KEY` identifies
you and `KALSHI_PRIV` proves it.

---

### `kalshi_client.py`
Low-level Kalshi API v2 wrapper. All other files import from here.

**What it does:**
- Loads and reconstructs your RSA private key from `.env`
- Signs every outbound request (timestamp + method + path → SHA256/PKCS1v15)
- Exposes clean functions for every API action you need

**Key functions:**

| Function | Description |
|---|---|
| `get_balance()` | Returns your Kalshi account balance in cents |
| `get_markets(series_ticker)` | Lists open markets for a given series (e.g. `KXBTCD`) |
| `search_markets(keyword)` | Fallback keyword search across all open markets |
| `get_positions()` | Your current open positions |
| `get_orders(status)` | Resting, filled, or cancelled orders |
| `place_order(...)` | Place a limit or market order |
| `cancel_order(order_id)` | Cancel a single resting order |
| `cancel_all_resting()` | Emergency: cancel every open order |

You generally do not call this file directly — `trader.py` uses it.

---

### `strategy.py`
All trading logic lives here. No API calls, no I/O — pure computation.

**What it does:**

**1. Indicators** (`compute_indicators`)
Takes a pandas Series of hourly closing prices and computes:
- SMA 20 and SMA 50 (simple moving averages)
- EMA 20 (exponential moving average)
- RSI 14 (relative strength index)

**2. Signal generation** (`generate_signal`)
Scores each indicator for bullishness/bearishness:

| Condition | Score |
|---|---|
| RSI < 30 (oversold) | +2 |
| RSI < 40 | +1 |
| RSI > 60 | −1 |
| RSI > 70 (overbought) | −2 |
| SMA20 > SMA50 (golden cross) | +1 |
| SMA20 < SMA50 (death cross) | −1 |
| Price > EMA20 | +1 |
| Price < EMA20 | −1 |

Total score maps to a signal:

| Score | Signal | Direction |
|---|---|---|
| ≥ 3 | STRONG BUY | up |
| 1–2 | BUY | up |
| 0 | HOLD | neutral |
| −1 to −2 | SELL | down |
| ≤ −3 | STRONG SELL | down |

**3. Market selection** (`select_market`)
Given a direction (up/down) and a list of Kalshi markets, picks the best one:
- Filters for markets where the relevant-side ask is between 15–85 cents
  (avoids near-certain or near-impossible contracts)
- Ranks by volume × proximity to 50 cents (most liquid, most informative)

**4. Order sizing** (`compute_order`)
Converts a signal + market into order parameters:
- STRONG signal → 100% of `max_contracts`
- Regular signal → 50% of `max_contracts`
- Sets a limit price 1 cent above the current ask to ensure a quick fill

---

### `trader.py`
The autonomous trading daemon. Run this in a separate terminal and leave it
running. It operates completely independently of the dashboard.

**What it does each cycle (every 60 seconds by default):**

1. Fetches 30 days of hourly BTC price history from CoinGecko
2. Computes indicators and generates a signal via `strategy.py`
3. Refreshes your Kalshi portfolio (balance + open positions)
4. Checks all guard rails (trading enabled? signal changed? risk limit OK?)
5. Selects the best Kalshi BTC market and sizes the order
6. Places the order (or logs it if dry run is on)
7. Writes everything to `trading_state.json` for the dashboard

**Guard rails (in order):**
- `enabled: false` → no orders placed (default)
- `dry_run: true` → signals logged, no real orders sent (default)
- Signal unchanged + `only_on_change: true` → skip (avoids spamming)
- HOLD signal → skip (no trade when there is no edge)
- Open risk ≥ `max_open_risk_usd` → skip (position size hard cap)

**Config changes take effect on the next cycle** — you never need to restart
the daemon.

---

### `dashboard.py`
The Streamlit web UI. Auto-refreshes every 30 seconds.

**Sections:**

**Top row — live metrics**
- BTC price and 24-hour change
- 24-hour trading volume
- Current RSI value and zone (Neutral / Overbought / Oversold)
- Strategy signal badge (color-coded STRONG BUY → STRONG SELL)
- Best open Kalshi BTC market yes-price

**Price chart**
- 7-day OHLC candlesticks
- SMA 20 (orange), SMA 50 (blue), EMA 20 (purple dashed)
- RSI subplot with overbought (red) and oversold (green) zones

**Kalshi Markets table**
- All open BTC markets with yes/no ask prices, volume, and expiry

**Automated Trading section**
- Status banner: shows whether the daemon is running, and whether trading
  is live or in dry-run mode
- Portfolio snapshot: balance, open position count, unrealized P&L
- Open positions table
- Recent orders table (last 20)
- Error log from the most recent trader cycle

**Trading Configuration form**
- Toggle dry run / enable live trading
- Set max contracts per trade, max open risk in USD
- Set the Kalshi series ticker and cycle interval
- Changes are saved to `trading_config.json` immediately

---

### `trading_config.json`
Created automatically on the first run of `trader.py`. Edit it directly or
use the dashboard config form.

```json
{
  "enabled":           false,
  "dry_run":           true,
  "series_ticker":     "KXBTCD",
  "max_contracts":     5,
  "max_open_risk_usd": 50.0,
  "only_on_change":    true,
  "loop_interval_sec": 60
}
```

| Field | Description |
|---|---|
| `enabled` | Master switch — must be `true` to place real orders |
| `dry_run` | Log signals only, no real orders. Overrides `enabled`. |
| `series_ticker` | Kalshi market series to trade (`KXBTCD` = daily BTC) |
| `max_contracts` | Max contracts placed per signal (strong signal uses 100%, regular uses 50%) |
| `max_open_risk_usd` | Stop placing orders when total open exposure exceeds this dollar amount |
| `only_on_change` | Only trade when the signal changes (prevents repeated orders on the same signal) |
| `loop_interval_sec` | How often the trader daemon runs, in seconds |

---

### `trading_state.json`
Written by `trader.py` at the end of every cycle. The dashboard reads this
file on every refresh. You can inspect it directly to debug the trader.

---

### `trader.log`
A plain-text log of every trader cycle: signals, decisions, orders placed,
errors. Check this first when something goes wrong.

---

## Setup

**1. Install dependencies**
```
pip install -r requirements.txt
```

**2. Verify your `.env`**
Make sure `GECKO_API`, `KALSHI_API_KEY`, and `KALSHI_PRIV` are all set.

**3. Start the trader daemon** (terminal 1)
```
python trader.py
```
On first run this creates `trading_config.json` with `dry_run: true` and
`enabled: false`. No orders will be placed.

**4. Start the dashboard** (terminal 2)
```
streamlit run dashboard.py
```
Open `http://localhost:8501` in your browser.

---

## Going Live

Before enabling real trading, run in dry-run mode for at least a few cycles
to confirm that:
- The daemon is picking up signal changes correctly
- The Kalshi market selection looks sensible
- The order sizes and costs are what you expect

When you are ready:

1. Open the dashboard at `http://localhost:8501`
2. Scroll to **Trading Configuration**
3. Uncheck **Dry Run**
4. Check **Enable live trading**
5. Set your **Max contracts** and **Max open risk (USD)** limits
6. Click **Save Configuration**

The daemon will pick up the change on its next cycle (within 60 seconds) and
begin placing real orders on Kalshi.

To stop trading at any time: re-check **Dry Run** or uncheck **Enable live
trading** in the dashboard, then click Save. Takes effect within one cycle.
Emergency stop: edit `trading_config.json` directly and set `"enabled": false`.

---

## How the Signal Maps to Kalshi Orders

Kalshi BTC markets are binary questions of the form:
*"Will Bitcoin be above $X on [date]?"*

| Signal | Side | Reasoning |
|---|---|---|
| STRONG BUY / BUY | Buy YES | We expect BTC to rise — bet it closes above the strike |
| STRONG SELL / SELL | Buy NO | We expect BTC to fall — bet it closes below the strike |
| HOLD | No order | No clear edge, stay flat |

The market selector prefers contracts where the relevant side is priced
between 15–85 cents, meaning the market considers the outcome genuinely
uncertain. This is where your signal has the most value.

---

## Risk Warnings

- This system places real financial bets on Kalshi using your account funds.
- Past indicator signals do not guarantee future performance.
- Kalshi contracts can expire worthless — you can lose 100% of what you bet.
- Start with the minimum `max_contracts` (1) and `max_open_risk_usd` ($10)
  until you have validated the strategy over multiple real cycles.
- Always monitor `trader.log` when live trading is active.
