# SickSeven — BTC / Kalshi Automated Trading System

A real-time Bitcoin analysis and autonomous trading engine. Generates directional
signals from technical indicators, an LLM probability model, and live news, then
executes binary-option orders on Kalshi.

---

## What it does

**Long-term strategy (trader daemon)**
Runs every 60 seconds. Pulls 30 days of hourly BTC prices from CoinGecko, scores
5 technical factors (RSI, MACD, MA crossover, EMA200 trend, Bollinger Bands), and
places Kalshi binary-option orders when there is a clear directional edge.

**Short-term monitor (1m / 5m / 15m)**
A separate Streamlit dashboard with live Kraken candlestick charts, short-term
indicators (VWAP, fast/slow EMA cross), and a real-time signal badge. Refreshes
every 15 seconds.

**LLM probability model**
Uses the Claude API (claude-sonnet-4-6) to blend the technical signal with live BTC
news headlines, Fear & Greed Index, and BTC dominance into a 0–100% probability
estimate. Displayed as a gauge on the short-term monitor.

---

## Project Structure

```
sickseven/
├── .env                    # API keys and secrets (never commit)
├── requirements.txt        # Python dependencies
│
├── strategy.py             # All indicator + signal + sizing logic (long + short term)
├── kalshi_client.py        # Kalshi API v2 wrapper (RSA auth + retry)
├── news_fetcher.py         # BTC news aggregator (8 RSS feeds + 3 Reddit, no keys)
├── probability_model.py    # LLM probability model (Claude API)
│
├── trader.py               # Autonomous trading daemon
├── dashboard.py            # Long-term trading control UI (port 8501)
├── monitor.py              # Short-term market monitor (port 8502)
│
├── trading_config.json     # Runtime config (auto-created on first run)
├── trading_state.json      # Live state written by daemon, read by dashboards
└── trader.log              # Rolling log of every trader cycle
```

---

## File Descriptions

### `.env`
Holds all secrets. Never share or commit this file.

| Variable | What it is |
|---|---|
| `GECKO_API` | CoinGecko demo API key for 30-day hourly price history |
| `KALSHI_API_KEY` | Your Kalshi key UUID (identifies who you are) |
| `KALSHI_PRIV` | RSA private key used to sign every Kalshi request |
| `ANTHROPIC_API_KEY` | Claude API key for the LLM probability model |

Kalshi uses RSA signature authentication — every API request is signed with
your private key, not a simple bearer token.

---

### `strategy.py`
All trading logic in one place. No API calls, no I/O — pure computation.
Everything else imports from here; never duplicate indicator logic in other files.

**Long-term signal (5 factors, ±7 max score):**

| Factor | Bullish | Bearish |
|---|---|---|
| RSI (±2) | RSI < 30 → +2, RSI < 40 → +1 | RSI > 70 → -2, RSI > 60 → -1 |
| MACD (±2) | Line > signal +1, line > 0 +1 | Line < signal -1, line < 0 -1 |
| MA crossover (±1) | SMA20 > SMA50 | SMA20 < SMA50 |
| EMA200 trend (±1) | Price > EMA200 | Price < EMA200 |
| Bollinger %B (±1) | %B ≤ 0.05 (at lower band) | %B ≥ 0.95 (at upper band) |

Score → signal: ≥4 = STRONG BUY, 2–3 = BUY, -1–1 = HOLD, -2–-3 = SELL, ≤-4 = STRONG SELL

**Short-term signal (1m / 5m / 15m):** Same RSI and MACD factors, but replaces the
SMA/EMA200 trend filters with a fast/slow EMA cross (±1) and VWAP comparison (±1).
Indicator periods adapt to the selected timeframe.

**Position sizing:** ATR-based volatility scaling. At 2× normal volatility, position
size halves to keep dollar risk roughly constant.

---

### `kalshi_client.py`
Low-level Kalshi API v2 wrapper. Every request is RSA-signed with a fresh
timestamp. Includes 3-attempt exponential backoff retry (4xx errors are not retried).

Key functions: `get_balance`, `get_markets`, `get_positions`, `get_orders`,
`place_order`, `close_position` (for stop-loss), `cancel_all_resting` (emergency).

---

### `news_fetcher.py`
Aggregates BTC headlines from 11 free sources with no API keys required:
- **8 RSS feeds**: CoinDesk, CoinTelegraph, Bitcoin Magazine, Decrypt, Bitcoinist,
  NewsBTC, CryptoNews, BeInCrypto
- **3 Reddit JSON feeds**: r/Bitcoin, r/CryptoCurrency (BTC-filtered), r/btc

Fetches all sources in parallel, deduplicates by title, and returns headlines
sorted newest-first with age in minutes.

---

### `probability_model.py`
Calls the Claude API to estimate the probability BTC moves up over the next ~4 hours.
Inputs: current technical indicators, live news headlines, Fear & Greed Index,
BTC dominance, and total crypto market cap change.

The final probability is a 50/50 blend of the technical score and the LLM estimate.
Falls back to technical-only if the Claude API is unavailable.

---

### `trader.py`
The autonomous daemon. Run it in a terminal and leave it running.

Each cycle (every 60s by default):
1. Fetch 30 days of hourly BTC prices from CoinGecko
2. Compute indicators and generate a signal
3. Refresh Kalshi portfolio (balance + open positions)
4. Run stop-loss checks — close any position down more than `stop_loss_pct`
5. Check all guard rails (enabled? cooldown? risk limit? signal changed?)
6. Select the best Kalshi BTC market and size the order
7. Place the order (or log it in dry-run mode)
8. Write everything to `trading_state.json`

---

### `dashboard.py`
Long-term trading control UI. Auto-refreshes every 30 seconds.

- 7-day OHLC candlestick chart with Bollinger Bands, SMA20/50, EMA20, EMA200,
  RSI subplot, and MACD subplot
- Live Kalshi market odds table
- Portfolio snapshot: balance, open positions, unrealized P&L
- Trading configuration form (enable/disable, risk limits, stop-loss, cooldown)

---

### `monitor.py`
Short-term market monitor. Auto-refreshes every 15 seconds.

- 1m / 5m / 15m Kraken candlestick charts with EMAs, VWAP, Bollinger Bands,
  RSI, and MACD
- Short-term technical signal badge (BUY / SELL / HOLD + score)
- LLM probability gauge (Claude API estimate, refreshes every 5 minutes)
- Live BTC news feed (last 3 hours, up to 20 headlines)
- Fear & Greed Index + BTC dominance snapshot
- Kalshi market odds

---

## Setup

**1. Install dependencies**
```
pip install -r requirements.txt
```

**2. Configure `.env`**
```
GECKO_API         = your_coingecko_demo_key
KALSHI_API_KEY    = your_kalshi_uuid
KALSHI_PRIV       = 'your_rsa_private_key_base64'
ANTHROPIC_API_KEY = your_anthropic_key
```

**3. Start the trader daemon** (terminal 1)
```
python trader.py
```
Creates `trading_config.json` with `dry_run: true` and `enabled: false` on first run.
No orders will be placed until you explicitly enable them.

**4. Start the long-term dashboard** (terminal 2)
```
streamlit run dashboard.py
```
Open `http://localhost:8501`

**5. Start the short-term monitor** (terminal 3)
```
streamlit run monitor.py --server.port 8502
```
Open `http://localhost:8502`

---

## Going Live

Before enabling real trading, run in dry-run mode for at least several cycles to
confirm signals look correct and order sizing is reasonable.

When ready:
1. Open `http://localhost:8501`
2. Scroll to **Trading Configuration**
3. Uncheck **Dry Run** → check **Enable live trading**
4. Set your **Max contracts** and **Max open risk (USD)** conservatively
5. Click **Save Configuration**

The daemon picks up the change within 60 seconds.

**Emergency stop:** set `"enabled": false` directly in `trading_config.json`,
or use the dashboard toggle. Takes effect within one cycle.

---

## How the Signal Maps to Kalshi

Kalshi BTC markets are binary: *"Will Bitcoin be above $X on [date]?"*

| Signal | Side | Logic |
|---|---|---|
| STRONG BUY / BUY | Buy YES | Expect BTC to rise above the strike |
| STRONG SELL / SELL | Buy NO | Expect BTC to stay below the strike |
| HOLD | No order | No clear edge — stay flat |

The market selector targets contracts where the relevant side is priced 15–85 cents
(genuinely uncertain outcome — maximum value for your signal).

---

## Risk Warnings

- This system places real financial bets on Kalshi using your account funds.
- Past indicator signals do not guarantee future performance.
- Kalshi binary options can expire worthless — you can lose 100% of what you bet.
- Start with `max_contracts: 1` and `max_open_risk_usd: 10` until you have validated
  the strategy over multiple live cycles.
- Always monitor `trader.log` when live trading is active.
