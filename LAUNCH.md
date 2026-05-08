# SickSeven — Launch Guide

Step-by-step instructions for setting up and running the full system from scratch.

---

## Prerequisites

- Python 3.10 or later
- `pip` (comes with Python)
- Four open terminal windows (or tabs)
- A Kalshi account with funds deposited

---

## Step 1 — Install dependencies

Run once. From the `sickseven/` directory:

```
pip install -r requirements.txt
```

---

## Step 2 — Configure `.env`

The `.env` file must exist in the `sickseven/` directory and contain four variables:

```
GECKO_API         = your_coingecko_demo_key
KALSHI_API_KEY    = your_kalshi_uuid
KALSHI_PRIV       = 'your_rsa_private_key_pem_base64'
ANTHROPIC_API_KEY = your_anthropic_key
```

### Where to get each key

| Key | Where to get it |
|---|---|
| `GECKO_API` | coingecko.com → Developer Dashboard → Demo API key (free) |
| `KALSHI_API_KEY` | kalshi.com → Settings → API → Create Key → copy the UUID |
| `KALSHI_PRIV` | Same Kalshi page — download the RSA private key, paste the base64 content (no headers, no line breaks) in single quotes |
| `ANTHROPIC_API_KEY` | console.anthropic.com → API Keys → Create Key |

### KALSHI_PRIV formatting

The key must be pasted as a single quoted string, no PEM headers, no newlines:

```
KALSHI_PRIV = 'MIIEowIBAAKCAQEA...rest of key...'
```

The code strips all whitespace and re-wraps at 64 characters internally — you do not need to format it manually.

---

## Step 3 — Verify connectivity (optional but recommended)

Before launching anything, confirm your keys work:

```
python -c "from price_feed import get_composite_price; r = get_composite_price(); print(f'BTC: \${r[\"price\"]:,.0f}  ({r[\"source_count\"]}/4 exchanges)')"
```

Expected output: `BTC: $95,142  (4/4 exchanges)`

---

## Step 4 — Start each process

Open four terminal windows. Start them in this order:

### Terminal 1 — Trader daemon

```
python trader.py
```

**What it does:** Fetches 30-day BTC price history every 60 seconds, computes indicators, and places Kalshi orders when there is a clear signal.

**First run:** Creates `trading_config.json` with `enabled: false` and `dry_run: true`. No orders will be placed until you explicitly enable trading.

**What you should see:**
```
=== BTC Trader daemon started ===
Signal unchanged: HOLD  RSI=52.3  score=+0
Next cycle in 60s
```

If you see `Signal computation failed` on the first run, check your `GECKO_API` key.

---

### Terminal 2 — Trading dashboard (port 8501)

```
streamlit run dashboard.py
```

Open: `http://localhost:8501`

**What it does:** Shows a 7-day OHLC chart with Bollinger Bands, RSI, and MACD. Displays live Kalshi market odds, your portfolio balance, open positions, and a configuration form.

**What you should see:** A chart loads within a few seconds. If the chart is blank, check your `GECKO_API` key. If Kalshi data is missing, check `KALSHI_API_KEY` and `KALSHI_PRIV`.

---

### Terminal 3 — Short-term monitor (port 8502)

```
streamlit run monitor.py --server.port 8502
```

Open: `http://localhost:8502`

**What it does:** Real-time 1m/5m/15m Kraken candlestick charts, short-term technical signal badge, LLM probability gauge (Claude), live BTC news feed, Trump tweet signal, and Fear & Greed Index.

**What you should see:** Price metrics across the top row, including `BRTI ≈ Price` showing the composite price from 4 exchanges with a spread reading. The LLM gauge takes up to 10 seconds on first load (it calls the Claude API).

If `ANTHROPIC_API_KEY` is missing, the LLM gauge shows `0%` — everything else still works.

---

### Terminal 4 — Trump tweet watcher (optional)

```
python trump_watcher.py
```

**What it does:** Polls Trump's Truth Social feed every 30 seconds. Classifies each new tweet for BTC/USD market impact using Claude Haiku and writes the result to `trump_state.json`. The probability model and monitor pick it up automatically.

**This process is optional.** If you skip it, everything else runs normally — the Trump signal card on the monitor just stays empty.

---

## Step 5 — Confirm the system is healthy

Check all of these before enabling live trading:

| Check | Where to look |
|---|---|
| Daemon is running | Terminal 1 logs a cycle every 60s without errors |
| Price is live | `http://localhost:8502` → BRTI ≈ Price shows a realistic BTC price |
| Signal is computing | `http://localhost:8502` → Signal badge shows BUY / SELL / HOLD |
| Kalshi balance visible | `http://localhost:8501` → Portfolio section shows your balance |
| LLM gauge working | `http://localhost:8502` → Probability gauge shows a number other than 0 |

---

## Step 6 — Run in dry-run mode first

The system starts in dry-run mode by default. In this mode it computes every signal and logs what orders it *would* place, but sends nothing to Kalshi.

Watch Terminal 1 for lines like:

```
[DRY RUN] Would place: BUY | RSI=38.2 MACD=▲ score=+3 | vol=1.00 greek=0.61 → 1×@ 48¢ | Δ=24.6¢/$1k
```

Run for at least a few cycles (5–10 minutes) and confirm:
- Signals look reasonable for the current market
- Order sizes are what you expect
- No repeated errors in the log

---

## Step 7 — Enable live trading

When you are ready to trade with real money:

1. Open `http://localhost:8501`
2. Scroll down to **Trading Configuration**
3. Set your position limits (see table below)
4. Uncheck **Dry Run**
5. Check **Enable live trading**
6. Click **Save Configuration**

The daemon picks up the change within 60 seconds. You will see orders placed in Terminal 1.

### Recommended position limits by account size

| Account size | max_contracts | max_open_risk_usd | stop_loss_pct | cooldown_minutes |
|---|---|---|---|---|
| < $50 | 1 | $3 | 0.30 | 30 |
| $200 | 3 | $20 | 0.35 | 30 |
| $500 | 5 | $50 | 0.40 | 15 |
| $1,000+ | 8 | $100 | 0.40 | 15 |

**Rule of thumb:** `max_open_risk_usd` = 10% of your total account. Even a total wipeout of all open positions costs you at most 10%.

---

## Stopping the system

**Normal stop:** Press `Ctrl+C` in each terminal. The trader daemon writes `is_running: false` to `trading_state.json` before it exits.

**Emergency stop (fastest):** Edit `trading_config.json` and set `"enabled": false`. The daemon reads this file every cycle — it takes effect within 60 seconds without restarting anything.

```json
{
  "enabled": false,
  ...
}
```

---

## Files written at runtime

| File | Written by | Purpose |
|---|---|---|
| `trading_config.json` | dashboard.py | Your risk settings — edit here or via the dashboard |
| `trading_state.json` | trader.py | Live state: signal, balance, positions, recent orders |
| `trump_state.json` | trump_watcher.py | Latest tweet classification and probability adjustment |
| `trader.log` | trader.py | Rolling log of every cycle — monitor this when live |
| `trump_watcher.log` | trump_watcher.py | Log of every detected tweet and its classification |

---

## Common issues

**`Signal computation failed: 429`**
CoinGecko rate limit hit. The free tier allows ~30 requests/min. Increase `loop_interval_sec` to 90 in `trading_config.json`.

**`Portfolio refresh failed`**
Kalshi API auth error. Double-check `KALSHI_API_KEY` (the UUID) and `KALSHI_PRIV` (the base64 private key, no PEM headers).

**LLM gauge shows `0%` or `Error`**
`ANTHROPIC_API_KEY` is missing or invalid. The rest of the system keeps running.

**`No suitable market found`**
No Kalshi BTC markets are currently open with odds in the 15–85¢ range. This is normal outside of active trading hours. The daemon logs this and skips the cycle.

**Monitor shows `Could not fetch Kraken data`**
Kraken public API is temporarily down. The monitor retries on the next 15-second refresh automatically.

**BRTI price shows `0` or `1/4 exchanges`**
One or more exchange APIs is temporarily unavailable. The system still uses whichever exchanges respond. If all four fail, it falls back to the Kraken ticker.
