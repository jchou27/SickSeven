"""
Autonomous BTC trading daemon.

Usage:
    python trader.py

The daemon reads trading_config.json each cycle, so you can change settings
(including enable/disable) at runtime without restarting. State is written to
trading_state.json for the dashboard to display.
"""
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

import kalshi_client as kc
from strategy import compute_indicators, generate_signal, select_market, compute_order

load_dotenv()

GECKO_API_KEY  = os.getenv("GECKO_API", "")
GECKO_BASE     = "https://api.coingecko.com/api/v3"
STATE_FILE     = Path("trading_state.json")
CONFIG_FILE    = Path("trading_config.json")
LOG_FILE       = Path("trader.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("trader")

DEFAULT_CONFIG: dict = {
    "enabled":            False,   # must be set True to place real orders
    "series_ticker":      "KXBTCD",
    "max_contracts":      5,
    "max_open_risk_usd":  50.0,    # stop placing orders when open risk >= this
    "only_on_change":     True,    # skip order if signal hasn't changed
    "loop_interval_sec":  60,
    "dry_run":            True,    # log orders but don't actually send them
}


# ---------------------------------------------------------------------------
# Config / state helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text()))
        except Exception as e:
            log.warning(f"Bad config file: {e}")
    return cfg


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return _empty_state()


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _empty_state() -> dict:
    return {
        "is_running":        False,
        "enabled":           False,
        "dry_run":           True,
        "last_run":          None,
        "last_signal":       None,
        "last_signal_time":  None,
        "current_price":     None,
        "current_indicators": {},
        "balance_cents":     None,
        "active_positions":  [],
        "recent_orders":     [],
        "total_realized_pnl_cents": 0,
        "errors":            [],
    }


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_price_history() -> pd.Series:
    url = (f"{GECKO_BASE}/coins/bitcoin/market_chart"
           f"?vs_currency=usd&days=30&interval=hourly"
           f"&x_cg_demo_api_key={GECKO_API_KEY}")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return pd.Series([p[1] for p in r.json()["prices"]])


def open_risk_usd(positions: list) -> float:
    """Rough cost basis of all open long positions."""
    total = 0.0
    for p in positions:
        contracts = abs(p.get("position", 0))
        cost_cents = p.get("market_exposure", 0) or p.get("total_cost", 0) or 0
        total += contracts * cost_cents / 100
    return total


# ---------------------------------------------------------------------------
# Core trading cycle
# ---------------------------------------------------------------------------

def run_cycle(config: dict, state: dict) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    state["last_run"]   = now
    state["enabled"]    = config["enabled"]
    state["dry_run"]    = config["dry_run"]
    state["errors"]     = []

    # 1. Compute signal
    try:
        prices     = fetch_price_history()
        indicators = compute_indicators(prices)
        signal     = generate_signal(indicators)

        state["current_price"]      = indicators["price"]
        state["current_indicators"] = indicators

        prev_label = state.get("last_signal")
        if signal["label"] != prev_label:
            log.info(f"Signal change: {prev_label} → {signal['label']}  "
                     f"(RSI={signal['rsi']:.1f}, score={signal['bull_score']:+d})")
            state["last_signal"]      = signal["label"]
            state["last_signal_time"] = now
        else:
            log.info(f"Signal unchanged: {signal['label']}  (RSI={signal['rsi']:.1f})")
    except Exception as e:
        log.error(f"Signal computation failed: {e}", exc_info=True)
        state["errors"].append(f"Signal: {e}")
        return state

    # 2. Refresh portfolio
    try:
        bal = kc.get_balance()
        state["balance_cents"] = bal.get("balance", 0)

        positions = kc.get_positions()
        state["active_positions"] = positions
    except Exception as e:
        log.warning(f"Portfolio refresh failed: {e}")
        state["errors"].append(f"Portfolio: {e}")

    # 3. Guard rails before placing any order
    if not config["enabled"]:
        log.info("Trading disabled — no order placed")
        return state

    if signal["direction"] == "neutral":
        log.info("HOLD — no order placed")
        return state

    if config["only_on_change"] and signal["label"] == prev_label:
        log.info("Signal unchanged and only_on_change=true — skipping")
        return state

    risk = open_risk_usd(state.get("active_positions", []))
    if risk >= config["max_open_risk_usd"]:
        msg = f"Open risk ${risk:.2f} >= limit ${config['max_open_risk_usd']} — skipping"
        log.warning(msg)
        state["errors"].append(msg)
        return state

    # 4. Select market and size order
    try:
        series   = config.get("series_ticker", "KXBTCD")
        markets  = kc.get_markets(series_ticker=series)
        if not markets:
            log.warning(f"No markets for {series}, falling back to BTC keyword search")
            markets = kc.search_markets("btc")

        market = select_market(markets, signal["direction"])
        if not market:
            state["errors"].append("No suitable market found")
            return state

        order_params = compute_order(signal, market, config["max_contracts"])
        if not order_params:
            log.info("No order computed (missing price?)")
            return state
    except Exception as e:
        log.error(f"Market/order selection failed: {e}", exc_info=True)
        state["errors"].append(f"Selection: {e}")
        return state

    # 5. Execute (or dry-run)
    order_record = {
        "timestamp":   now,
        "signal":      signal["label"],
        "ticker":      order_params["ticker"],
        "side":        order_params["side"],
        "count":       order_params["count"],
        "price_cents": order_params["price_cents"],
        "cost_usd":    order_params["cost_usd"],
        "rationale":   order_params["rationale"],
        "dry_run":     config["dry_run"],
        "result":      None,
    }

    if config["dry_run"]:
        log.info(f"[DRY RUN] Would place: {order_params}")
        order_record["result"] = "dry_run"
    else:
        try:
            result = kc.place_order(
                ticker      = order_params["ticker"],
                action      = order_params["action"],
                side        = order_params["side"],
                count       = order_params["count"],
                order_type  = order_params["order_type"],
                price_cents = order_params["price_cents"],
            )
            order_record["result"] = result
            log.info(f"Order placed: {result}")
        except Exception as e:
            log.error(f"Order placement failed: {e}", exc_info=True)
            order_record["result"] = f"ERROR: {e}"
            state["errors"].append(f"Order: {e}")

    state["recent_orders"] = ([order_record] + state.get("recent_orders", []))[:100]
    return state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Write default config if it doesn't exist
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        log.info(f"Created default config at {CONFIG_FILE}")

    log.info("=== BTC Trader daemon started ===")
    state = load_state()
    state["is_running"] = True
    save_state(state)

    try:
        while True:
            config = load_config()
            state  = load_state()
            state["is_running"] = True

            try:
                state = run_cycle(config, state)
            except Exception as e:
                log.error(f"Unexpected cycle error: {e}", exc_info=True)
                state["errors"] = state.get("errors", []) + [str(e)]

            save_state(state)
            interval = config.get("loop_interval_sec", 60)
            log.info(f"Next cycle in {interval}s")
            time.sleep(interval)

    except KeyboardInterrupt:
        log.info("Trader stopped by keyboard interrupt")
    finally:
        state = load_state()
        state["is_running"] = False
        save_state(state)
        log.info("=== Trader daemon stopped ===")


if __name__ == "__main__":
    main()
