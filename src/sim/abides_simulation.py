# src/sim/abides_simulation.py

from __future__ import annotations

from pathlib import Path
import pandas as pd

from abides_core import abides
from abides_markets.configs import rmsc04

from src.sim.extract_agent_features import extract_agent_features
from src.data.hub_price_loader import get_price_for_hub_date

SCALE = 10_000

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "simulated_trades"

def _debug_exchange_trades(end_state: dict, symbol: str = "ABM") -> None:
    agents = end_state.get("agents", [])
    ex = None
    for a in agents:
        if a.__class__.__name__ == "ExchangeAgent" or "Exchange" in a.__class__.__name__:
            ex = a
            break

    print("[DEBUG] exchange:", ex.__class__.__name__ if ex else None)
    if ex is None:
        return

    try:
        print("[DEBUG] exchange order_books keys:", list(ex.order_books.keys())[:10])
    except Exception as e:
        print("[DEBUG] cannot read exchange.order_books:", repr(e))
        return

    if symbol not in ex.order_books:
        print(f"[DEBUG] symbol {symbol} not in order_books")
        return

    ob = ex.order_books[symbol]
    print("[DEBUG] order book type:", type(ob))

    tape = getattr(ob, "tape", None)
    if tape is None:
        print("[DEBUG] order book has no .tape attribute")
        cand = [x for x in dir(ob) if any(k in x.lower() for k in ["tape", "trade", "trans", "fill"])]
        print("[DEBUG] candidate order book attrs:", cand[:50])
        return

    try:
        print("[DEBUG] tape len:", len(tape))
        if len(tape) > 0:
            print("[DEBUG] first tape item:", tape[0])
            print("[DEBUG] last tape item:", tape[-1])
    except Exception as e:
        print("[DEBUG] tape exists but cannot inspect:", repr(e))

def get_r_bar_for_date(date: str, hub: str = "TTF", prices_csv: str = "data/eu_hub_prices.csv") -> float:
    p = float(get_price_for_hub_date(hub=hub, date_str=date, csv_path=prices_csv))
    return p * SCALE


def run_rmsc04_simulation(date: str, hub: str = "TTF", prices_csv: str = "data/eu_hub_prices.csv"):
    try:
        config = rmsc04.build_config(
            seed=123,
            date=date,
            log_orders=False,
            hub=hub,
            prices_csv=prices_csv,
        )
    except TypeError:
        config = rmsc04.build_config(
            seed=123,
            date=date,
            log_orders=False,
        )

    end_state = abides.run(config)

    try:
        exch = end_state["agents"][0]  
        print("[DEBUG] exchange order_books keys:", list(exch.order_books.keys())[:10])
    except Exception:
        pass

    return end_state


def run_and_save_agent_features(
    date: str,
    hub: str = "TTF",
    prices_csv: str = "data/eu_hub_prices.csv",
) -> Path:
    hub = str(hub).strip().upper()

    end_state = run_rmsc04_simulation(date=date, hub=hub, prices_csv=prices_csv)
    _debug_exchange_trades(end_state)

    print("[DEBUG] end_state type:", type(end_state))
    if isinstance(end_state, dict):
        print("[DEBUG] end_state keys:", list(end_state.keys())[:20])
        if "agents" in end_state:
            print("[DEBUG] #agents:", len(end_state["agents"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fills_buy = fills_sell = fills_total = 0
    vol_buy = vol_sell = vol_total = 0.0
    last_trade = None

    try:
        ex = None
        for a in end_state.get("agents", []):
            if "Exchange" in a.__class__.__name__:
                ex = a
                break

        if ex is not None and hasattr(ex, "order_books") and "ABM" in ex.order_books:
            ob = ex.order_books["ABM"]

            buy_tx = getattr(ob, "buy_transactions", None) or []
            sell_tx = getattr(ob, "sell_transactions", None) or []

            fills_buy = int(len(buy_tx))
            fills_sell = int(len(sell_tx))
            fills_total = int(fills_buy + fills_sell)

            if hasattr(ob, "get_transacted_volume"):
                vb, vs = ob.get_transacted_volume()
                vol_buy = float(vb)
                vol_sell = float(vs)
                vol_total = float(vol_buy + vol_sell)

            last_trade = getattr(ob, "last_trade", None)

    except Exception as e:
        print("[WARN] could not extract exchange trade stats:", repr(e))

    df = extract_agent_features(end_state, date)

    df["date"] = date
    cols = ["date"] + [c for c in df.columns if c != "date"]
    df = df[cols]

    df.insert(0, "hub", hub)

    features_path = OUT_DIR / f"agent_features_{hub}_{date}.csv"
    df.to_csv(features_path, index=False)
    print(f"[OK] Saved {len(df)} rows -> {features_path}")

    r_bar = get_r_bar_for_date(date=date, hub=hub, prices_csv=prices_csv)

    summary = pd.DataFrame([{
        "date": date,
        "hub": hub,
        "prices_csv": prices_csv,
        "r_bar": float(r_bar),                 
        "r_bar_scaled": float(r_bar / SCALE),  

        "fills_buy": fills_buy,
        "fills_sell": fills_sell,
        "fills_total": fills_total,
        "vol_buy": vol_buy,
        "vol_sell": vol_sell,
        "vol_total": vol_total,
        "last_trade": last_trade,
    }])

    summary_path = OUT_DIR / f"day_summary_{hub}_{date}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Saved oracle summary -> {summary_path}")

    return features_path