# src/extract_agent_features.py
from typing import List, Any, Optional, Tuple
import pandas as pd
import ast

AGENT_TYPES = {
    "NoiseAgent",
    "ValueAgent",
    "MomentumAgent",
    "AdaptivePOVMarketMakerAgent",
}

DEFAULT_SYMBOL = "ABM"


def _safe_call(obj: Any, method: str, *args):
    if not hasattr(obj, method):
        return None
    try:
        return getattr(obj, method)(*args)
    except Exception:
        return None


def _parse_triplet(x) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Parses values like "(99995, 99996, 99996)" or "(None, None, None)" into floats/Nones.
    Handles tuple objects or stringified tuples.
    """
    if x is None:
        return None, None, None

    if isinstance(x, tuple) and len(x) == 3:
        b, a, m = x
        return (None if b is None else float(b),
                None if a is None else float(a),
                None if m is None else float(m))

    # string case
    if isinstance(x, str):
        try:
            t = ast.literal_eval(x)
            if isinstance(t, tuple) and len(t) == 3:
                b, a, m = t
                return (None if b is None else float(b),
                        None if a is None else float(a),
                        None if m is None else float(m))
        except Exception:
            pass

    return None, None, None


def _get_executed_orders(agent: Any, symbol: str):
    """
    Try to pull executed orders for this symbol from common ABIDES agent storage.
    We keep this defensive because different agent types store differently.
    """
    eo = getattr(agent, "executed_orders", None)
    if eo is None:
        return None

    # Often a dict: symbol -> list
    if isinstance(eo, dict):
        return eo.get(symbol)

    # Sometimes just a list of fills
    if isinstance(eo, list):
        return eo

    return None


def _trade_stats_from_fills(fills):
    """
    Given a list of fill objects/dicts, compute:
      - n_fills
      - total_qty
      - vwap
    If we can't read price/qty, returns None.
    """
    if not fills:
        return 0, 0.0, None

    total_qty = 0.0
    total_px_qty = 0.0
    n = 0

    for f in fills:
        # try common patterns: dict or object with attributes
        px = None
        qty = None

        if isinstance(f, dict):
            px = f.get("price")
            qty = f.get("quantity") or f.get("qty") or f.get("size")
        else:
            px = getattr(f, "price", None)
            qty = getattr(f, "quantity", None) or getattr(f, "qty", None) or getattr(f, "size", None)

        if px is None or qty is None:
            continue

        n += 1
        total_qty += float(qty)
        total_px_qty += float(px) * float(qty)

    if n == 0 or total_qty == 0:
        return n, total_qty, None

    vwap = total_px_qty / total_qty
    return n, total_qty, vwap


def extract_agent_features(end_state: dict, date_str: str, symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    agents: List[Any] = end_state["agents"]
    rows = []

    for agent in agents:
        agent_type = agent.__class__.__name__
        if agent_type not in AGENT_TYPES:
            continue

        known_triplet = _safe_call(agent, "get_known_bid_ask_midpoint", symbol)
        best_bid, best_ask, mid = _parse_triplet(known_triplet)
        spread = None if (best_bid is None or best_ask is None) else (best_ask - best_bid)

        # First try ABIDES helper methods
        avg_tx = _safe_call(agent, "get_average_transaction_price", symbol)
        vol = _safe_call(agent, "get_transacted_volume", symbol)

        # Fallback: compute from executed orders if methods are None
        fills = _get_executed_orders(agent, symbol)
        n_fills, total_qty, vwap = _trade_stats_from_fills(fills)

        if avg_tx is None:
            avg_tx = vwap
        if vol is None:
            vol = total_qty

        rows.append({
            "date": date_str,
            "agent_id": getattr(agent, "id", None),
            "agent_type": agent_type,

            # order book / quote features
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "spread": spread,

            # trade activity features
            "n_fills": n_fills,
            "transacted_volume": vol,
            "avg_tx_price": avg_tx,
        })

    return pd.DataFrame(rows)