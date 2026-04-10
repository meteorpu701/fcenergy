# src/run_abides_days.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import traceback

from src.sim.abides_simulation import run_and_save_agent_features


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "simulated_trades"

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run ABIDES day-by-day for one or more EU gas hubs using a hub price CSV."
    )

    ap.add_argument(
        "--hub",
        default=None,
        help='Single hub code in your price CSV (e.g. "TTF", "NBP", "FIN"). If set, overrides --hubs.',
    )

    ap.add_argument(
        "--hubs",
        default="TTF",
        help='Comma-separated hubs to simulate (e.g. "TTF,NBP,FIN,LTU"). Default: TTF',
    )

    ap.add_argument(
        "--prices",
        default="data/eu_hub_prices.csv",
        help="CSV containing columns at least: date, hub, price",
    )

    ap.add_argument("--start", default=None, help="Optional start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD (inclusive)")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of days per hub")

    ap.add_argument(
        "--outdir",
        default=str(DEFAULT_OUT_DIR),
        help="Where ABIDES outputs are saved (used for skip-check). Default: data/simulated_trades",
    )

    ap.add_argument(
        "--force",
        action="store_true",
        help="If set, do NOT skip existing days; re-run everything.",
    )

    return ap.parse_args()


def _norm_hub(h: str) -> str:
    return str(h).strip().upper()


def _parse_hubs(args: argparse.Namespace) -> list[str]:
    if args.hub:
        return [_norm_hub(args.hub)]
    hubs = [_norm_hub(x) for x in str(args.hubs).split(",") if x.strip()]
    if not hubs:
        raise ValueError("No hubs provided. Use --hub HUB or --hubs HUB1,HUB2,...")
    return hubs


def _load_dates(prices_path: str, hub: str, start: str | None, end: str | None) -> list[str]:
    df = pd.read_csv(prices_path, parse_dates=["date"])

    required = {"date", "hub", "price"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{prices_path} missing columns: {sorted(missing)} (need {sorted(required)})")

    df["hub"] = df["hub"].astype(str).str.strip().str.upper()

    sub = df[df["hub"] == _norm_hub(hub)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for hub={hub} in {prices_path}")

    sub = sub.sort_values("date")

    if start is not None:
        sub = sub[sub["date"] >= pd.to_datetime(start)]
    if end is not None:
        sub = sub[sub["date"] <= pd.to_datetime(end)]

    sub["price"] = pd.to_numeric(sub["price"], errors="coerce")
    sub = sub.dropna(subset=["price"])

    dates = [d.strftime("%Y-%m-%d") for d in sub["date"].tolist()]
    return dates


def _already_done(out_dir: Path, hub: str, date_str: str) -> bool:
    expected = out_dir / f"agent_features_{hub}_{date_str}.csv"
    return expected.exists()


def main() -> None:
    args = _parse_args()
    hubs = _parse_hubs(args)
    out_dir = Path(args.outdir)

    print(f"[INFO] prices={args.prices} outdir={out_dir} hubs={hubs}")
    if args.start or args.end:
        print(f"[INFO] date_filter start={args.start} end={args.end}")
    if args.limit is not None:
        print(f"[INFO] limit_per_hub={args.limit}")
    if args.force:
        print("[INFO] force=True (will not skip existing days)")

    totals = {"planned": 0, "ran": 0, "skipped": 0, "failed": 0}

    for hub in hubs:
        try:
            dates = _load_dates(args.prices, hub, args.start, args.end)
        except Exception as e:
            print(f"[FAIL] hub={hub} could not load dates: {repr(e)}")
            totals["failed"] += 1
            continue

        if args.limit is not None:
            dates = dates[: args.limit]

        totals["planned"] += len(dates)
        print(f"\n[HUB] {hub} days={len(dates)}")

        for d in dates:
            if (not args.force) and _already_done(out_dir, hub, d):
                print(f"[SKIP] hub={hub} date={d} (already exists)")
                totals["skipped"] += 1
                continue

            print(f"[RUN] hub={hub} date={d}")
            try:
                run_and_save_agent_features(date=d, hub=hub, prices_csv=args.prices)
                totals["ran"] += 1
            except KeyboardInterrupt:
                print("\n[STOP] KeyboardInterrupt")
                print(f"[SUMMARY] {totals}")
                raise
            except Exception as e:
                print(f"[FAIL] hub={hub} date={d}: {repr(e)}")
                totals["failed"] += 1
                traceback.print_exc()
                continue

    print(f"\n[SUMMARY] {totals}")


if __name__ == "__main__":
    main()