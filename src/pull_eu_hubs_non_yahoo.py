"""
Download EU natural gas hub prices.

Policy:
1) Try Yahoo Finance via yfinance for each hub ticker.
2) Only if Yahoo fails/empty -> try EEX DataSource REST (getSpot/csv style).
3) Write one CSV per hub + a download report CSV you can cite in the thesis.

Usage:
  python -m src.pull_eu_hubs_non_yahoo --start 2020-11-01 --end 2026-02-26

Env vars for EEX (only needed if you want fallback to work):
  export EEX_USER="..."
  export EEX_PASS="..."

Notes:
- Yahoo tickers are not guaranteed to exist for all hubs.
- EEX endpoints require access; if you get 401, you need EEX to enable your account.
"""

from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import date, timedelta
from typing import Optional, List

import pandas as pd

# Optional imports (Yahoo + requests)
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import requests
except Exception:
    requests = None


# -----------------------
# Configuration
# -----------------------

OUT_DIR = Path("data/eu_hub_prices")
REPORT_PATH = OUT_DIR / "download_report.csv"

HUBS = {
    "TTF": {"yahoo": "TTF=F", "eex_marketarea": "ttf"},
    "NBP": {"yahoo": "NBP=F", "eex_marketarea": "nbp"},
    "THE": {"yahoo": "THE=F", "eex_marketarea": "the"},
    "PEG": {"yahoo": "PEG=F", "eex_marketarea": "peg"},
    "PSV": {"yahoo": "PSV=F", "eex_marketarea": "psv"},
}

EEX_BASE = "https://api1.datasource.eex-group.com/getSpot/csv"


# -----------------------
# Helpers
# -----------------------

def iso(d: date) -> str:
    return d.isoformat()

def daterange(start: date, end: date) -> List[date]:
    cur = start
    out: List[date] = []
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


@dataclass
class DownloadResult:
    hub: str
    source: str              # "yahoo" or "eex"
    ok: bool
    rows: int
    path: str
    details: str


# -----------------------
# Yahoo (yfinance)
# -----------------------

def download_from_yahoo(hub: str, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    if yf is None:
        raise RuntimeError("yfinance not installed. pip install yfinance")

    # IMPORTANT FIX: ensure ticker is a string (not a tuple)
    ticker = str(ticker)

    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df is None or df.empty:
        return None

    # If Yahoo returns MultiIndex columns, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    # Normalize columns
    rename = {}
    for c in df.columns:
        if c == "Date":
            rename[c] = "date"
        elif c.lower() in ("open", "high", "low", "close", "adj close", "volume"):
            rename[c] = c.lower().replace(" ", "_")
    df = df.rename(columns=rename)

    keep = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df["hub"] = hub
    df["ticker"] = ticker
    df["source"] = "Yahoo Finance (via yfinance)"
    return df


# -----------------------
# EEX fallback (only for hubs that fail Yahoo)
# -----------------------

def eex_auth():
    user = os.getenv("EEX_USER")
    pw = os.getenv("EEX_PASS")
    if not user or not pw:
        return None
    return (user, pw)

def fetch_one_eex_spot(trade_date: str, marketarea: str, commodity: str = "NATGAS", return_type: str = "results") -> pd.DataFrame:
    if requests is None:
        raise RuntimeError("requests not installed. pip install requests")

    auth = eex_auth()
    if auth is None:
        raise RuntimeError("Missing EEX_USER/EEX_PASS env vars for EEX fallback.")

    params = {
        "returnType": return_type,
        "commodity": commodity,
        "tradeDate": trade_date,
        "marketarea": marketarea,
    }

    r = requests.get(EEX_BASE, params=params, auth=auth, timeout=60)
    if r.status_code == 401:
        raise PermissionError("EEX returned 401 Unauthorized. Your account likely has no API access enabled yet.")
    r.raise_for_status()

    text = r.text.strip()
    if not text:
        return pd.DataFrame()

    from io import StringIO
    return pd.read_csv(StringIO(text))

def download_from_eex(hub: str, marketarea: str, start: str, end: str) -> Optional[pd.DataFrame]:
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)

    rows = []
    for d in daterange(start_d, end_d):
        ds = iso(d)
        try:
            day = fetch_one_eex_spot(ds, marketarea=marketarea)
            if day is None or day.empty:
                continue
            day["tradeDate_requested"] = ds
            rows.append(day)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if isinstance(e, PermissionError):
                raise
            time.sleep(0.2)
            continue

    if not rows:
        return None

    df = pd.concat(rows, ignore_index=True)
    df["hub"] = hub
    df["marketarea"] = marketarea
    df["source"] = "EEX Group DataSource (getSpot/csv)"
    return df


# -----------------------
# Main
# -----------------------

def save_df(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

def append_report(results: List[DownloadResult], report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    recs = [r.__dict__ for r in results]
    rep = pd.DataFrame(recs)
    if report_path.exists():
        old = pd.read_csv(report_path)
        rep = pd.concat([old, rep], ignore_index=True)
    rep.to_csv(report_path, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    report_path = out_dir / "download_report.csv"

    results: List[DownloadResult] = []

    for hub, cfg in HUBS.items():
        yahoo_ticker = cfg.get("yahoo")
        eex_marketarea = cfg.get("eex_marketarea")

        used_fallback = False

        # 1) Yahoo
        if yahoo_ticker:
            try:
                dfy = download_from_yahoo(hub, yahoo_ticker, args.start, args.end)
                if dfy is not None and not dfy.empty:
                    out_path = out_dir / f"{hub}_yahoo.csv"
                    save_df(dfy, out_path)
                    results.append(DownloadResult(
                        hub=hub, source="yahoo", ok=True, rows=len(dfy),
                        path=str(out_path), details=f"ticker={yahoo_ticker}"
                    ))
                    print(f"[OK] {hub} Yahoo rows={len(dfy)} -> {out_path}")
                    continue
                else:
                    used_fallback = True
                    print(f"[MISS] {hub} Yahoo empty -> will try EEX fallback")
            except Exception as e:
                used_fallback = True
                print(f"[MISS] {hub} Yahoo failed ({repr(e)}) -> will try EEX fallback")

        # 2) EEX fallback ONLY if Yahoo failed
        if used_fallback:
            try:
                dfe = download_from_eex(hub, eex_marketarea, args.start, args.end)
                if dfe is not None and not dfe.empty:
                    out_path = out_dir / f"{hub}_eex.csv"
                    save_df(dfe, out_path)
                    results.append(DownloadResult(
                        hub=hub, source="eex", ok=True, rows=len(dfe),
                        path=str(out_path), details=f"marketarea={eex_marketarea}"
                    ))
                    print(f"[OK] {hub} EEX rows={len(dfe)} -> {out_path}")
                else:
                    results.append(DownloadResult(
                        hub=hub, source="eex", ok=False, rows=0, path="",
                        details="EEX returned no rows (or all days empty)."
                    ))
                    print(f"[FAIL] {hub} EEX no data.")
            except PermissionError as e:
                results.append(DownloadResult(
                    hub=hub, source="eex", ok=False, rows=0, path="",
                    details=str(e)
                ))
                print(f"[FAIL] {hub} EEX unauthorized: {e}")
            except Exception as e:
                results.append(DownloadResult(
                    hub=hub, source="eex", ok=False, rows=0, path="",
                    details=repr(e)
                ))
                print(f"[FAIL] {hub} EEX error: {repr(e)}")
        else:
            results.append(DownloadResult(
                hub=hub, source="yahoo", ok=False, rows=0, path="",
                details="No Yahoo ticker configured."
            ))
            print(f"[SKIP] {hub} no Yahoo ticker configured.")

    # IMPORTANT FIX: pass report_path
    append_report(results, report_path)
    print(f"[OK] wrote report -> {report_path}")


if __name__ == "__main__":
    main()