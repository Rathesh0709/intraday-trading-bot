#!/usr/bin/env python3
"""
Refresh optional reference CSVs under BOT_DATA_DIR (or monorepo root by default).

  - nifty50_stocks_daily.csv  — daily OHLCV per NIFTY 50 ticker (yfinance)
  - macro_sentiment.csv       — macro proxies aligned with merge_macro_features()

Run on your laptop or in CI before deploy, or on the server on a schedule:

  cd trading-app-backend
  set BOT_DATA_DIR=F:\\data\\bot-ref   # Windows
  export BOT_DATA_DIR=/data/bot-ref    # Linux
  python scripts/refresh_bot_reference_data.py

Requires: yfinance, pandas (already in requirements.txt).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python scripts/foo.py` from trading-app-backend
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from engine.config import (  # noqa: E402
    MACRO_SENTIMENT_CSV,
    NIFTY_DAILY_CSV,
    NIFTY_50_TICKERS,
    get_bot_data_dir,
)


def _ist_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        idx = idx.tz_convert("Asia/Kolkata")
    df = df.copy()
    df.index = idx
    return df


def refresh_daily_panel(years: int = 5) -> Path:
    import yfinance as yf

    out_dir = get_bot_data_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / NIFTY_DAILY_CSV
    rows: list[pd.DataFrame] = []
    period = f"{max(1, years)}y"
    for t in NIFTY_50_TICKERS:
        raw = yf.download(t, period=period, interval="1d", progress=False, auto_adjust=True, threads=False)
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [str(c[0]).lower() for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]
        raw = _ist_index(raw)
        raw["ticker"] = t
        rows.append(raw)
    if not rows:
        raise RuntimeError("No daily data downloaded — check tickers / network.")
    full = pd.concat(rows, axis=0).sort_index()
    full.index.name = "datetime"
    full.to_csv(dest)
    print(f"Wrote {len(full):,} rows -> {dest}")
    return dest


def refresh_macro_sentiment(years: int = 5) -> Path:
    import yfinance as yf

    out_dir = get_bot_data_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / MACRO_SENTIMENT_CSV
    period = f"{max(1, years)}y"

    series_map = {
        "^GSPC": "sp500",
        "INR=X": "usdinr",
        "GC=F": "gold",
        "CL=F": "crude",
        "^VIX": "us_vix",
        "^NSEBANK": "bank",
    }

    combined: pd.DataFrame | None = None
    for yahoo_sym, prefix in series_map.items():
        df = yf.download(
            yahoo_sym,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if df.empty:
            print(f"[warn] No rows for {yahoo_sym}, skipping.")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]) for c in df.columns]
        df = _ist_index(df)
        df.columns = [f"{prefix}_{str(c).lower()}" for c in df.columns]
        combined = df if combined is None else combined.join(df, how="outer")
    if combined is None or combined.empty:
        raise RuntimeError("Macro download failed — check network / Yahoo.")

    combined = combined.sort_index().ffill().dropna(how="all")
    # Match legacy file: first column name Date for readability (index written as first col)
    combined.index.name = "Date"
    combined.to_csv(dest)
    print(f"Wrote macro panel {combined.shape} -> {dest}")
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description="Refresh BOT_DATA_DIR reference CSVs.")
    p.add_argument("--daily-only", action="store_true")
    p.add_argument("--macro-only", action="store_true")
    p.add_argument("--years", type=int, default=5)
    args = p.parse_args()

    print(f"BOT_DATA_DIR -> {get_bot_data_dir()}")
    if not args.macro_only:
        refresh_daily_panel(years=args.years)
    if not args.daily_only:
        refresh_macro_sentiment(years=args.years)
    print("Done.")


if __name__ == "__main__":
    main()
