"""
engine/data.py — Live market data fetching via yfinance.
Fetches 5m candles for the last 7 days and resamples to 1h.
"""

from __future__ import annotations
import pandas as pd
from datetime import datetime, timedelta
from typing import List
import time
import io
import logging
from contextlib import redirect_stderr


def fetch_live_candles(tickers: List[str]) -> dict[str, pd.DataFrame]:
    """
    Download the last 7 days of 5-minute candles for each ticker.
    Returns {"5m": df_5m, "1h": df_1h}.
    Both DataFrames have columns: open, high, low, close, volume, ticker
    and a timezone-aware DatetimeIndex (Asia/Kolkata).
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance not installed — pip install yfinance")
    # Avoid noisy provider logs in server output on transient Yahoo failures.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    try:
        import pytz
        IST = pytz.timezone("Asia/Kolkata")
    except Exception:
        IST = None

    now = datetime.utcnow()
    start_str = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    end_str   = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    rows_5m, rows_1h, failed = [], [], []

    for ticker in tickers:
        try:
            raw = pd.DataFrame()
            # Retry with small backoff for transient provider/network failures.
            for attempt in range(3):
                try:
                    with redirect_stderr(io.StringIO()):
                        raw = yf.download(
                            ticker,
                            start=start_str,
                            end=end_str,
                            interval="5m",
                            progress=False,
                            auto_adjust=True,
                            threads=False,
                        )
                    if not raw.empty:
                        break
                except Exception:
                    pass
                time.sleep(0.6 * (attempt + 1))
            if raw.empty:
                failed.append(ticker)
                continue

            # Flatten multi-level columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]

            raw.index = pd.DatetimeIndex(raw.index)
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
            else:
                raw.index = raw.index.tz_convert("Asia/Kolkata")

            raw["ticker"] = ticker
            rows_5m.append(raw[["open", "high", "low", "close", "volume", "ticker"]])

            # Resample to 1h
            ohlc_1h = raw.resample("1h", closed="right", label="right").agg(
                open=("open", "first"), high=("high", "max"),
                low=("low", "min"),   close=("close", "last"),
                volume=("volume", "sum"),
            ).dropna(subset=["close"])
            ohlc_1h["ticker"] = ticker
            rows_1h.append(ohlc_1h)

        except Exception:
            failed.append(ticker)
            continue

    df5  = pd.concat(rows_5m).sort_index() if rows_5m else pd.DataFrame()
    df1h = pd.concat(rows_1h).sort_index() if rows_1h else pd.DataFrame()

    return {"5m": df5, "1h": df1h, "failed": failed}


def get_latest_prices(tickers: List[str]) -> dict[str, float]:
    """
    Returns the latest close price for each ticker as a dict.
    Used by the 2-minute price refresh endpoint.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {}
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    prices = {}
    for ticker in tickers:
        try:
            hist = pd.DataFrame()
            for attempt in range(3):
                try:
                    t = yf.Ticker(ticker)
                    with redirect_stderr(io.StringIO()):
                        hist = t.history(period="1d", interval="5m")
                    if not hist.empty:
                        break
                except Exception:
                    pass
                time.sleep(0.4 * (attempt + 1))
            if not hist.empty:
                prices[ticker] = float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return prices
