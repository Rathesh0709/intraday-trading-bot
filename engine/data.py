"""
engine/data.py — Live market data fetching via yfinance.
Fetches 5m candles for the last 7 days and resamples to 1h.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
import time
import io
import logging
from contextlib import redirect_stderr
import requests

from engine.features_live import enrich_intraday_panel, stub_missing_master_columns
from engine.bot_training_mirror import apply_stock_bot_1h_panel

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


def _process_ticker_raw(ticker: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    raw = enrich_intraday_panel(raw)

    # Resample to 1h
    ohlc_1h = raw.resample("1h", closed="right", label="right").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    ohlc_1h["ticker"] = ticker
    ohlc_1h = apply_stock_bot_1h_panel(ohlc_1h)
    ohlc_1h = stub_missing_master_columns(ohlc_1h)
    ohlc_1h = ohlc_1h.replace([np.inf, -np.inf], np.nan)
    return raw, ohlc_1h


def _download_from_chart_api(
    ticker: str,
    interval: str = "5m",
    range_: str = "7d",
    timeout_s: int = 20,
) -> pd.DataFrame:
    """
    Alternate provider path (non-yfinance) using Yahoo chart endpoint directly.
    Helps when yfinance internals fail/parsing breaks.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "interval": interval,
        "range": range_,
        "events": "div,splits",
        "includePrePost": "false",
    }
    resp = requests.get(url, params=params, headers=_HTTP_HEADERS, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    chart = (payload.get("chart") or {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"chart_api_error: {error}")
    result = (chart.get("result") or [None])[0]
    if not result:
        return pd.DataFrame()

    ts = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [None])[0] or {}
    if not ts:
        return pd.DataFrame()

    df = pd.DataFrame({
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", []),
    })
    if df.empty:
        return pd.DataFrame()

    # Keep only complete rows with close/open/high/low
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return pd.DataFrame()

    # Timestamp alignment
    ts = pd.to_datetime(pd.Series(ts), unit="s", utc=True)
    df = df.iloc[: len(ts)].copy()
    df.index = pd.DatetimeIndex(ts.iloc[: len(df)])
    return df


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
    pending = list(dict.fromkeys(tickers))

    # First try a bulk download to reduce provider throttling.
    if pending:
        try:
            with redirect_stderr(io.StringIO()):
                bulk = yf.download(
                    pending,
                    start=start_str,
                    end=end_str,
                    interval="5m",
                    progress=False,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=False,
                )
            if not bulk.empty and isinstance(bulk.columns, pd.MultiIndex):
                succeeded = set()
                for ticker in pending:
                    if ticker not in bulk.columns.get_level_values(0):
                        continue
                    part = bulk[ticker].copy()
                    if part.empty:
                        continue
                    try:
                        five, oneh = _process_ticker_raw(ticker, part)
                        rows_5m.append(five)
                        rows_1h.append(oneh)
                        succeeded.add(ticker)
                    except Exception:
                        continue
                pending = [t for t in pending if t not in succeeded]
        except Exception:
            # Fall back to per-ticker mode below.
            pass

    for ticker in pending:
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

            # Alternate data source path (chart API direct) if yfinance failed.
            if raw.empty:
                for attempt in range(2):
                    try:
                        raw = _download_from_chart_api(ticker, interval="5m", range_="7d")
                        if not raw.empty:
                            break
                    except Exception:
                        pass
                    time.sleep(0.8 * (attempt + 1))

            if raw.empty:
                failed.append(ticker)
                continue

            five, oneh = _process_ticker_raw(ticker, raw)
            rows_5m.append(five)
            rows_1h.append(oneh)

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

    prices: dict[str, float] = {}
    uniq = list(dict.fromkeys(tickers))
    pending = uniq[:]

    # Fast bulk attempt first.
    if pending:
        try:
            with redirect_stderr(io.StringIO()):
                bulk = yf.download(
                    pending,
                    period="1d",
                    interval="5m",
                    progress=False,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=False,
                )
            if not bulk.empty and isinstance(bulk.columns, pd.MultiIndex):
                succeeded = set()
                for ticker in pending:
                    if ticker not in bulk.columns.get_level_values(0):
                        continue
                    part = bulk[ticker]
                    if "Close" in part.columns and not part["Close"].dropna().empty:
                        prices[ticker] = float(part["Close"].dropna().iloc[-1])
                        succeeded.add(ticker)
                pending = [t for t in pending if t not in succeeded]
        except Exception:
            pass

    for ticker in pending:
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
            if hist.empty:
                for attempt in range(2):
                    try:
                        hist = _download_from_chart_api(ticker, interval="5m", range_="1d")
                        if not hist.empty:
                            break
                    except Exception:
                        pass
                    time.sleep(0.6 * (attempt + 1))
            if not hist.empty and "Close" in hist.columns:
                close = hist["Close"].dropna()
                if not close.empty:
                    prices[ticker] = float(close.iloc[-1])
            elif not hist.empty and "close" in hist.columns:
                close = hist["close"].dropna()
                if not close.empty:
                    prices[ticker] = float(close.iloc[-1])
        except Exception:
            pass
    return prices
