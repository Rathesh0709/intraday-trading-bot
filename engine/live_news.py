"""
engine/live_news.py — Live news sentiment via BSE announcements API.

This is the inference-time equivalent of `stock_trading_bot.py` live sentiment:
- Fetch recent BSE corporate announcements for a ticker
- Score the headlines into a sentiment score in [-1.0, +1.0]

Used by backend signal ranking: 70% model + 30% news (with SELL inversion).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Same mapping as `stock_trading_bot.py` (NIFTY 50 only).
BSE_CODE_MAP: dict[str, int] = {
    "RELIANCE": 500325,
    "TCS": 532540,
    "HDFCBANK": 500180,
    "ICICIBANK": 532174,
    "INFY": 500209,
    "ITC": 500875,
    "LT": 500510,
    "SBIN": 500112,
    "BHARTIARTL": 532454,
    "BAJFINANCE": 500034,
    "KOTAKBANK": 500247,
    "AXISBANK": 532215,
    "ASIANPAINT": 500820,
    "HINDUNILVR": 500696,
    "TITAN": 500114,
    "MARUTI": 532500,
    "SUNPHARMA": 524715,
    "TATASTEEL": 500470,
    "M&M": 500520,
    "ULTRACEMCO": 532538,
    "POWERGRID": 532898,
    "NTPC": 532555,
    "WIPRO": 507685,
    "NESTLEIND": 500790,
    "HCLTECH": 532281,
    "BAJAJFINSV": 532978,
    "ONGC": 500312,
    "TECHM": 532755,
    "HINDALCO": 500440,
    "ADANIENT": 512599,
    "ADANIPORTS": 532921,
    "GRASIM": 500300,
    "JSWSTEEL": 500228,
    "CIPLA": 500087,
    "DRREDDY": 500124,
    "TATACONSUM": 500800,
    "EICHERMOT": 505200,
    "APOLLOHOSP": 508869,
    "DIVISLAB": 532488,
    "BRITANNIA": 500825,
    "COALINDIA": 533278,
    "HEROMOTOCO": 500182,
    "BPCL": 500547,
    "BAJAJ-AUTO": 532977,
    "HDFCLIFE": 540777,
    "PIDILITIND": 500331,
    "INDUSINDBK": 532187,
    "SBILIFE": 540719,
    "LTIM": 540005,
    "BEL": 500049,
}


def score_announcement(subject: str, description: str = "") -> float:
    text = (subject + " " + description).lower()
    score = 0.0
    strong_positive = {
        "dividend": 0.6,
        "buyback": 0.8,
        "bonus": 0.7,
        "stock split": 0.5,
        "order received": 0.7,
        "contract awarded": 0.7,
        "acquisition": 0.5,
        "record profit": 0.9,
        "highest ever": 0.8,
        "promoter buying": 0.6,
        "share repurchase": 0.7,
        "capacity expansion": 0.5,
    }
    moderate_positive = {
        "profit": 0.4,
        "revenue growth": 0.4,
        "positive": 0.2,
        "increase": 0.2,
        "approval": 0.3,
        "launch": 0.2,
        "partnership": 0.3,
        "agreement": 0.2,
    }
    strong_negative = {
        "fraud": 0.9,
        "default": 0.9,
        "sebi probe": 0.8,
        "ed raid": 0.8,
        "cbi probe": 0.8,
        "penalty": 0.6,
        "fine": 0.5,
        "loss": 0.7,
        "promoter selling": 0.6,
        "insolvency": 0.9,
        "bankruptcy": 0.9,
        "resignation": 0.4,
        "downsizing": 0.5,
    }
    moderate_negative = {
        "decline": 0.3,
        "lower": 0.2,
        "miss": 0.4,
        "weak": 0.3,
        "delay": 0.2,
        "dispute": 0.3,
        "lawsuit": 0.4,
    }
    for kw, w in strong_positive.items():
        if kw in text:
            score += w
    for kw, w in moderate_positive.items():
        if kw in text:
            score += w
    for kw, w in strong_negative.items():
        if kw in text:
            score -= w
    for kw, w in moderate_negative.items():
        if kw in text:
            score -= w
    return float(max(-1.0, min(1.0, score)))


def _fetch_bse_announcements(symbol: str, lookback_hours: int = 48) -> pd.DataFrame:
    scrip_code = BSE_CODE_MAP.get(symbol)
    if not scrip_code:
        return pd.DataFrame()

    now = datetime.utcnow()
    from_dt = (now - timedelta(hours=lookback_hours)).strftime("%Y%m%d")
    to_dt = (now + timedelta(days=1)).strftime("%Y%m%d")

    url = (
        "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w?"
        f"pageno=1&strCat=-1&strPrevDate={from_dt}&strScrip={scrip_code}"
        f"&strSearch=P&strToDate={to_dt}&strType=C&subcategory=-1"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.bseindia.com",
        "Referer": "https://www.bseindia.com/",
        "Connection": "keep-alive",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        items = data.get("Table", []) or []
        if not items:
            return pd.DataFrame()
        rows = []
        for item in items:
            dt_str = (
                item.get("DT_TM", "")
                or item.get("News_submission_dt", "")
                or item.get("NEWS_DT", "")
                or item.get("DissemDT", "")
            )
            if not dt_str:
                continue
            dt = pd.to_datetime(str(dt_str), errors="coerce")
            if pd.isna(dt):
                continue
            subject = (item.get("NEWSSUB", "") or item.get("HEADLINE", "") or "")
            category = (item.get("CATEGORYNAME", "") or item.get("ANNOUNCEMENT_TYPE", "") or "")
            if not subject:
                continue
            rows.append(
                {
                    "datetime": dt,
                    "symbol": symbol,
                    "subject": str(subject)[:200],
                    "description": str(category)[:200],
                }
            )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_live_combined_sentiment(ticker: str) -> float:
    """
    Return live sentiment score in [-1, +1] based on recent BSE announcements.
    Mirrors `stock_trading_bot.get_live_combined_sentiment` (BSE-only).
    """
    symbol = ticker.replace(".NS", "")
    ann = _fetch_bse_announcements(symbol, lookback_hours=48)
    if ann.empty:
        return 0.0

    # Time-decay weighted scoring over 48h (same intent as stock bot)
    now = pd.Timestamp.utcnow()
    score = 0.0
    events = 0
    for _, row in ann.iterrows():
        try:
            dt = pd.to_datetime(row["datetime"], errors="coerce")
            if pd.isna(dt):
                continue
            hours_ago = max(0.0, (now - dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT") if dt.tzinfo is None else now - dt).total_seconds() / 3600)
            weight = max(0.1, 1.0 - hours_ago / 48.0)
            s = score_announcement(str(row["subject"]), str(row.get("description", "")))
            score += s * weight
            events += 1
        except Exception:
            continue
    if events == 0:
        return 0.0
    return float(max(-1.0, min(1.0, score)))

