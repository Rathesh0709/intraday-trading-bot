"""
Inference-only feature pipeline (logic aligned with `stock_trading_bot.py`).

Used by the API so production does **not** import the full training script (xgboost stack
is not required here). Optional CSVs are read from **BOT_DATA_DIR** — see `get_bot_data_dir()`
in `engine.config`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import ta

from engine.config import (
    ANNOUNCEMENTS_CSV,
    MACRO_SENTIMENT_CSV,
    NIFTY_DAILY_CSV,
    get_bot_data_dir,
)

logger = logging.getLogger(__name__)


def load_daily_panel_data() -> pd.DataFrame:
    path = get_bot_data_dir() / NIFTY_DAILY_CSV
    if not path.is_file():
        logger.warning(
            "%s not found at %s — pivot / daily-linked columns use zeros.",
            NIFTY_DAILY_CSV,
            path,
        )
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower().strip() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    df.sort_index(inplace=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _calc(group):
        g = group.copy()
        g["sma_9"] = g["close"].rolling(9).mean()
        g["sma_21"] = g["close"].rolling(21).mean()
        g["sma_50"] = g["close"].rolling(50).mean()
        g["sma_200"] = g["close"].rolling(200).mean()
        g["ema_9"] = g["close"].ewm(span=9, adjust=False).mean()
        g["ema_21"] = g["close"].ewm(span=21, adjust=False).mean()
        g["ema_50"] = g["close"].ewm(span=50, adjust=False).mean()
        g["rsi"] = ta.momentum.RSIIndicator(g["close"], window=14).rsi()
        macd = ta.trend.MACD(g["close"])
        g["macd"] = macd.macd()
        g["macd_signal"] = macd.macd_signal()
        g["macd_histogram"] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(g["close"], window=20)
        g["bb_upper"] = bb.bollinger_hband()
        g["bb_lower"] = bb.bollinger_lband()
        g["bb_middle"] = bb.bollinger_mavg()
        g["bb_width"] = (g["bb_upper"] - g["bb_lower"]) / (g["bb_middle"] + 1e-9)
        g["bb_pct"] = (g["close"] - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"] + 1e-9)
        g["atr"] = ta.volatility.AverageTrueRange(
            g["high"], g["low"], g["close"], window=14
        ).average_true_range()
        stoch = ta.momentum.StochasticOscillator(g["high"], g["low"], g["close"])
        g["stoch_k"] = stoch.stoch()
        g["stoch_d"] = stoch.stoch_signal()
        adx = ta.trend.ADXIndicator(g["high"], g["low"], g["close"], window=14)
        g["adx"] = adx.adx()
        g["plus_di"] = adx.adx_pos()
        g["minus_di"] = adx.adx_neg()
        date_grp = g.index.normalize()
        tp = (g["high"] + g["low"] + g["close"]) / 3
        g["vwap"] = (tp * g["volume"]).groupby(date_grp).cumsum() / (
            g["volume"].groupby(date_grp).cumsum().clip(lower=1e-9)
        )
        g["dist_from_vwap"] = (g["close"] - g["vwap"]) / (g["vwap"] + 1e-9)
        g["price_above_vwap"] = (g["close"] > g["vwap"]).astype(int)
        g["obv"] = ta.volume.OnBalanceVolumeIndicator(
            g["close"], g["volume"]
        ).on_balance_volume()
        g["volume_sma_20"] = g["volume"].rolling(20).mean()
        g["volume_ratio"] = g["volume"] / (g["volume_sma_20"] + 1e-9)
        g["returns"] = g["close"].pct_change()
        g["log_returns"] = np.log(g["close"] / g["close"].shift(1))
        g["high_low_range"] = (g["high"] - g["low"]) / (g["close"] + 1e-9)
        g["close_open_range"] = (g["close"] - g["open"]) / (g["open"] + 1e-9)
        for w in [5, 10, 20]:
            g[f"rolling_mean_{w}"] = g["close"].rolling(w).mean()
            g[f"rolling_std_{w}"] = g["close"].rolling(w).std()
        g["zscore_20"] = (g["close"] - g["rolling_mean_20"]) / (g["rolling_std_20"] + 1e-9)
        g["ema_9_21_cross"] = (g["ema_9"] > g["ema_21"]).astype(int)
        g["sma_50_200_cross"] = (g["sma_50"] > g["sma_200"]).astype(int)
        hl2 = (g["high"] + g["low"]) / 2
        g["supertrend_upper"] = hl2 + 3.0 * g["atr"]
        g["supertrend_lower"] = hl2 - 3.0 * g["atr"]
        for lag in [1, 2, 3, 5, 10]:
            g[f"return_lag_{lag}"] = g["returns"].shift(lag)
            g[f"close_lag_{lag}"] = g["close"].shift(lag)
        g["momentum_score"] = (
            (g["rsi"] - 50) / 50 * 0.3
            + (g["macd_histogram"].fillna(0)) / (g["atr"] + 1e-9) * 0.3
            + g["dist_from_vwap"] * 0.2
            + (g["volume_ratio"] - 1) * 0.1
            + g["close_open_range"] * 0.1
        )
        g["trend_strength"] = g["adx"] / 100.0
        g["vol_breakout"] = (g["bb_width"] > g["bb_width"].rolling(20).mean()).astype(int)
        return g

    df = df.groupby("ticker", group_keys=False).apply(_calc)
    if "vix_close" in df.columns and df["vix_close"].notna().sum() > 10:
        df["vix_ma10"] = df["vix_close"].rolling(10).mean()
        df["vix_regime_hi"] = (df["vix_close"] > 20).astype(int)
        df["vix_regime_lo"] = (df["vix_close"] < 12).astype(int)
    else:
        df["vix_ma10"] = 0.0
        df["vix_regime_hi"] = 0
        df["vix_regime_lo"] = 0
    return df


def _merge_daily_pivots(df: pd.DataFrame, daily_pivot_df: pd.DataFrame) -> pd.DataFrame:
    idx_tz = df.index.tz
    df_no_tz = df.copy()
    df_no_tz.index = df_no_tz.index.tz_localize(None)
    pivot_shifted = daily_pivot_df.shift(1)
    pivot_idx = pd.to_datetime(pivot_shifted.index)
    if pivot_idx.tz is not None:
        pivot_idx = pivot_idx.tz_localize(None)
    pivot_shifted.index = pivot_idx.normalize()
    df_no_tz["_date"] = df_no_tz.index.normalize()
    merged = df_no_tz.join(pivot_shifted, on="_date", how="left", rsuffix="_pv")
    merged.drop(columns=["_date"], inplace=True)
    merged.index = df_no_tz.index
    merged.index = merged.index.tz_localize(idx_tz)
    for col in daily_pivot_df.columns:
        if col in merged.columns:
            merged[col] = merged[col].ffill()
    return merged


def add_pivot_features(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    zero_cols = [
        "classic_pivot",
        "classic_r1",
        "classic_r2",
        "classic_r3",
        "classic_s1",
        "classic_s2",
        "classic_s3",
        "cam_r1",
        "cam_r2",
        "cam_r3",
        "cam_r4",
        "cam_s1",
        "cam_s2",
        "cam_s3",
        "cam_s4",
        "fib_pivot",
        "fib_r1",
        "fib_r2",
        "fib_r3",
        "fib_s1",
        "fib_s2",
        "fib_s3",
        "woodie_pivot",
        "woodie_r1",
        "woodie_r2",
        "woodie_s1",
        "woodie_s2",
        "pct_dist_pivot",
        "pct_dist_r1",
        "pct_dist_r2",
        "pct_dist_r3",
        "pct_dist_s1",
        "pct_dist_s2",
        "pct_dist_s3",
        "pct_dist_cam_r3",
        "pct_dist_cam_s3",
        "pct_dist_cam_r4",
        "pct_dist_cam_s4",
        "pct_dist_fib_r1",
        "pct_dist_fib_s1",
        "pct_dist_fib_r2",
        "pct_dist_fib_s2",
        "above_pivot",
        "above_r1",
        "above_r2",
        "above_r3",
        "below_s1",
        "below_s2",
        "below_s3",
        "pivot_zone",
        "r1_touched_rejected",
        "r2_touched_rejected",
        "s1_touched_bounced",
        "s2_touched_bounced",
        "r1_breakout",
        "r2_breakout",
        "s1_breakdown",
        "s2_breakdown",
        "cam_range",
        "inside_cam_range",
    ]
    if daily.empty:
        for c in zero_cols:
            df[c] = 0.0
        return df

    def _pivot_for_ticker(group):
        t = group["ticker"].iloc[0]
        tick_daily = daily[daily["ticker"] == t].copy() if "ticker" in daily.columns else daily.copy()
        if tick_daily.empty:
            for c in zero_cols:
                group[c] = 0.0
            return group
        d = tick_daily.copy()
        d.index = pd.to_datetime(d.index).normalize()
        hl = d["high"] - d["low"]
        tp = (d["high"] + d["low"] + d["close"]) / 3
        classic = pd.DataFrame(index=d.index)
        classic["classic_pivot"] = tp
        classic["classic_r1"] = 2 * tp - d["low"]
        classic["classic_r2"] = tp + hl
        classic["classic_r3"] = d["high"] + 2 * (tp - d["low"])
        classic["classic_s1"] = 2 * tp - d["high"]
        classic["classic_s2"] = tp - hl
        classic["classic_s3"] = d["low"] - 2 * (d["high"] - tp)
        group = _merge_daily_pivots(group, classic)
        cam = pd.DataFrame(index=d.index)
        cam["cam_r1"] = d["close"] + hl * 1.1 / 12
        cam["cam_r2"] = d["close"] + hl * 1.1 / 6
        cam["cam_r3"] = d["close"] + hl * 1.1 / 4
        cam["cam_r4"] = d["close"] + hl * 1.1 / 2
        cam["cam_s1"] = d["close"] - hl * 1.1 / 12
        cam["cam_s2"] = d["close"] - hl * 1.1 / 6
        cam["cam_s3"] = d["close"] - hl * 1.1 / 4
        cam["cam_s4"] = d["close"] - hl * 1.1 / 2
        group = _merge_daily_pivots(group, cam)
        fib = pd.DataFrame(index=d.index)
        fib["fib_pivot"] = tp
        fib["fib_r1"] = tp + 0.382 * hl
        fib["fib_r2"] = tp + 0.618 * hl
        fib["fib_r3"] = tp + 1.000 * hl
        fib["fib_s1"] = tp - 0.382 * hl
        fib["fib_s2"] = tp - 0.618 * hl
        fib["fib_s3"] = tp - 1.000 * hl
        group = _merge_daily_pivots(group, fib)
        wood = pd.DataFrame(index=d.index)
        wood["woodie_pivot"] = (d["high"] + d["low"] + 2 * d["close"]) / 4
        wood["woodie_r1"] = 2 * wood["woodie_pivot"] - d["low"]
        wood["woodie_r2"] = wood["woodie_pivot"] + hl
        wood["woodie_s1"] = 2 * wood["woodie_pivot"] - d["high"]
        wood["woodie_s2"] = wood["woodie_pivot"] - hl
        group = _merge_daily_pivots(group, wood)
        c = group["close"]
        for col, name in [
            ("classic_pivot", "pct_dist_pivot"),
            ("classic_r1", "pct_dist_r1"),
            ("classic_r2", "pct_dist_r2"),
            ("classic_r3", "pct_dist_r3"),
            ("classic_s1", "pct_dist_s1"),
            ("classic_s2", "pct_dist_s2"),
            ("classic_s3", "pct_dist_s3"),
            ("cam_r3", "pct_dist_cam_r3"),
            ("cam_s3", "pct_dist_cam_s3"),
            ("cam_r4", "pct_dist_cam_r4"),
            ("cam_s4", "pct_dist_cam_s4"),
            ("fib_r1", "pct_dist_fib_r1"),
            ("fib_s1", "pct_dist_fib_s1"),
            ("fib_r2", "pct_dist_fib_r2"),
            ("fib_s2", "pct_dist_fib_s2"),
        ]:
            if col in group.columns:
                group[name] = (c - group[col]) / (group[col] + 1e-9) * 100
        group["above_pivot"] = (c > group.get("classic_pivot", 0)).astype(int)
        group["above_r1"] = (c > group.get("classic_r1", 0)).astype(int)
        group["above_r2"] = (c > group.get("classic_r2", 0)).astype(int)
        group["above_r3"] = (c > group.get("classic_r3", 0)).astype(int)
        group["below_s1"] = (c < group.get("classic_s1", 9e9)).astype(int)
        group["below_s2"] = (c < group.get("classic_s2", 9e9)).astype(int)
        group["below_s3"] = (c < group.get("classic_s3", 9e9)).astype(int)
        conditions = [
            c > group.get("classic_r3", 9e9),
            c > group.get("classic_r2", 9e9),
            c > group.get("classic_r1", 9e9),
            c > group.get("classic_pivot", 9e9),
            c > group.get("classic_s1", 9e9),
            c > group.get("classic_s2", 9e9),
            c > group.get("classic_s3", 9e9),
        ]
        group["pivot_zone"] = np.select(conditions, [6, 5, 4, 3, 2, 1, 0], default=-1)
        h, lo = group["high"], group["low"]
        group["r1_touched_rejected"] = (
            (h >= group.get("classic_r1", 9e9)) & (c < group.get("classic_r1", 9e9))
        ).astype(int)
        group["r2_touched_rejected"] = (
            (h >= group.get("classic_r2", 9e9)) & (c < group.get("classic_r2", 9e9))
        ).astype(int)
        group["s1_touched_bounced"] = (
            (lo <= group.get("classic_s1", 0)) & (c > group.get("classic_s1", 0))
        ).astype(int)
        group["s2_touched_bounced"] = (
            (lo <= group.get("classic_s2", 0)) & (c > group.get("classic_s2", 0))
        ).astype(int)
        group["r1_breakout"] = (c > group.get("classic_r1", 9e9)).astype(int)
        group["r2_breakout"] = (c > group.get("classic_r2", 9e9)).astype(int)
        group["s1_breakdown"] = (c < group.get("classic_s1", 0)).astype(int)
        group["s2_breakdown"] = (c < group.get("classic_s2", 0)).astype(int)
        group["cam_range"] = group.get("cam_r3", 0) - group.get("cam_s3", 0)
        group["inside_cam_range"] = (
            (c >= group.get("cam_s3", 0)) & (c <= group.get("cam_r3", 9e9))
        ).astype(int)
        return group

    df = df.groupby("ticker", group_keys=False).apply(_pivot_for_ticker)
    for c in zero_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df


def add_nifty_specific_features(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _nifty_feats(group):
        t = group["ticker"].iloc[0]
        tick_daily = (
            daily[daily["ticker"] == t].copy()
            if ("ticker" in daily.columns and not daily.empty)
            else daily.copy()
        )
        if not tick_daily.empty:
            d = tick_daily.copy()
            d.index = pd.to_datetime(d.index).normalize()
            df_date = group.index.normalize().tz_localize(None)
            group["pdh"] = d["high"].shift(1).reindex(df_date).values
            group["pdl"] = d["low"].shift(1).reindex(df_date).values
            group["pdc"] = d["close"].shift(1).reindex(df_date).values
        else:
            group["pdh"] = np.nan
            group["pdl"] = np.nan
            group["pdc"] = np.nan
        group["pdh"] = group["pdh"].ffill().fillna(0)
        group["pdl"] = group["pdl"].ffill().fillna(0)
        group["pdc"] = group["pdc"].ffill().fillna(0)
        group["dist_pdh"] = (group["close"] - group["pdh"]) / (group["pdh"] + 1e-9)
        group["dist_pdl"] = (group["close"] - group["pdl"]) / (group["pdl"] + 1e-9)
        group["above_pdh"] = (group["close"] > group["pdh"]).astype(int)
        group["below_pdl"] = (group["close"] < group["pdl"]).astype(int)
        daily_open = group.groupby(group.index.date)["open"].transform("first")
        group["gap_pct"] = (daily_open - group["pdc"]) / (group["pdc"] + 1e-9)
        group["gap_up"] = (group["gap_pct"] > 0.003).astype(int)
        group["gap_down"] = (group["gap_pct"] < -0.003).astype(int)
        group["hour"] = group.index.hour
        group["minute"] = group.index.minute
        group["day_of_week"] = group.index.dayofweek
        group["day_of_month"] = group.index.day
        group["month"] = group.index.month
        group["is_monday"] = (group.index.dayofweek == 0).astype(int)
        group["is_friday"] = (group.index.dayofweek == 4).astype(int)
        h = group.index.hour
        m = group.index.minute
        total_min = h * 60 + m
        group["is_opening_hour"] = ((total_min >= 9 * 60 + 15) & (total_min < 10 * 60 + 30)).astype(int)
        group["is_lunch_hour"] = ((total_min >= 12 * 60) & (total_min < 14 * 60)).astype(int)
        group["is_closing_hour"] = ((total_min >= 14 * 60 + 30) & (total_min <= 15 * 60 + 30)).astype(int)
        group["mins_since_open"] = total_min - (9 * 60 + 15)
        group["mins_to_close"] = (15 * 60 + 30) - total_min
        idx_series = group.index.to_series()
        last_thu_of_month = idx_series.groupby([group.index.year, group.index.month]).transform(
            lambda s: s[s.dt.dayofweek == 3].max() if (s.dt.dayofweek == 3).any() else pd.NaT
        )
        idx_tz = group.index.tz
        lt_idx = pd.DatetimeIndex(last_thu_of_month.values)
        if idx_tz is not None and getattr(lt_idx, "tz", None) is None:
            lt_idx = lt_idx.tz_localize(idx_tz)
        last_thu_of_month = pd.Series(lt_idx, index=group.index)
        win_start = last_thu_of_month - pd.Timedelta(days=4)
        group["is_expiry_week"] = (
            (idx_series >= win_start) & (idx_series <= last_thu_of_month)
        ).astype(int)
        return group

    df = df.groupby("ticker", group_keys=False).apply(_nifty_feats)
    return df


def add_macro_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "vix_close" in df.columns:
        if "vix_ma10" not in df.columns:
            df["vix_ma10"] = df["vix_close"].rolling(10).mean()
        if "vix_regime_hi" not in df.columns:
            df["vix_regime_hi"] = (df["vix_close"] > 20).astype(int)
        if "vix_regime_lo" not in df.columns:
            df["vix_regime_lo"] = (df["vix_close"] < 12).astype(int)
    else:
        if "vix_ma10" not in df.columns:
            df["vix_ma10"] = 0.0
        if "vix_regime_hi" not in df.columns:
            df["vix_regime_hi"] = 0
        if "vix_regime_lo" not in df.columns:
            df["vix_regime_lo"] = 0
    if "banknifty" in df.columns and "close" in df.columns:
        df["bank_nifty_ratio"] = df["banknifty"] / (df["close"] + 1e-9)
    if "sp500_ret1d" in df.columns:
        df["us_strong_bull"] = (df["sp500_ret1d"] > 0.01).astype(int)
        df["us_strong_bear"] = (df["sp500_ret1d"] < -0.01).astype(int)
    risk_cols = [c for c in ["global_fear", "crude_spike", "rupee_weak", "risk_off_gold"] if c in df.columns]
    df["macro_risk_score"] = df[risk_cols].mean(axis=1) if risk_cols else 0.0
    macro_prefixes = [
        "sp500",
        "usdinr",
        "crude",
        "gold",
        "us_vix",
        "banknifty",
        "dow_fut",
        "news_",
        "rupee",
        "risk_off",
        "us_bull",
        "us_bear",
        "global_",
        "bank_",
    ]
    for col in df.columns:
        if any(col.startswith(p) for p in macro_prefixes):
            if df[col].isna().any():
                df[col] = df[col].ffill().fillna(0)
    return df


def merge_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    path = get_bot_data_dir() / MACRO_SENTIMENT_CSV
    if not path.is_file():
        logger.warning("%s not found at %s — macro columns stay unset/zero.", MACRO_SENTIMENT_CSV, path)
        return df
    macro = pd.read_csv(path, index_col=0, parse_dates=True)
    macro.columns = [c.lower().strip() for c in macro.columns]
    macro.index.name = "datetime"
    if macro.index.tz is None:
        macro.index = macro.index.tz_localize("Asia/Kolkata")
    else:
        macro.index = macro.index.tz_convert("Asia/Kolkata")
    for prefix, possible_cols in [
        ("sp500", ["sp500_close", "sp500_adj_close"]),
        ("usdinr", ["usdinr_close", "usdinr_adj_close"]),
        ("crude", ["crude_close", "crude_adj_close"]),
        ("gold", ["gold_close", "gold_adj_close"]),
        ("us_vix", ["us_vix_close", "us_vix_adj_close"]),
        ("banknifty", ["bank_close", "banknifty_close"]),
        ("dow_fut", ["dow_fut_close", "dow_fut_adj_close", "sp500_close", "sp500_adj_close"]),
    ]:
        close_col = next((c for c in possible_cols if c in macro.columns), None)
        if close_col:
            macro[prefix] = macro[close_col]
            macro[f"{prefix}_ret1d"] = macro[close_col].pct_change(1)
            macro[f"{prefix}_ret5d"] = macro[close_col].pct_change(5)
        else:
            macro[prefix] = 0.0
            macro[f"{prefix}_ret1d"] = 0.0
            macro[f"{prefix}_ret5d"] = 0.0
    macro = macro.ffill().fillna(0)
    df_tz = df.index.tz
    df_reset = df.reset_index()
    df_reset.rename(columns={"index": "datetime"}, inplace=True)
    df_reset["datetime"] = pd.to_datetime(df_reset["datetime"]).dt.tz_localize(None)
    macro_reset = macro.reset_index()
    macro_reset["datetime"] = pd.to_datetime(macro_reset["datetime"]).dt.tz_localize(None)
    existing = set(df_reset.columns)
    macro_cols = ["datetime"] + [c for c in macro_reset.columns if c != "datetime" and c not in existing]
    macro_reset = macro_reset[macro_cols]
    df_reset = df_reset.sort_values("datetime")
    macro_reset = macro_reset.sort_values("datetime")
    merged = pd.merge_asof(df_reset, macro_reset, on="datetime", direction="backward")
    merged = merged.set_index("datetime")
    merged.index = pd.DatetimeIndex(merged.index)
    if df_tz is not None:
        merged.index = merged.index.tz_localize(df_tz)
    logger.debug("Merged %d macro-derived columns into panel.", len(macro_cols) - 1)
    return merged


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
    return max(-1.0, min(1.0, score))


def create_announcement_features(
    df_1h: pd.DataFrame,
    announcements_db: pd.DataFrame,
) -> pd.DataFrame:
    news_col_names = [
        "news_score_6h",
        "news_score_24h",
        "news_score_48h",
        "has_positive_news_6h",
        "has_negative_news_6h",
        "has_positive_news_24h",
        "has_negative_news_24h",
        "hours_since_last_ann",
        "ann_count_24h",
        "earnings_event_24h",
        "dividend_event_24h",
        "penalty_event_24h",
        "order_event_24h",
    ]
    if announcements_db.empty:
        for col in news_col_names:
            df_1h[col] = 0.0
        return df_1h

    df = df_1h.copy()
    ann = announcements_db.copy()
    ann["datetime"] = pd.to_datetime(ann["datetime"])
    ann["score"] = ann.apply(
        lambda r: score_announcement(r["subject"], r.get("description", "")), axis=1
    )
    ann["is_earnings"] = (
        ann["subject"]
        .str.lower()
        .str.contains(
            "financial result|quarterly result|annual result|profit|revenue",
            na=False,
        )
        .astype(int)
    )
    ann["is_dividend"] = (
        ann["subject"].str.lower().str.contains("dividend|bonus|buyback|split", na=False).astype(int)
    )
    ann["is_penalty"] = (
        ann["subject"].str.lower().str.contains("penalty|fine|probe|sebi|ed |cbi|fraud", na=False).astype(int)
    )
    ann["is_order"] = (
        ann["subject"].str.lower().str.contains("order|contract|award|win|bagged", na=False).astype(int)
    )

    ann_by_symbol = {sym: grp.sort_values("datetime") for sym, grp in ann.groupby("symbol")}

    logger.info("Announcement features: scanning %d rows …", len(df))
    new_cols: dict[str, Any] = {
        "news_score_6h": np.zeros(len(df)),
        "news_score_24h": np.zeros(len(df)),
        "news_score_48h": np.zeros(len(df)),
        "has_positive_news_6h": np.zeros(len(df)),
        "has_negative_news_6h": np.zeros(len(df)),
        "has_positive_news_24h": np.zeros(len(df)),
        "has_negative_news_24h": np.zeros(len(df)),
        "hours_since_last_ann": np.full(len(df), 999.0),
        "ann_count_24h": np.zeros(len(df)),
        "earnings_event_24h": np.zeros(len(df)),
        "dividend_event_24h": np.zeros(len(df)),
        "penalty_event_24h": np.zeros(len(df)),
        "order_event_24h": np.zeros(len(df)),
    }
    for i, (idx, row) in enumerate(df.iterrows()):
        symbol = row["ticker"].replace(".NS", "")
        if symbol not in ann_by_symbol:
            continue
        sym_ann = ann_by_symbol[symbol]
        ts_naive = (
            idx.tz_convert("Asia/Kolkata").replace(tzinfo=None)
            if hasattr(idx, "tzinfo") and idx.tzinfo
            else idx
        )
        past_ann = sym_ann[sym_ann["datetime"] < ts_naive]
        if past_ann.empty:
            continue
        t_6h = ts_naive - pd.Timedelta(hours=6)
        t_24h = ts_naive - pd.Timedelta(hours=24)
        t_48h = ts_naive - pd.Timedelta(hours=48)
        ann_6h = past_ann[past_ann["datetime"] >= t_6h]
        ann_24h = past_ann[past_ann["datetime"] >= t_24h]
        ann_48h = past_ann[past_ann["datetime"] >= t_48h]
        most_recent = past_ann["datetime"].max()
        new_cols["hours_since_last_ann"][i] = min(
            (ts_naive - most_recent).total_seconds() / 3600, 999.0
        )
        if not ann_6h.empty:
            s6 = ann_6h["score"].values
            new_cols["news_score_6h"][i] = float(np.mean(s6))
            new_cols["has_positive_news_6h"][i] = int(np.any(s6 > 0.1))
            new_cols["has_negative_news_6h"][i] = int(np.any(s6 < -0.1))
        if not ann_24h.empty:
            s24 = ann_24h["score"].values
            new_cols["news_score_24h"][i] = float(np.mean(s24))
            new_cols["has_positive_news_24h"][i] = int(np.any(s24 > 0.1))
            new_cols["has_negative_news_24h"][i] = int(np.any(s24 < -0.1))
            new_cols["ann_count_24h"][i] = len(ann_24h)
            new_cols["earnings_event_24h"][i] = int(ann_24h["is_earnings"].any())
            new_cols["dividend_event_24h"][i] = int(ann_24h["is_dividend"].any())
            new_cols["penalty_event_24h"][i] = int(ann_24h["is_penalty"].any())
            new_cols["order_event_24h"][i] = int(ann_24h["is_order"].any())
        if not ann_48h.empty:
            new_cols["news_score_48h"][i] = float(ann_48h["score"].mean())
    for col, values in new_cols.items():
        df[col] = values
    df[list(new_cols.keys())] = df[list(new_cols.keys())].fillna(0.0)
    return df


def _announcement_layer(df: pd.DataFrame) -> pd.DataFrame:
    flag = os.environ.get("LIVE_ANNOUNCEMENT_FEATURES", "auto").strip().lower()
    ann_path = get_bot_data_dir() / ANNOUNCEMENTS_CSV
    ann_exists = ann_path.is_file()

    if flag in ("0", "false", "off"):
        return create_announcement_features(df, pd.DataFrame())
    if flag in ("1", "true", "on"):
        if not ann_exists:
            return create_announcement_features(df, pd.DataFrame())
        ann_db = pd.read_csv(ann_path, parse_dates=["datetime"])
        return create_announcement_features(df, ann_db)
    if not ann_exists:
        return create_announcement_features(df, pd.DataFrame())
    ann_db = pd.read_csv(ann_path, parse_dates=["datetime"])
    return create_announcement_features(df, ann_db)


def apply_inference_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full parity path: technicals → pivots → Nifty session → macro merge → sentiment → announcements.
    Same order as former `bot_training_mirror` + stock_trading_bot live pipeline.
    """
    if df.empty:
        return df
    df_daily = load_daily_panel_data()
    out = df.copy()
    out = add_technical_indicators(out)
    out = add_pivot_features(out, df_daily)
    out = add_nifty_specific_features(out, df_daily)
    out = merge_macro_features(out)
    out = add_macro_sentiment_features(out)
    out = _announcement_layer(out)
    return out
