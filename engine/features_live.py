"""
Live inference feature pipeline aligned with stock_trading_bot.py training schema.

Training selects TOP_N_FEATURES from ALL_FEATURE_COLUMNS; live OHLCV must expose those
names (missing macro/pivot/news columns are stubbed with 0 like empty-daily paths in training).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from engine.bot_training_mirror import apply_stock_bot_5m_panel

# Master list — must stay in sync with stock_trading_bot.py ALL_FEATURE_COLUMNS.
ALL_FEATURE_COLUMNS = [
    "sma_9", "sma_21", "sma_50", "sma_200", "ema_9", "ema_21", "ema_50",
    "rsi", "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_lower", "bb_middle", "bb_width", "bb_pct", "atr",
    "stoch_k", "stoch_d", "adx", "plus_di", "minus_di",
    "vwap", "dist_from_vwap", "price_above_vwap", "obv", "volume_sma_20", "volume_ratio",
    "returns", "log_returns", "high_low_range", "close_open_range",
    "rolling_mean_5", "rolling_std_5", "rolling_mean_10", "rolling_std_10",
    "rolling_mean_20", "rolling_std_20", "zscore_20",
    "ema_9_21_cross", "sma_50_200_cross",
    "supertrend_upper", "supertrend_lower",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5", "return_lag_10",
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_10",
    "momentum_score", "trend_strength", "vol_breakout",
    "classic_pivot", "classic_r1", "classic_r2", "classic_r3",
    "classic_s1", "classic_s2", "classic_s3",
    "cam_r1", "cam_r2", "cam_r3", "cam_r4", "cam_s1", "cam_s2", "cam_s3", "cam_s4",
    "fib_pivot", "fib_r1", "fib_r2", "fib_r3", "fib_s1", "fib_s2", "fib_s3",
    "woodie_pivot", "woodie_r1", "woodie_r2", "woodie_s1", "woodie_s2",
    "pct_dist_pivot", "pct_dist_r1", "pct_dist_r2", "pct_dist_r3",
    "pct_dist_s1", "pct_dist_s2", "pct_dist_s3",
    "pct_dist_cam_r3", "pct_dist_cam_s3", "pct_dist_cam_r4", "pct_dist_cam_s4",
    "pct_dist_fib_r1", "pct_dist_fib_s1", "pct_dist_fib_r2", "pct_dist_fib_s2",
    "above_pivot", "above_r1", "above_r2", "above_r3",
    "below_s1", "below_s2", "below_s3", "pivot_zone",
    "r1_touched_rejected", "r2_touched_rejected",
    "s1_touched_bounced", "s2_touched_bounced",
    "r1_breakout", "r2_breakout", "s1_breakdown", "s2_breakdown",
    "cam_range", "inside_cam_range",
    "hour", "minute", "day_of_week", "day_of_month", "month",
    "is_monday", "is_friday", "is_opening_hour", "is_lunch_hour", "is_closing_hour",
    "mins_since_open", "mins_to_close",
    "vix_ma10", "vix_regime_hi", "vix_regime_lo",
    "pdh", "pdl", "pdc", "dist_pdh", "dist_pdl", "above_pdh", "below_pdl",
    "gap_pct", "gap_up", "gap_down", "is_expiry_week",
    "sp500", "sp500_ret1d", "sp500_ret5d",
    "usdinr", "usdinr_ret1d",
    "crude", "crude_ret1d", "crude_spike",
    "gold", "gold_ret1d", "risk_off_gold",
    "us_vix", "us_vix_ret1d", "global_fear",
    "banknifty", "banknifty_ret1d", "bank_nifty_ratio",
    "macro_risk_score",
    "news_score_6h", "news_score_24h", "news_score_48h",
    "has_positive_news_6h", "has_negative_news_6h",
    "has_positive_news_24h", "has_negative_news_24h",
    "hours_since_last_ann", "ann_count_24h",
    "earnings_event_24h", "dividend_event_24h",
    "penalty_event_24h", "order_event_24h",
]


def _technical_calc(group: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker block mirroring stock_trading_bot.add_technical_indicators._calc."""
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
    g["obv"] = ta.volume.OnBalanceVolumeIndicator(g["close"], g["volume"]).on_balance_volume()
    g["volume_sma_20"] = g["volume"].rolling(20).mean()
    g["volume_ratio"] = g["volume"] / (g["volume_sma_20"] + 1e-9)
    g["returns"] = g["close"].pct_change()
    g["log_returns"] = np.log(g["close"] / g["close"].shift(1))
    g["high_low_range"] = (g["high"] - g["low"]) / (g["close"] + 1e-9)
    g["close_open_range"] = (g["close"] - g["open"]) / (g["open"] + 1e-9)
    for w in (5, 10, 20):
        g[f"rolling_mean_{w}"] = g["close"].rolling(w).mean()
        g[f"rolling_std_{w}"] = g["close"].rolling(w).std()
    g["zscore_20"] = (g["close"] - g["rolling_mean_20"]) / (g["rolling_std_20"] + 1e-9)
    g["ema_9_21_cross"] = (g["ema_9"] > g["ema_21"]).astype(int)
    g["sma_50_200_cross"] = (g["sma_50"] > g["sma_200"]).astype(int)
    hl2 = (g["high"] + g["low"]) / 2
    g["supertrend_upper"] = hl2 + 3.0 * g["atr"]
    g["supertrend_lower"] = hl2 - 3.0 * g["atr"]
    for lag in (1, 2, 3, 5, 10):
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


def add_technical_indicators_live(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return df
    out = df.groupby("ticker", group_keys=False).apply(_technical_calc)
    if "vix_close" in out.columns and out["vix_close"].notna().sum() > 10:
        out["vix_ma10"] = out["vix_close"].rolling(10).mean()
        out["vix_regime_hi"] = (out["vix_close"] > 20).astype(int)
        out["vix_regime_lo"] = (out["vix_close"] < 12).astype(int)
    else:
        out["vix_ma10"] = 0.0
        out["vix_regime_hi"] = 0
        out["vix_regime_lo"] = 0
    return out


def _session_prevday_calc(g: pd.DataFrame) -> pd.DataFrame:
    """IST calendar features + prior-day H/L/C from the intraday window itself."""
    g = g.sort_index()
    idx_n = pd.DatetimeIndex(g.index.normalize())
    daily_h = g.groupby(idx_n, sort=False)["high"].max()
    daily_l = g.groupby(idx_n, sort=False)["low"].min()
    daily_c = g.groupby(idx_n, sort=False)["close"].last()
    pdh_s = daily_h.shift(1)
    pdl_s = daily_l.shift(1)
    pdc_s = daily_c.shift(1)
    g = g.copy()
    g["pdh"] = pdh_s.reindex(idx_n).to_numpy()
    g["pdl"] = pdl_s.reindex(idx_n).to_numpy()
    g["pdc"] = pdc_s.reindex(idx_n).to_numpy()
    g["pdh"] = g["pdh"].ffill().fillna(0.0)
    g["pdl"] = g["pdl"].ffill().fillna(0.0)
    g["pdc"] = g["pdc"].ffill().fillna(0.0)
    g["dist_pdh"] = (g["close"] - g["pdh"]) / (g["pdh"] + 1e-9)
    g["dist_pdl"] = (g["close"] - g["pdl"]) / (g["pdl"] + 1e-9)
    g["above_pdh"] = (g["close"] > g["pdh"]).astype(int)
    g["below_pdl"] = (g["close"] < g["pdl"]).astype(int)
    daily_open = g.groupby(g.index.normalize(), sort=False)["open"].transform("first")
    g["gap_pct"] = (daily_open - g["pdc"]) / (g["pdc"] + 1e-9)
    g["gap_up"] = (g["gap_pct"] > 0.003).astype(int)
    g["gap_down"] = (g["gap_pct"] < -0.003).astype(int)

    g["hour"] = g.index.hour
    g["minute"] = g.index.minute
    g["day_of_week"] = g.index.dayofweek
    g["day_of_month"] = g.index.day
    g["month"] = g.index.month
    g["is_monday"] = (g.index.dayofweek == 0).astype(int)
    g["is_friday"] = (g.index.dayofweek == 4).astype(int)
    h = g.index.hour
    m = g.index.minute
    total_min = h * 60 + m
    g["is_opening_hour"] = ((total_min >= 9 * 60 + 15) & (total_min < 10 * 60 + 30)).astype(int)
    g["is_lunch_hour"] = ((total_min >= 12 * 60) & (total_min < 14 * 60)).astype(int)
    g["is_closing_hour"] = ((total_min >= 14 * 60 + 30) & (total_min <= 15 * 60 + 30)).astype(int)
    g["mins_since_open"] = total_min - (9 * 60 + 15)
    g["mins_to_close"] = (15 * 60 + 30) - total_min

    idx_series = g.index.to_series()
    last_thu_of_month = idx_series.groupby([g.index.year, g.index.month]).transform(
        lambda s: s[s.dt.dayofweek == 3].max() if (s.dt.dayofweek == 3).any() else pd.NaT
    )
    g["is_expiry_week"] = (
        (idx_series >= (last_thu_of_month - pd.Timedelta(days=4)))
        & (idx_series <= last_thu_of_month)
    ).astype(int)
    return g


def add_session_and_prevday_live(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return df
    return df.groupby("ticker", group_keys=False).apply(_session_prevday_calc)


def stub_missing_master_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every ALL_FEATURE_COLUMNS name exists (float 0) for scaler column alignment."""
    out = df.copy()
    for c in ALL_FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = 0.0
    return out


def enrich_intraday_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build training-aligned columns on OHLCV intraday data (one or many tickers).
    Expects lowercase open, high, low, close, volume, ticker; DatetimeIndex (IST).

    Uses the same pipeline as `stock_trading_bot.py` live/training (technical → pivot
    → Nifty → macro → sentiment → announcements). See `BOT_DATA_DIR` and
    `LIVE_ANNOUNCEMENT_FEATURES` in `bot_training_mirror.py`.
    """
    if df.empty:
        return df
    out = apply_stock_bot_5m_panel(df)
    out = stub_missing_master_columns(out)
    return out.replace([np.inf, -np.inf], np.nan)
