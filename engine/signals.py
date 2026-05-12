"""
engine/signals.py — Load trained models and generate live BUY/SELL signals.

Model artifacts must be copied into `models/` inside `trading-app-backend`.
Both backend aliases and filenames from stock_trading_bot.py are supported.
"""

from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from engine.config import (
    LOOKBACK, MIN_CONFIDENCE, MIN_CONFIDENCE_SHORT,
    INFERENCE_TAIL_BARS,
    NEWS_BLOCK_THRESHOLD,
    XGB_MODEL_CANDIDATES, LGB_MODEL_CANDIDATES, CAT_MODEL_CANDIDATES,
    TCN_MODEL_CANDIDATES, SCALER_CANDIDATES, FEATURE_COLS_CANDIDATES,
)
from engine.live_news import get_live_combined_sentiment

# ── Module-level model cache ─────────────────────────────────────────────────
_models_cache: dict | None = None


def _pick_existing(paths: tuple[str, ...], required: bool = False) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    if required:
        raise FileNotFoundError(
            "Missing required model artifact. Checked: "
            + ", ".join(paths)
        )
    return None


def _safe_optional_load(path: Optional[str], name: str):
    if not path:
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"[Models] Skipping optional {name} ({path}): {exc}")
        return None


def load_models() -> dict:
    """Load all trained models from disk (cached after first load)."""
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    scaler_path = _pick_existing(SCALER_CANDIDATES, required=True)
    features_path = _pick_existing(FEATURE_COLS_CANDIDATES, required=True)

    xgb_path = _pick_existing(XGB_MODEL_CANDIDATES, required=True)
    lgb_path = _pick_existing(LGB_MODEL_CANDIDATES)
    cat_path = _pick_existing(CAT_MODEL_CANDIDATES)

    xgb_m = joblib.load(xgb_path)
    lgb_m = _safe_optional_load(lgb_path, "LightGBM model")
    cat_m = _safe_optional_load(cat_path, "CatBoost model")
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(features_path)
    if hasattr(feature_cols, "tolist"):
        feature_cols = feature_cols.tolist()
    feature_cols = list(feature_cols)

    seq_m = None
    seq_path = _pick_existing(TCN_MODEL_CANDIDATES)
    if seq_path:
        try:
            import tensorflow as tf
            seq_m = tf.keras.models.load_model(seq_path)
        except Exception:
            pass

    _models_cache = {
        "xgb": xgb_m, "lgb": lgb_m, "cat": cat_m,
        "tcn": seq_m, "scaler": scaler, "feature_cols": feature_cols,
    }
    print("[Models] All models loaded and cached.")
    return _models_cache


def models_ready() -> bool:
    """Return True if models have been loaded successfully."""
    try:
        load_models()
        return True
    except Exception as exc:
        print(f"[Models] Not ready: {exc}")
        return False


def _ensemble_predict_window(X_sc: np.ndarray, models: dict) -> float:
    """
    Match `stock_trading_bot.ensemble_predict` behavior:
    - Tree models score only the latest row
    - Sequence model (tcn slot) scores last LOOKBACK rows
    Weights: XGB 28% | LGB 22% | CAT 28% | TCN 22%
    """
    if X_sc.size == 0:
        raise RuntimeError("Empty feature window")

    xgb = models.get("xgb")
    lgb = models.get("lgb")
    cat = models.get("cat")
    seq = models.get("tcn")

    if xgb is None:
        raise RuntimeError("XGB model missing")

    xgb_p = float(xgb.predict_proba(X_sc[-1:])[:, 1][0])
    lgb_p = float(lgb.predict_proba(X_sc[-1:])[:, 1][0]) if lgb is not None else xgb_p
    cat_p = float(cat.predict_proba(X_sc[-1:])[:, 1][0]) if cat is not None else xgb_p

    if seq is not None and len(X_sc) >= LOOKBACK:
        try:
            seq_in = X_sc[-LOOKBACK:].reshape(1, LOOKBACK, -1)
            seq_p = float(seq.predict(seq_in, verbose=0).flatten()[0])
            return 0.28 * xgb_p + 0.22 * lgb_p + 0.28 * cat_p + 0.22 * seq_p
        except Exception:
            return 0.35 * xgb_p + 0.30 * lgb_p + 0.35 * cat_p

    return 0.35 * xgb_p + 0.30 * lgb_p + 0.35 * cat_p


def generate_signals(df_5m: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Generate BUY/SELL signals for the given tickers.
    Returns (DataFrame, diagnostics dict). DataFrame columns:
      ticker, signal, confidence, close, news_score, combined
    sorted by combined score descending.
    """
    meta: dict = {
        "tickers_requested": len(tickers),
        "skipped_short_history": 0,
        "skipped_no_close": 0,
        "skipped_empty_frame": 0,
        "model_errors": 0,
        "skipped_low_confidence": 0,
        "rows_5m_for_first_ticker": None,
        "feature_cols_count": 0,
        "feature_columns_native_present": None,
        "feature_columns_zero_filled": None,
        "max_buy_prob": None,
        "max_short_strength": None,
        "min_confidence_buy_pct": MIN_CONFIDENCE,
        "min_confidence_short_pct": MIN_CONFIDENCE_SHORT,
        "picks_before_news_filter": 0,
        "inference_tail_bars": INFERENCE_TAIL_BARS,
        "live_news_enabled": True,
    }

    models = load_models()
    scaler      = models["scaler"]
    feature_cols = models["feature_cols"]
    meta["feature_cols_count"] = len(feature_cols)

    rows = []
    max_buy_prob = -1.0
    max_short_strength = -1.0
    first_ticker = tickers[0] if tickers else None

    for ticker in tickers:
        tick_df = df_5m[df_5m["ticker"] == ticker].sort_index().tail(INFERENCE_TAIL_BARS)
        if first_ticker is not None and ticker == first_ticker:
            meta["rows_5m_for_first_ticker"] = int(len(tick_df))

        if len(tick_df) < LOOKBACK:
            meta["skipped_short_history"] += 1
            continue

        if "close" not in tick_df.columns:
            meta["skipped_no_close"] += 1
            continue

        present = [c for c in feature_cols if c in tick_df.columns]
        if first_ticker is not None and ticker == first_ticker:
            meta["feature_columns_native_present"] = len(present)
            meta["feature_columns_zero_filled"] = len(feature_cols) - len(present)

        # Align to training column order; missing live-only columns become 0 (same as training
        # fill for macro/pivot/news when unavailable).
        feature_frame = (
            tick_df.reindex(columns=feature_cols)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        if feature_frame.empty:
            meta["skipped_empty_frame"] += 1
            continue

        try:
            X_all = scaler.transform(feature_frame)
            prob = float(_ensemble_predict_window(X_all, models))
        except Exception:
            meta["model_errors"] += 1
            continue

        max_buy_prob = max(max_buy_prob, prob)
        max_short_strength = max(max_short_strength, 1.0 - prob)

        if prob >= MIN_CONFIDENCE / 100:
            signal = "BUY"
            conf   = prob * 100
        elif (1 - prob) >= MIN_CONFIDENCE_SHORT / 100:
            signal = "SELL"
            conf   = (1 - prob) * 100
        else:
            meta["skipped_low_confidence"] += 1
            continue

        rows.append({
            "ticker":     ticker,
            "signal":     signal,
            "confidence": round(conf, 2),
            "close":      float(tick_df["close"].iloc[-1]),
            "news_score": 0.0,   # News sentiment added later if needed
            "combined":   round((conf / 100.0) * 0.7, 4),
        })

    meta["max_buy_prob"] = round(max_buy_prob, 4) if max_buy_prob >= 0 else None
    meta["max_short_strength"] = round(max_short_strength, 4) if max_short_strength >= 0 else None

    if not rows:
        meta["picks_after_news_filter"] = 0
        return pd.DataFrame(), meta

    picks = pd.DataFrame(rows).sort_values("combined", ascending=False)
    meta["picks_before_news_filter"] = int(len(picks))

    # ── Live news sentiment (70% model + 30% news), like stock_trading_bot.py ──
    use_live_news = os.environ.get("USE_LIVE_NEWS", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )
    meta["live_news_enabled"] = bool(use_live_news)
    if use_live_news and not picks.empty:
        news_scores: dict[str, float] = {}
        for t in picks["ticker"].unique().tolist():
            try:
                news_scores[t] = float(get_live_combined_sentiment(t))
            except Exception:
                news_scores[t] = 0.0
        picks["news_score"] = picks["ticker"].map(news_scores).fillna(0.0)

        # news_norm maps [-1,+1] → [0,1] for BUY alignment; invert for shorts
        news_norm = (picks["news_score"].astype(float) + 1.0) / 2.0
        news_norm = np.where(picks["signal"] == "SELL", 1.0 - news_norm, news_norm)

        # Reconstruct prob_up from confidence+signal
        conf01 = picks["confidence"].astype(float) / 100.0
        prob_up = np.where(picks["signal"] == "BUY", conf01, 1.0 - conf01)
        picks["combined"] = np.round(0.70 * prob_up + 0.30 * news_norm, 4)
        picks = picks.sort_values("combined", ascending=False).reset_index(drop=True)

    # Block BUY signals with very negative news
    picks = picks[
        ~((picks["signal"] == "BUY") & (picks["news_score"] < NEWS_BLOCK_THRESHOLD))
    ].reset_index(drop=True)

    meta["picks_after_news_filter"] = int(len(picks))
    return picks, meta
