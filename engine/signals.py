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
    NEWS_BLOCK_THRESHOLD,
    XGB_MODEL_CANDIDATES, LGB_MODEL_CANDIDATES, CAT_MODEL_CANDIDATES,
    TCN_MODEL_CANDIDATES, SCALER_CANDIDATES, FEATURE_COLS_CANDIDATES,
)

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


def _ensemble_predict(X: np.ndarray, models: dict) -> np.ndarray:
    """
    Run the 4-model weighted ensemble and return probability of BUY.
    Weights: XGB 28% | LGB 22% | CAT 28% | TCN 22%
    """
    probs = []

    if models["xgb"] is not None:
        p = models["xgb"].predict_proba(X)[:, 1]
        probs.append((p, 0.28))

    if models["lgb"] is not None:
        p = models["lgb"].predict_proba(X)[:, 1]
        probs.append((p, 0.22))

    if models["cat"] is not None:
        p = models["cat"].predict_proba(X)[:, 1]
        probs.append((p, 0.28))

    if models["tcn"] is not None:
        try:
            seq = X.reshape(X.shape[0], LOOKBACK, -1)
            p = models["tcn"].predict(seq, verbose=0).flatten()
            probs.append((p, 0.22))
        except Exception:
            pass

    if not probs:
        raise RuntimeError("No models available for prediction")

    # Normalise weights
    total_w = sum(w for _, w in probs)
    ensemble = sum(p * (w / total_w) for p, w in probs)
    return ensemble


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
        "skipped_feature_overlap": 0,
        "skipped_empty_frame": 0,
        "model_errors": 0,
        "skipped_low_confidence": 0,
        "rows_5m_for_first_ticker": None,
        "feature_cols_count": 0,
        "min_feature_overlap": 0,
        "max_buy_prob": None,
        "max_short_strength": None,
        "min_confidence_buy_pct": MIN_CONFIDENCE,
        "min_confidence_short_pct": MIN_CONFIDENCE_SHORT,
        "picks_before_news_filter": 0,
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
        tick_df = df_5m[df_5m["ticker"] == ticker].sort_index().tail(LOOKBACK + 5)
        if first_ticker is not None and ticker == first_ticker:
            meta["rows_5m_for_first_ticker"] = int(len(tick_df))

        if len(tick_df) < LOOKBACK:
            meta["skipped_short_history"] += 1
            continue

        if "close" not in tick_df.columns:
            meta["skipped_no_close"] += 1
            continue

        # Build full training feature vector in exact order expected by scaler/models.
        available = [c for c in feature_cols if c in tick_df.columns]
        min_needed = max(5, int(len(feature_cols) * 0.2))
        meta["min_feature_overlap"] = min_needed
        if len(available) < min_needed:
            # Not enough feature overlap; skip this ticker to avoid garbage inference.
            meta["skipped_feature_overlap"] += 1
            continue
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
            X_last = X_all[-1:].copy()
            prob = float(_ensemble_predict(X_last, models)[0])
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

    # Block BUY signals with very negative news
    picks = picks[
        ~((picks["signal"] == "BUY") & (picks["news_score"] < NEWS_BLOCK_THRESHOLD))
    ].reset_index(drop=True)

    meta["picks_after_news_filter"] = int(len(picks))
    return picks, meta
