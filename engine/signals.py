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


def generate_signals(df_5m: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Generate BUY/SELL signals for the given tickers.
    Returns a DataFrame with columns:
      ticker, signal, confidence, close, news_score, combined
    sorted by combined score descending.
    """
    models = load_models()
    scaler      = models["scaler"]
    feature_cols = models["feature_cols"]

    rows = []
    for ticker in tickers:
        tick_df = df_5m[df_5m["ticker"] == ticker].sort_index().tail(LOOKBACK + 5)
        if len(tick_df) < LOOKBACK:
            continue

        if "close" not in tick_df.columns:
            continue

        # Build full training feature vector in exact order expected by scaler/models.
        feature_frame = (
            tick_df.reindex(columns=feature_cols)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        if feature_frame.empty:
            continue

        try:
            X_all = scaler.transform(feature_frame)
            X_last = X_all[-1:].copy()
            prob = float(_ensemble_predict(X_last, models)[0])
        except Exception:
            continue

        if prob >= MIN_CONFIDENCE / 100:
            signal = "BUY"
            conf   = prob * 100
        elif (1 - prob) >= MIN_CONFIDENCE_SHORT / 100:
            signal = "SELL"
            conf   = (1 - prob) * 100
        else:
            continue

        rows.append({
            "ticker":     ticker,
            "signal":     signal,
            "confidence": round(conf, 2),
            "close":      float(tick_df["close"].iloc[-1]),
            "news_score": 0.0,   # News sentiment added later if needed
            "combined":   round(prob * 0.7, 4),
        })

    if not rows:
        return pd.DataFrame()

    picks = pd.DataFrame(rows).sort_values("combined", ascending=False)

    # Block BUY signals with very negative news
    picks = picks[
        ~((picks["signal"] == "BUY") & (picks["news_score"] < NEWS_BLOCK_THRESHOLD))
    ].reset_index(drop=True)

    return picks
