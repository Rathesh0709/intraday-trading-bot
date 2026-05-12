"""
Compatibility shim: same entrypoints as before, now backed by `engine.inference_features`.

No `stock_trading_bot.py` import — logic lives in-repo under `engine/inference_features.py`.

Reference CSVs (optional) resolve via **BOT_DATA_DIR** — see `engine.config.get_bot_data_dir`.
"""

from __future__ import annotations

import pandas as pd

from engine.inference_features import apply_inference_feature_pipeline


def apply_stock_bot_5m_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Same steps as the former stock_trading_bot live 5m pipeline."""
    return apply_inference_feature_pipeline(df)


def apply_stock_bot_1h_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Same steps as the former 1h training/inference pipeline."""
    return apply_inference_feature_pipeline(df)
