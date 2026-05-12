"""
Run the same 5m / 1h feature pipelines as `stock_trading_bot.py` for live inference.

Loads `stock_trading_bot.py` from the repo (or STOCK_TRADING_BOT_PATH) and calls:
  add_technical_indicators → add_pivot_features → add_nifty_specific_features
  → merge_macro_features → add_macro_sentiment_features
  → create_announcement_features (optional; see env below)

CSV paths inside the bot are **relative to the process cwd**. Use BOT_DATA_DIR (default:
parent of `trading-app-backend`, i.e. the monorepo root) so `nifty50_stocks_daily.csv`,
`macro_sentiment.csv`, and `nse_announcements_full.csv` resolve like local training.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_stb: Any = None


def _monorepo_root() -> Path:
    # trading-app-backend/engine/bot_training_mirror.py → …/stock-price-predictor
    return Path(__file__).resolve().parent.parent.parent


def _resolve_bot_py() -> Path:
    env = os.environ.get("STOCK_TRADING_BOT_PATH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _monorepo_root() / "stock_trading_bot.py"


def _load_stock_trading_bot():
    global _stb
    if _stb is not None:
        return _stb
    bot_py = _resolve_bot_py()
    if not bot_py.is_file():
        raise FileNotFoundError(
            f"Cannot find stock_trading_bot.py at {bot_py}. "
            "Clone the full repo or set STOCK_TRADING_BOT_PATH to that file."
        )
    spec = importlib.util.spec_from_file_location("stock_trading_bot_live", bot_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {bot_py}")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so relative imports inside the file behave
    sys.modules["stock_trading_bot_live"] = mod
    spec.loader.exec_module(mod)
    _stb = mod
    return mod


@contextmanager
def _chdir_data_root():
    """
    stock_trading_bot uses relative paths: nifty50_stocks_daily.csv, macro_sentiment.csv,
    nse_announcements_full.csv (ANN_DB_PATH).
    """
    root = os.environ.get("BOT_DATA_DIR", "").strip() or str(_monorepo_root())
    prev = os.getcwd()
    os.chdir(root)
    try:
        yield Path(root)
    finally:
        os.chdir(prev)


def _announcement_features(stb: Any, df: pd.DataFrame) -> pd.DataFrame:
    """
    Match training when ANN DB exists; otherwise same as paper live path (zeros).

    LIVE_ANNOUNCEMENT_FEATURES=0 skips reading the CSV even if present (faster on large panels).
    LIVE_ANNOUNCEMENT_FEATURES=1 forces loading ANN_DB_PATH when present.
    Default `auto`: load when file exists under BOT_DATA_DIR.
    """
    flag = os.environ.get("LIVE_ANNOUNCEMENT_FEATURES", "auto").strip().lower()
    with _chdir_data_root():
        ann_exists = os.path.isfile(stb.ANN_DB_PATH)
    if flag in ("0", "false", "off"):
        return stb.create_announcement_features(df, pd.DataFrame())
    if flag in ("1", "true", "on"):
        if not ann_exists:
            return stb.create_announcement_features(df, pd.DataFrame())
        with _chdir_data_root():
            ann_db = pd.read_csv(stb.ANN_DB_PATH, parse_dates=["datetime"])
        return stb.create_announcement_features(df, ann_db)
    if not ann_exists:
        return stb.create_announcement_features(df, pd.DataFrame())
    with _chdir_data_root():
        ann_db = pd.read_csv(stb.ANN_DB_PATH, parse_dates=["datetime"])
    return stb.create_announcement_features(df, ann_db)


def apply_stock_bot_5m_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Same order as `run_live_pipeline` / live cycle in stock_trading_bot.py (5m)."""
    if df.empty:
        return df
    stb = _load_stock_trading_bot()
    with _chdir_data_root():
        df_daily = stb.load_daily_panel_data()
    out = df.copy()
    out = stb.add_technical_indicators(out)
    with _chdir_data_root():
        out = stb.add_pivot_features(out, df_daily)
        out = stb.add_nifty_specific_features(out, df_daily)
        out = stb.merge_macro_features(out)
    out = stb.add_macro_sentiment_features(out)
    out = _announcement_features(stb, out)
    return out


def apply_stock_bot_1h_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same steps as training `run_training_pipeline` for the 1h panel (Steps 1–4 + announcements).
    Matches the richer 1h path used in training rather than the shortened backtest-only snippet.
    """
    if df.empty:
        return df
    stb = _load_stock_trading_bot()
    with _chdir_data_root():
        df_daily = stb.load_daily_panel_data()
    out = df.copy()
    out = stb.add_technical_indicators(out)
    with _chdir_data_root():
        out = stb.add_pivot_features(out, df_daily)
        out = stb.add_nifty_specific_features(out, df_daily)
        out = stb.merge_macro_features(out)
    out = stb.add_macro_sentiment_features(out)
    out = _announcement_features(stb, out)
    return out
