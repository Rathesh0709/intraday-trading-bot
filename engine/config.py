# ═══════════════════════════════════════════════════════════════════
# engine/config.py  —  All trading constants in one place
# ═══════════════════════════════════════════════════════════════════

import os
from pathlib import Path

# Reference CSVs used at inference (see scripts/refresh_bot_reference_data.py).
NIFTY_DAILY_CSV = "nifty50_stocks_daily.csv"
MACRO_SENTIMENT_CSV = "macro_sentiment.csv"
ANNOUNCEMENTS_CSV = "nse_announcements_full.csv"


def get_bot_data_dir() -> Path:
    """
    Directory where optional reference CSVs live.

    Set env **BOT_DATA_DIR** to an absolute path on the server (e.g. Railway volume).
    If unset, defaults to the **monorepo root** (parent of `trading-app-backend/`),
    which matches local development when the full repo is checked out.
    """
    raw = os.environ.get("BOT_DATA_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    # engine/config.py → …/trading-app-backend/engine → …/trading-app-backend → monorepo root
    return Path(__file__).resolve().parent.parent.parent


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


INITIAL_CAPITAL      = 5_00_000
MIN_SHARES           = 1
TOP_K_STOCKS         = 10
# Tunable via Railway env if the ensemble rarely clears the bar (e.g. scaler drift).
MIN_CONFIDENCE       = _env_float("MIN_CONFIDENCE", 58.0)
MIN_CONFIDENCE_SHORT = _env_float("MIN_CONFIDENCE_SHORT", 60.0)
MAX_OPEN_POSITIONS   = 10
RISK_PER_TRADE       = 0.02

SL_PCT           = 0.025
TP_PCT           = 0.050
ATR_SL_MULT      = 2.0
ATR_TP_MULT      = 4.0
MAX_SL_PCT       = 0.035
USE_ATR_SIZING   = True

USE_TRAILING_STOP  = True
TRAILING_ATR_MULT  = 1.5

BROKERAGE_PCT    = 0.0003
SLIPPAGE_PCT     = 0.0005
STT_PCT          = 0.00025
ROUND_TRIP_COST  = (BROKERAGE_PCT + SLIPPAGE_PCT) * 2 + STT_PCT

MAX_DAILY_LOSS_PCT       = 0.02
MAX_PORTFOLIO_EXPOSURE   = 0.95

EXIT_HOUR_IST    = 15
EXIT_MIN_IST     = 15
ENTRY_CUTOFF_HOUR = 15
ENTRY_CUTOFF_MIN  = 30

LOOKBACK   = 10
LOOKAHEAD  = 2

try:
    INFERENCE_TAIL_BARS = max(120, int(os.environ.get("INFERENCE_TAIL_BARS", "400")))
except ValueError:
    INFERENCE_TAIL_BARS = 400

NEWS_BLOCK_THRESHOLD   = -0.4
NEWS_CONFIRM_THRESHOLD =  0.4

# All 50 NIFTY tickers
NIFTY_50_TICKERS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS",
    "ITC.NS","LT.NS","SBIN.NS","BHARTIARTL.NS","BAJFINANCE.NS",
    "KOTAKBANK.NS","AXISBANK.NS","ASIANPAINT.NS","HINDUNILVR.NS",
    "TITAN.NS","MARUTI.NS","SUNPHARMA.NS","TATASTEEL.NS","M&M.NS",
    "ULTRACEMCO.NS","POWERGRID.NS","NTPC.NS","WIPRO.NS","NESTLEIND.NS",
    "HCLTECH.NS","BAJAJFINSV.NS","ONGC.NS","TECHM.NS","HINDALCO.NS",
    "ADANIENT.NS","ADANIPORTS.NS","GRASIM.NS","JSWSTEEL.NS","CIPLA.NS",
    "DRREDDY.NS","TATACONSUM.NS","EICHERMOT.NS","APOLLOHOSP.NS",
    "DIVISLAB.NS","BRITANNIA.NS","COALINDIA.NS","HEROMOTOCO.NS",
    "BPCL.NS","BAJAJ-AUTO.NS","HDFCLIFE.NS","PIDILITIND.NS",
    "INDUSINDBK.NS","SBILIFE.NS","LTIM.NS","BEL.NS",
]

# Model files expected in models/ folder.
# Includes both backend aliases and filenames produced by stock_trading_bot.py.
MODEL_DIR = "models"

XGB_MODEL_CANDIDATES = (
    f"{MODEL_DIR}/xgb_model.pkl",
    f"{MODEL_DIR}/nifty_xgb_model.pkl",
)
LGB_MODEL_CANDIDATES = (
    f"{MODEL_DIR}/lgb_model.pkl",
    f"{MODEL_DIR}/nifty_lgb_model.pkl",
)
CAT_MODEL_CANDIDATES = (
    f"{MODEL_DIR}/cat_model.pkl",
    f"{MODEL_DIR}/nifty_cat_model.pkl",
)
TCN_MODEL_CANDIDATES = (
    f"{MODEL_DIR}/tcn_model.keras",
    f"{MODEL_DIR}/nifty_tcn_model.keras",
    f"{MODEL_DIR}/tcn_model.pkl",
)
SCALER_CANDIDATES = (
    f"{MODEL_DIR}/scaler.pkl",
    f"{MODEL_DIR}/nifty_scaler.pkl",
)
FEATURE_COLS_CANDIDATES = (
    f"{MODEL_DIR}/feature_cols.pkl",
    f"{MODEL_DIR}/nifty_features.pkl",
)
