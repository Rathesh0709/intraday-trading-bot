# Intraday Trading Bot

A FastAPI service that runs an **automated intraday paper-trading bot for the NIFTY 50**.
It pulls live market data, generates buy/sell signals from a machine-learning ensemble
enriched with news & macro sentiment, and simulates trades on a virtual portfolio during
NSE market hours.

> ⚠️ **Paper trading / educational use only.** This bot does not place real orders and is
> not financial advice.

## How it works

1. **Market data** — live candles and latest prices for the NIFTY 50 tickers are fetched
   via `yfinance` (`engine/data.py`).
2. **Feature engineering** — technical indicators (via the `ta` library) plus live news
   and macro sentiment features (`engine/features_live.py`, `engine/live_news.py`).
3. **Signal generation** — a trained ensemble of models scores each stock
   (`engine/signals.py`), using the pickled models in `models/`:
   - Gradient boosting: **XGBoost**, **LightGBM**, **CatBoost**
   - A **TCN** (Temporal Convolutional Network, Keras) model
   - Feature-column definitions and `StandardScaler`s (NIFTY-specific variants included)
4. **Paper trading engine** — `engine/paper.py` runs trade cycles against a virtual
   portfolio seeded with `INITIAL_CAPITAL` (`engine/config.py`).
5. **Scheduling** — `APScheduler` triggers trade cycles only on NSE trading days; a
   hardcoded NSE holiday calendar for 2025–2026 is built into `main.py`.
6. **Persistence** — trade state and history are stored in **Supabase** (`db/client.py`).

## Tech stack

- **API:** FastAPI + Uvicorn
- **ML:** scikit-learn, XGBoost, LightGBM, CatBoost, Keras (TCN), joblib
- **Data:** yfinance, `ta`, pandas, numpy
- **News/sentiment:** feedparser, BeautifulSoup4
- **Database:** Supabase
- **Scheduling:** APScheduler
- **Deploy:** `Procfile` (Railway / Heroku-style)

## Project structure

```
main.py                     # FastAPI app, scheduler, NSE market calendar
engine/
  config.py                 # NIFTY 50 tickers, initial capital
  data.py                   # live candle / price fetching
  features_live.py          # live feature engineering
  inference_features.py     # features used at inference time
  live_news.py              # news ingestion
  signals.py                # ensemble model loading + signal generation
  paper.py                  # paper-trading cycle
  bot_training_mirror.py    # training-time feature mirror
db/client.py                # Supabase client
models/                     # trained models + scalers (xgb/lgb/cat/tcn, *.pkl/.keras)
scripts/refresh_bot_reference_data.py
*.csv                       # macro_sentiment, news_sentiment, nifty50 daily reference data
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in Supabase + any API keys
uvicorn main:app --reload   # run locally
```

Provide the environment variables referenced in `.env.example` (Supabase URL/key, etc.)
before starting. On a host like Railway, the `Procfile` defines the start command.
