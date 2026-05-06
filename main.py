import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from db.client import get_supabase
from engine.signals import models_ready
from engine.data import fetch_live_candles, get_latest_prices
from engine.paper import run_cycle

# ── Scheduler ───────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler()

async def scheduled_paper_run():
    """Run paper trading cycle for all active bots."""
    if not models_ready():
        print("[Scheduler] Models not ready, skipping cycle.")
        return

    print("[Scheduler] Starting hourly cycle...")
    sb = get_supabase()
    # Get all active bots
    res = sb.table("bot_instances").select("*").eq("status", "running").execute()
    bots = res.data
    if not bots:
        print("[Scheduler] No active bots running.")
        return

    # Collect all unique tickers needed across all bots
    all_tickers = set()
    for b in bots:
        if b.get("tickers"):
            all_tickers.update(b["tickers"])
    
    if not all_tickers:
        return
        
    # Fetch data once for everyone
    data = fetch_live_candles(list(all_tickers))
    df_5m = data["5m"]
    df_1h = data["1h"]
    
    if df_5m.empty:
        print("[Scheduler] Failed to fetch live data. Skipping.")
        return

    # Run cycle for each bot
    for b in bots:
        try:
            log = run_cycle(b["id"], df_5m, df_1h)
            print(f"[Scheduler] Bot {b['id']} cycle complete: {log['cycle_pnl']} PnL")
        except Exception as e:
            print(f"[Scheduler] Error running bot {b['id']}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start scheduler on app startup
    # Run at minute 16 of hours 10, 11, 12, 13, 14, 15 IST
    # Note: APScheduler uses server timezone by default. Ensure server is IST or configure explicitly.
    try:
        from pytz import timezone
        ist = timezone('Asia/Kolkata')
        scheduler.add_job(scheduled_paper_run, 'cron', hour='10-15', minute='16', timezone=ist)
        scheduler.start()
        print("[App] Scheduler started.")
    except Exception as e:
        print(f"[App] Failed to start scheduler: {e}")
        
    yield
    
    # Shutdown
    scheduler.shutdown()
    print("[App] Scheduler stopped.")


# ── API ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Trading Bot API", lifespan=lifespan)

class BotConfig(BaseModel):
    initial_capital: float
    tickers: List[str]
    mode: str = "custom"

@app.get("/")
def health_check():
    return {"status": "ok", "models_ready": models_ready()}

@app.post("/bot/create")
def create_bot(config: BotConfig, user_id: str):
    """Create a new bot instance for a user."""
    sb = get_supabase()
    data = {
        "user_id": user_id,
        "mode": config.mode,
        "initial_capital": config.initial_capital,
        "current_capital": config.initial_capital,
        "tickers": config.tickers,
        "status": "stopped"
    }
    res = sb.table("bot_instances").insert(data).execute()
    return {"message": "Bot created", "bot": res.data[0]}

@app.post("/bot/{bot_id}/start")
def start_bot(bot_id: str):
    sb = get_supabase()
    sb.table("bot_instances").update({"status": "running"}).eq("id", bot_id).execute()
    return {"message": f"Bot {bot_id} started"}

@app.post("/bot/{bot_id}/stop")
def stop_bot(bot_id: str):
    sb = get_supabase()
    sb.table("bot_instances").update({"status": "stopped"}).eq("id", bot_id).execute()
    return {"message": f"Bot {bot_id} stopped"}

@app.get("/bot/{bot_id}/status")
def get_bot_status(bot_id: str):
    sb = get_supabase()
    bot = sb.table("bot_instances").select("*").eq("id", bot_id).single().execute()
    pos = sb.table("positions").select("*").eq("bot_id", bot_id).execute()
    
    # Get latest prices for open positions
    open_tickers = [p["ticker"] for p in pos.data]
    latest_prices = get_latest_prices(open_tickers)
    
    # Enrich positions with live unrealized PnL
    enriched_pos = []
    for p in pos.data:
        t = p["ticker"]
        cur_price = latest_prices.get(t, p["entry_price"])
        unrealized = (cur_price - float(p["entry_price"])) * p["qty"] * p["direction"]
        enriched_pos.append({**p, "current_price": cur_price, "unrealized_pnl": unrealized})
        
    return {
        "bot": bot.data,
        "positions": enriched_pos
    }

@app.get("/bot/{bot_id}/history")
def get_bot_history(bot_id: str):
    sb = get_supabase()
    trades = sb.table("trades").select("*").eq("bot_id", bot_id).order("exit_time", desc=True).execute()
    return {"trades": trades.data}
