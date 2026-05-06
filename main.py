import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import httpx
from uuid import UUID

from db.client import get_supabase
from engine.signals import models_ready
from engine.data import fetch_live_candles, get_latest_prices
from engine.paper import run_cycle

# ── Scheduler ───────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler()

def _raise_api_error(exc: Exception, context: str) -> None:
    """Map dependency/runtime exceptions to clean API responses."""
    msg = str(exc)
    if isinstance(exc, httpx.ConnectError) or "Network is unreachable" in msg:
        raise HTTPException(
            status_code=503,
            detail=f"{context}: database service unreachable (Supabase/network issue)."
        )
    # Surface Supabase/PostgREST error details as 4xx when possible.
    code = getattr(exc, "code", None)
    details = getattr(exc, "details", None)
    hint = getattr(exc, "hint", None)
    if code or details or hint:
        raise HTTPException(
            status_code=400,
            detail={
                "context": context,
                "message": msg,
                "code": code,
                "details": details,
                "hint": hint,
            },
        )
    raise HTTPException(status_code=500, detail=f"{context}: {msg}")

async def scheduled_paper_run():
    """Run paper trading cycle for all active bots."""
    if not models_ready():
        print("[Scheduler] Models not ready, skipping cycle.")
        return

    print("[Scheduler] Starting hourly cycle...")
    try:
        sb = get_supabase()
        # Get all active bots
        res = sb.table("bot_instances").select("*").eq("status", "running").execute()
        bots = res.data
    except Exception as e:
        print(f"[Scheduler] Supabase unreachable, skipping cycle: {e}")
        return
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

# Allow browser apps (Flutter web, local dev frontends) to call this API.
_cors_origins = os.environ.get("CORS_ORIGINS", "*")
allow_origins = [o.strip() for o in _cors_origins.split(",")] if _cors_origins else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(httpx.ConnectError)
async def httpx_connect_error_handler(request, exc: httpx.ConnectError):
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Database service unreachable from backend. "
                      "Check SUPABASE_URL/SUPABASE_SERVICE_KEY and Railway outbound network."
        },
    )


@app.exception_handler(httpx.HTTPError)
async def httpx_error_handler(request, exc: httpx.HTTPError):
    # Catch other transport-level HTTPX failures from Supabase client usage.
    return JSONResponse(
        status_code=503,
        content={"detail": f"Upstream HTTP dependency failed: {exc}"},
    )

class BotConfig(BaseModel):
    """Bot create payload. user_id can be in body or query param."""
    user_id: Optional[UUID] = None
    initial_capital: float = 500000.0
    tickers: List[str] = []
    mode: str = "custom"

@app.get("/")
def health_check():
    return {"status": "ok", "models_ready": models_ready()}

@app.get("/debug")
def debug_env():
    """Check that environment variables are set correctly (safe — only shows partial values)."""
    url = os.environ.get("SUPABASE_URL", "NOT SET")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "NOT SET")
    return {
        "SUPABASE_URL":         url if url == "NOT SET" else url[:30] + "...",
        "SUPABASE_SERVICE_KEY": key if key == "NOT SET" else key[:10] + "...",
        "url_starts_with_https": url.startswith("https://"),
        "url_ends_with_supabase_co": url.endswith(".supabase.co"),
    }

@app.post("/bot/create")
def create_bot(config: BotConfig, user_id: Optional[UUID] = Query(default=None)):
    """Create a new bot instance for a user."""
    try:
        resolved_user_id = user_id or config.user_id
        if not resolved_user_id:
            raise HTTPException(status_code=422, detail="user_id is required in query or body.")
        sb = get_supabase()
        data = {
            "user_id": str(resolved_user_id),
            "mode": config.mode,
            "initial_capital": config.initial_capital,
            "current_capital": config.initial_capital,
            "tickers": config.tickers,
            "status": "stopped"
        }
        res = sb.table("bot_instances").insert(data).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Bot create failed: empty DB response.")
        return {"message": "Bot created", "bot": res.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        _raise_api_error(e, "Create bot failed")

@app.post("/bot/{bot_id}/start")
def start_bot(bot_id: str):
    try:
        sb = get_supabase()
        sb.table("bot_instances").update({"status": "running"}).eq("id", bot_id).execute()
        return {"message": f"Bot {bot_id} started"}
    except Exception as e:
        _raise_api_error(e, "Start bot failed")

@app.post("/bot/{bot_id}/stop")
def stop_bot(bot_id: str):
    try:
        sb = get_supabase()
        sb.table("bot_instances").update({"status": "stopped"}).eq("id", bot_id).execute()
        return {"message": f"Bot {bot_id} stopped"}
    except Exception as e:
        _raise_api_error(e, "Stop bot failed")

@app.get("/bot/{bot_id}/status")
def get_bot_status(bot_id: str):
    try:
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
    except Exception as e:
        _raise_api_error(e, "Get bot status failed")

@app.get("/bot/{bot_id}/history")
def get_bot_history(bot_id: str):
    try:
        sb = get_supabase()
        trades = sb.table("trades").select("*").eq("bot_id", bot_id).order("exit_time", desc=True).execute()
        return {"trades": trades.data}
    except Exception as e:
        _raise_api_error(e, "Get bot history failed")
