import os
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import httpx
from uuid import UUID
import uuid

from db.client import get_supabase
from engine.signals import models_ready
from engine.data import fetch_live_candles, get_latest_prices
from engine.paper import run_cycle
from engine.config import NIFTY_50_TICKERS, INITIAL_CAPITAL

# ── Scheduler ───────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler()
GENERAL_BOT_OWNER_ID = "00000000-0000-0000-0000-000000000001"
GENERAL_MODE = "general"
SCHEDULER_STATE = {
    "last_run_id": None,
    "last_started_at": None,
    "last_completed_at": None,
    "last_duration_ms": None,
    "last_status": "never_run",
    "last_error": None,
    "last_bots_processed": 0,
}


def _safe_insert_cycle_log(bot_id: str, payload: dict) -> None:
    """
    Persist cycle logs when bot_cycle_logs table exists.
    Non-blocking by design: logging failures should never break trading flow.
    """
    try:
        get_supabase().table("bot_cycle_logs").insert({
            "bot_id": bot_id,
            "payload": payload,
        }).execute()
    except Exception as e:
        print(f"[Scheduler] bot_cycle_logs insert skipped: {e}")


def _safe_insert_cycle_metric(payload: dict) -> None:
    """
    Persist per-cycle execution metrics when table exists.
    Non-blocking: metrics must not break scheduling.
    """
    try:
        get_supabase().table("bot_cycle_metrics").insert(payload).execute()
    except Exception as e:
        print(f"[Scheduler] bot_cycle_metrics insert skipped: {e}")


def _get_general_bot() -> Optional[dict]:
    try:
        r = get_supabase().table("bot_instances") \
            .select("*") \
            .eq("mode", GENERAL_MODE) \
            .order("created_at") \
            .limit(1) \
            .execute()
        rows = r.data or []
        return rows[0] if rows else None
    except Exception:
        return None


def _get_bot_row(bot_id: str) -> Optional[dict]:
    try:
        r = get_supabase().table("bot_instances") \
            .select("*") \
            .eq("id", bot_id) \
            .limit(1) \
            .execute()
        rows = r.data or []
        return rows[0] if rows else None
    except Exception:
        return None


def _require_bot_access(bot_id: str, user_id: Optional[UUID]) -> dict:
    """
    Authorize bot access:
    - General bot is public/readable and controllable by design.
    - Custom bot requires matching user_id.
    """
    bot = _get_bot_row(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found.")
    if bot.get("mode") == GENERAL_MODE:
        return bot
    if user_id is None:
        raise HTTPException(status_code=401, detail="user_id required for custom bot access.")
    if str(bot.get("user_id")) != str(user_id):
        raise HTTPException(status_code=403, detail="You do not own this bot.")
    return bot


def _ensure_general_bot(now_ist: datetime) -> Optional[dict]:
    """
    Ensure singleton general bot exists and is auto-started/stopped by time.
    Trading window: 09:00–16:00 IST.
    """
    sb = get_supabase()
    bot = _get_general_bot()
    should_run = 9 <= now_ist.hour < 16

    if bot is None:
        payload = {
            "user_id": GENERAL_BOT_OWNER_ID,
            "mode": GENERAL_MODE,
            "initial_capital": float(INITIAL_CAPITAL),
            "current_capital": float(INITIAL_CAPITAL),
            "tickers": NIFTY_50_TICKERS,
            "status": "running" if should_run else "stopped",
        }
        res = sb.table("bot_instances").insert(payload).execute()
        rows = res.data or []
        bot = rows[0] if rows else None
        if bot:
            print(f"[General Bot] Created {bot['id']} with status={bot['status']}")
        return bot

    desired = "running" if should_run else "stopped"
    if bot.get("status") != desired:
        sb.table("bot_instances").update({"status": desired}).eq("id", bot["id"]).execute()
        bot["status"] = desired
        print(f"[General Bot] Auto-set status={desired}")
    return bot

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

    run_id = str(uuid.uuid4())
    run_started_at = datetime.utcnow()
    SCHEDULER_STATE["last_run_id"] = run_id
    SCHEDULER_STATE["last_started_at"] = run_started_at.isoformat() + "Z"
    SCHEDULER_STATE["last_status"] = "running"
    SCHEDULER_STATE["last_error"] = None
    SCHEDULER_STATE["last_bots_processed"] = 0
    print(f"[Scheduler] Starting scheduled cycle run_id={run_id}")
    try:
        now_ist = datetime.now(__import__("pytz").timezone("Asia/Kolkata"))
        _ensure_general_bot(now_ist)
        sb = get_supabase()
        # Get all active bots
        res = sb.table("bot_instances").select("*").eq("status", "running").execute()
        bots = res.data
    except Exception as e:
        SCHEDULER_STATE["last_status"] = "error"
        SCHEDULER_STATE["last_error"] = str(e)
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
        SCHEDULER_STATE["last_status"] = "data_unavailable"
        print("[Scheduler] Failed to fetch live data. Skipping.")
        return

    # Run cycle for each bot
    bots_processed = 0
    for b in bots:
        cycle_started_at = datetime.utcnow()
        try:
            log = run_cycle(b["id"], df_5m, df_1h)
            bots_processed += 1
            cycle_completed_at = datetime.utcnow()
            execution_ms = int((cycle_completed_at - cycle_started_at).total_seconds() * 1000)
            structured = {
                "event": "cycle_complete",
                "bot_id": b["id"],
                "mode": b.get("mode"),
                "status": "ok",
                "started_at": cycle_started_at.isoformat() + "Z",
                "completed_at": cycle_completed_at.isoformat() + "Z",
                "execution_ms": execution_ms,
                "cycle": log,
            }
            print(f"[Scheduler] {structured}")
            _safe_insert_cycle_log(b["id"], structured)
            _safe_insert_cycle_metric({
                "run_id": run_id,
                "bot_id": b["id"],
                "mode": b.get("mode"),
                "status": "ok",
                "execution_ms": execution_ms,
                "started_at": cycle_started_at.isoformat() + "Z",
                "completed_at": cycle_completed_at.isoformat() + "Z",
                "cycle_pnl": (log or {}).get("cycle_pnl", 0.0),
            })
        except Exception as e:
            cycle_completed_at = datetime.utcnow()
            execution_ms = int((cycle_completed_at - cycle_started_at).total_seconds() * 1000)
            structured = {
                "event": "cycle_complete",
                "bot_id": b["id"],
                "mode": b.get("mode"),
                "status": "error",
                "started_at": cycle_started_at.isoformat() + "Z",
                "completed_at": cycle_completed_at.isoformat() + "Z",
                "execution_ms": execution_ms,
                "error_message": str(e),
            }
            print(f"[Scheduler] {structured}")
            _safe_insert_cycle_log(b["id"], structured)
            _safe_insert_cycle_metric({
                "run_id": run_id,
                "bot_id": b["id"],
                "mode": b.get("mode"),
                "status": "error",
                "execution_ms": execution_ms,
                "started_at": cycle_started_at.isoformat() + "Z",
                "completed_at": cycle_completed_at.isoformat() + "Z",
                "error_message": str(e),
            })

    run_completed_at = datetime.utcnow()
    SCHEDULER_STATE["last_completed_at"] = run_completed_at.isoformat() + "Z"
    SCHEDULER_STATE["last_duration_ms"] = int((run_completed_at - run_started_at).total_seconds() * 1000)
    SCHEDULER_STATE["last_status"] = "ok"
    SCHEDULER_STATE["last_bots_processed"] = bots_processed

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start scheduler on app startup
    # Run at minute 16 of hours 10, 11, 12, 13, 14, 15 IST
    # Note: APScheduler uses server timezone by default. Ensure server is IST or configure explicitly.
    try:
        from pytz import timezone
        ist = timezone('Asia/Kolkata')
        scheduler.add_job(scheduled_paper_run, 'cron', hour='9-16', minute='*/5', timezone=ist)
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
            "mode": "custom",
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
def start_bot(bot_id: str, user_id: Optional[UUID] = Query(default=None)):
    try:
        _require_bot_access(bot_id, user_id)
        sb = get_supabase()
        sb.table("bot_instances").update({"status": "running"}).eq("id", bot_id).execute()
        return {"message": f"Bot {bot_id} started"}
    except Exception as e:
        _raise_api_error(e, "Start bot failed")

@app.post("/bot/{bot_id}/stop")
def stop_bot(bot_id: str, user_id: Optional[UUID] = Query(default=None)):
    try:
        _require_bot_access(bot_id, user_id)
        sb = get_supabase()
        sb.table("bot_instances").update({"status": "stopped"}).eq("id", bot_id).execute()
        return {"message": f"Bot {bot_id} stopped"}
    except Exception as e:
        _raise_api_error(e, "Stop bot failed")

@app.get("/bot/{bot_id}/status")
def get_bot_status(bot_id: str, user_id: Optional[UUID] = Query(default=None)):
    try:
        _require_bot_access(bot_id, user_id)
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
def get_bot_history(
    bot_id: str,
    user_id: Optional[UUID] = Query(default=None),
    limit: int = 100,
    offset: int = 0,
    ticker: Optional[str] = Query(default=None),
    from_date: Optional[str] = Query(default=None),
    to_date: Optional[str] = Query(default=None),
):
    try:
        _require_bot_access(bot_id, user_id)
        sb = get_supabase()
        safe_limit = max(1, min(limit, 500))
        safe_offset = max(0, offset)
        q = sb.table("trades").select("*").eq("bot_id", bot_id)
        if ticker:
            q = q.eq("ticker", ticker)
        if from_date:
            q = q.gte("exit_time", from_date)
        if to_date:
            q = q.lte("exit_time", to_date)
        trades = q.order("exit_time", desc=True) \
            .range(safe_offset, safe_offset + safe_limit - 1).execute()
        return {"trades": trades.data or [], "limit": safe_limit, "offset": safe_offset}
    except Exception as e:
        _raise_api_error(e, "Get bot history failed")


@app.get("/bot/{bot_id}/dashboard")
def get_bot_dashboard(bot_id: str, user_id: Optional[UUID] = Query(default=None)):
    """
    Dashboard data for frontend cards:
    - current positions with live unrealized PnL
    - today's realized PnL and trade count
    - cumulative realized PnL
    """
    try:
        _require_bot_access(bot_id, user_id)
        sb = get_supabase()
        bot = sb.table("bot_instances").select("*").eq("id", bot_id).single().execute()
        positions = sb.table("positions").select("*").eq("bot_id", bot_id).execute()
        trades = sb.table("trades").select("*").eq("bot_id", bot_id).execute()

        open_positions = positions.data or []
        all_trades = trades.data or []

        # Compute today's realized PnL (UTC date for consistency with DB timestamps)
        today = datetime.utcnow().date().isoformat()
        today_trades = [
            t for t in all_trades
            if str(t.get("exit_time", "")).startswith(today)
        ]
        realized_today = sum(float(t.get("pnl", 0) or 0) for t in today_trades)
        realized_total = sum(float(t.get("pnl", 0) or 0) for t in all_trades)

        # Live prices for open positions
        tickers = [p["ticker"] for p in open_positions]
        latest_prices = get_latest_prices(tickers)
        enriched_positions = []
        for p in open_positions:
            cur = latest_prices.get(p["ticker"], float(p["entry_price"]))
            unrealized = (cur - float(p["entry_price"])) * p["qty"] * p["direction"]
            enriched_positions.append({
                **p,
                "current_price": cur,
                "unrealized_pnl": round(unrealized, 2),
            })

        return {
            "bot": bot.data,
            "positions": enriched_positions,
            "stats": {
                "open_positions": len(enriched_positions),
                "trades_today": len(today_trades),
                "realized_pnl_today": round(realized_today, 2),
                "realized_pnl_total": round(realized_total, 2),
                "unrealized_pnl_open": round(
                    sum(float(p["unrealized_pnl"]) for p in enriched_positions), 2
                ),
                "win_rate_today": round(
                    (sum(1 for t in today_trades if float(t.get("pnl", 0) or 0) > 0) / len(today_trades) * 100)
                    if today_trades else 0.0, 2
                ),
                "win_rate_total": round(
                    (sum(1 for t in all_trades if float(t.get("pnl", 0) or 0) > 0) / len(all_trades) * 100)
                    if all_trades else 0.0, 2
                ),
            },
        }
    except Exception as e:
        _raise_api_error(e, "Get bot dashboard failed")


@app.get("/bot/{bot_id}/logs")
def get_bot_logs(
    bot_id: str,
    limit: int = 30,
    offset: int = 0,
    user_id: Optional[UUID] = Query(default=None),
    from_date: Optional[str] = Query(default=None),
    to_date: Optional[str] = Query(default=None),
):
    """
    Read per-cycle bot logs if bot_cycle_logs table exists.
    """
    try:
        _require_bot_access(bot_id, user_id)
        sb = get_supabase()
        safe_limit = max(1, min(limit, 200))
        safe_offset = max(0, offset)
        q = sb.table("bot_cycle_logs").select("*").eq("bot_id", bot_id)
        if from_date:
            q = q.gte("created_at", from_date)
        if to_date:
            q = q.lte("created_at", to_date)
        rows = q.order("created_at", desc=True) \
            .range(safe_offset, safe_offset + safe_limit - 1) \
            .execute()
        return {"logs": rows.data or [], "limit": safe_limit, "offset": safe_offset}
    except Exception as e:
        # Keep this non-fatal for clients that haven't created bot_cycle_logs yet.
        return {"logs": [], "warning": f"bot_cycle_logs unavailable: {e}"}


@app.get("/bots")
def list_bots(user_id: UUID):
    """
    Return custom bots for the user + singleton general bot.
    """
    try:
        sb = get_supabase()
        custom = sb.table("bot_instances") \
            .select("*") \
            .eq("user_id", str(user_id)) \
            .neq("mode", GENERAL_MODE) \
            .order("created_at", desc=True) \
            .execute()
        general = _get_general_bot()
        return {
            "general_bot": general,
            "custom_bots": custom.data or [],
        }
    except Exception as e:
        _raise_api_error(e, "List bots failed")


@app.get("/ops/scheduler-health")
def scheduler_health():
    """
    Operational health for scheduler and last run telemetry.
    """
    return {
        "scheduler_running": scheduler.running,
        "state": SCHEDULER_STATE,
    }


@app.get("/ops/cycle-metrics")
def cycle_metrics(limit: int = 100, offset: int = 0):
    """
    Fetch recent per-bot cycle execution metrics (if bot_cycle_metrics table exists).
    """
    try:
        safe_limit = max(1, min(limit, 500))
        safe_offset = max(0, offset)
        rows = get_supabase().table("bot_cycle_metrics") \
            .select("*") \
            .order("completed_at", desc=True) \
            .range(safe_offset, safe_offset + safe_limit - 1) \
            .execute()
        return {"rows": rows.data or [], "limit": safe_limit, "offset": safe_offset}
    except Exception as e:
        return {"rows": [], "warning": f"bot_cycle_metrics unavailable: {e}"}


def _build_dashboard(bot_id: str) -> dict:
    sb = get_supabase()
    bot = sb.table("bot_instances").select("*").eq("id", bot_id).single().execute()
    positions = sb.table("positions").select("*").eq("bot_id", bot_id).execute()
    trades = sb.table("trades").select("*").eq("bot_id", bot_id).execute()

    open_positions = positions.data or []
    all_trades = trades.data or []
    today = datetime.utcnow().date().isoformat()
    today_trades = [t for t in all_trades if str(t.get("exit_time", "")).startswith(today)]
    realized_today = sum(float(t.get("pnl", 0) or 0) for t in today_trades)
    realized_total = sum(float(t.get("pnl", 0) or 0) for t in all_trades)

    tickers = [p["ticker"] for p in open_positions]
    latest_prices = get_latest_prices(tickers)
    enriched_positions = []
    for p in open_positions:
        cur = latest_prices.get(p["ticker"], float(p["entry_price"]))
        unrealized = (cur - float(p["entry_price"])) * p["qty"] * p["direction"]
        enriched_positions.append({
            **p,
            "current_price": cur,
            "unrealized_pnl": round(unrealized, 2),
        })

    return {
        "bot": bot.data,
        "positions": enriched_positions,
        "stats": {
            "open_positions": len(enriched_positions),
            "trades_today": len(today_trades),
            "realized_pnl_today": round(realized_today, 2),
            "realized_pnl_total": round(realized_total, 2),
            "unrealized_pnl_open": round(
                sum(float(p["unrealized_pnl"]) for p in enriched_positions), 2
            ),
            "win_rate_today": round(
                (sum(1 for t in today_trades if float(t.get("pnl", 0) or 0) > 0) / len(today_trades) * 100)
                if today_trades else 0.0, 2
            ),
            "win_rate_total": round(
                (sum(1 for t in all_trades if float(t.get("pnl", 0) or 0) > 0) / len(all_trades) * 100)
                if all_trades else 0.0, 2
            ),
        },
    }


@app.get("/bot/general/summary")
def get_general_bot_summary(limit: int = 30):
    """
    Public-like overview endpoint for the auto-managed General Bot.
    """
    try:
        bot = _get_general_bot()
        if not bot:
            return {"message": "General bot not initialized yet.", "bot": None, "logs": []}
        dashboard = _build_dashboard(bot["id"])
        logs = get_bot_logs(bot["id"], limit=limit)
        return {
            "bot_id": bot["id"],
            "dashboard": dashboard,
            "logs": logs.get("logs", []),
            "logs_warning": logs.get("warning"),
        }
    except Exception as e:
        _raise_api_error(e, "Get general bot summary failed")
