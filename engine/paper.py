"""
engine/paper.py — Supabase-backed paper trading engine.

One instance of PaperEngine per bot_id.
All state (positions, capital, trades) lives in Supabase, not on disk.
"""

from __future__ import annotations
import math
from datetime import datetime
from typing import Optional
import pandas as pd
import pytz

from db.client import get_supabase
from engine.config import (
    INITIAL_CAPITAL, MIN_SHARES, MAX_OPEN_POSITIONS, RISK_PER_TRADE,
    SL_PCT, TP_PCT, ATR_SL_MULT, ATR_TP_MULT, MAX_SL_PCT,
    USE_TRAILING_STOP, TRAILING_ATR_MULT, ROUND_TRIP_COST,
    MAX_DAILY_LOSS_PCT, EXIT_HOUR_IST, EXIT_MIN_IST, MAX_PORTFOLIO_EXPOSURE,
    ENTRY_CUTOFF_HOUR, ENTRY_CUTOFF_MIN,
)
from engine.signals import generate_signals

IST = pytz.timezone("Asia/Kolkata")


# ── Supabase helpers ─────────────────────────────────────────────────────────

def _get_bot(bot_id: str) -> dict:
    sb = get_supabase()
    r = sb.table("bot_instances").select("*").eq("id", bot_id).single().execute()
    return r.data


def _update_capital(bot_id: str, new_capital: float):
    get_supabase().table("bot_instances").update(
        {"current_capital": round(new_capital, 2)}
    ).eq("id", bot_id).execute()


def _get_positions(bot_id: str) -> list[dict]:
    r = get_supabase().table("positions").select("*").eq("bot_id", bot_id).execute()
    return r.data or []


def _upsert_position(bot_id: str, pos: dict):
    """Insert or update a position row."""
    sb = get_supabase()
    existing = sb.table("positions").select("id")\
        .eq("bot_id", bot_id).eq("ticker", pos["ticker"]).execute()
    if existing.data:
        sb.table("positions").update(pos).eq("id", existing.data[0]["id"]).execute()
    else:
        sb.table("positions").insert({**pos, "bot_id": bot_id}).execute()


def _delete_position(bot_id: str, ticker: str):
    get_supabase().table("positions").delete()\
        .eq("bot_id", bot_id).eq("ticker", ticker).execute()


def _insert_trade(bot_id: str, trade: dict):
    get_supabase().table("trades").insert({**trade, "bot_id": bot_id}).execute()


def _get_today_trades(bot_id: str, today_str: str) -> list[dict]:
    r = get_supabase().table("trades").select("pnl")\
        .eq("bot_id", bot_id).gte("exit_time", today_str).execute()
    return r.data or []


def _close_position_atomic(
    bot_id: str,
    pos: dict,
    exit_price: float,
    reason: str,
    now_iso: str,
    pnl: float,
) -> bool:
    """
    Prefer DB-side atomic close via RPC.
    Falls back to client-side multi-step write for backward compatibility.
    """
    sb = get_supabase()
    try:
        sb.rpc("close_position_atomic", {
            "p_bot_id": bot_id,
            "p_ticker": pos["ticker"],
            "p_entry_price": float(pos["entry_price"]),
            "p_exit_price": float(exit_price),
            "p_qty": int(pos["qty"]),
            "p_direction": int(pos["direction"]),
            "p_reason": str(reason),
            "p_exit_time": now_iso,
            "p_pnl": float(round(pnl, 2)),
        }).execute()
        return True
    except Exception:
        _insert_trade(bot_id, {
            "ticker": pos["ticker"],
            "direction": "BUY" if pos["direction"] == 1 else "SELL",
            "entry_price": pos["entry_price"],
            "exit_price": round(exit_price, 2),
            "qty": pos["qty"],
            "pnl": round(pnl, 2),
            "reason": reason,
            "exit_time": now_iso,
        })
        _delete_position(bot_id, pos["ticker"])
        return False


# ── Main cycle ────────────────────────────────────────────────────────────────

def run_cycle(bot_id: str, df_5m: pd.DataFrame, df_1h: pd.DataFrame) -> dict:
    """
    Run one hourly paper trading cycle for the given bot.
    Returns a summary dict of what happened this cycle.
    """
    bot          = _get_bot(bot_id)
    capital      = float(bot["current_capital"])
    starting_cap = float(bot["initial_capital"])
    tickers      = bot.get("tickers") or []

    now_ist   = datetime.now(IST)
    now_hour  = now_ist.hour
    now_min   = now_ist.minute
    today_str = now_ist.strftime("%Y-%m-%d")
    now_str   = now_ist.strftime("%H:%M")

    cycle_log   = {
        "time": now_str,
        "closed": [],
        "new_entries": [],
        "held": [],
        "cycle_pnl": 0.0,
        "capital": capital,
    }

    positions = _get_positions(bot_id)
    pos_map   = {p["ticker"]: p for p in positions}

    # ── EOD force-close (15:15+) ─────────────────────────────────────────────
    is_eod = now_hour > EXIT_HOUR_IST or (
        now_hour == EXIT_HOUR_IST and now_min >= EXIT_MIN_IST
    )
    if is_eod and pos_map:
        for ticker, pos in pos_map.items():
            t5 = df_5m[df_5m["ticker"] == ticker].sort_index()
            if t5.empty:
                continue
            cur_price = float(t5["close"].iloc[-1])
            cost = pos["entry_price"] * pos["qty"] * ROUND_TRIP_COST
            pnl  = ((cur_price - pos["entry_price"]) * pos["qty"] * pos["direction"]) - cost
            capital += pnl
            _close_position_atomic(
                bot_id=bot_id,
                pos=pos,
                exit_price=cur_price,
                reason="EOD",
                now_iso=now_ist.isoformat(),
                pnl=pnl,
            )
            cycle_log["closed"].append({
                "ticker": ticker, "reason": "EOD",
                "pnl": round(pnl, 2), "exit_price": round(cur_price, 2),
            })
            cycle_log["cycle_pnl"] += pnl

        _update_capital(bot_id, capital)
        cycle_log["capital"] = capital
        return cycle_log

    # ── Check SL / TP ────────────────────────────────────────────────────────
    to_close = []
    for ticker, pos in pos_map.items():
        t5 = df_5m[df_5m["ticker"] == ticker].sort_index()
        if t5.empty:
            continue
        cur_price = float(t5["close"].iloc[-1])
        direction = pos["direction"]
        sl, tp    = pos["sl"], pos["tp"]

        # Trailing SL
        if USE_TRAILING_STOP:
            atr = pos.get("atr", pos["entry_price"] * MAX_SL_PCT / ATR_SL_MULT)
            if direction == 1 and cur_price > pos.get("highest_since_entry", pos["entry_price"]):
                pos["highest_since_entry"] = cur_price
                new_sl = cur_price - atr * TRAILING_ATR_MULT
                if new_sl > sl:
                    sl = new_sl
                    pos["sl"] = round(sl, 2)
                    _upsert_position(bot_id, pos)
            elif direction == -1 and cur_price < pos.get("lowest_since_entry", pos["entry_price"]):
                pos["lowest_since_entry"] = cur_price
                new_sl = cur_price + atr * TRAILING_ATR_MULT
                if new_sl < sl:
                    sl = new_sl
                    pos["sl"] = round(sl, 2)
                    _upsert_position(bot_id, pos)

        hit_sl = (direction == 1 and cur_price <= sl) or (direction == -1 and cur_price >= sl)
        hit_tp = (direction == 1 and cur_price >= tp) or (direction == -1 and cur_price <= tp)

        if hit_sl or hit_tp:
            exit_price = sl if hit_sl else tp
            reason     = "SL" if hit_sl else "TP"
            cost  = pos["entry_price"] * pos["qty"] * ROUND_TRIP_COST
            pnl   = ((exit_price - pos["entry_price"]) * pos["qty"] * direction) - cost
            capital += pnl
            _close_position_atomic(
                bot_id=bot_id,
                pos=pos,
                exit_price=exit_price,
                reason=reason,
                now_iso=now_ist.isoformat(),
                pnl=pnl,
            )
            to_close.append(ticker)
            cycle_log["closed"].append({
                "ticker": ticker, "reason": reason,
                "pnl": round(pnl, 2), "exit_price": round(exit_price, 2),
            })
            cycle_log["cycle_pnl"] += pnl

    for ticker in to_close:
        _delete_position(bot_id, ticker)
        del pos_map[ticker]

    _update_capital(bot_id, capital)

    # ── Daily drawdown circuit breaker ───────────────────────────────────────
    today_trades = _get_today_trades(bot_id, today_str)
    realized_today = sum(float(t["pnl"]) for t in today_trades)
    if realized_today / starting_cap <= -MAX_DAILY_LOSS_PCT:
        cycle_log["circuit_breaker"] = True
        cycle_log["capital"] = capital
        return cycle_log

    # ── Entry cutoff ─────────────────────────────────────────────────────────
    if now_hour > ENTRY_CUTOFF_HOUR or (
        now_hour == ENTRY_CUTOFF_HOUR and now_min >= ENTRY_CUTOFF_MIN
    ):
        cycle_log["held"] = [p["ticker"] for p in pos_map.values()]
        cycle_log["capital"] = capital
        return cycle_log

    # ── Generate signals ─────────────────────────────────────────────────────
    picks, signal_meta = generate_signals(df_5m, tickers)
    cycle_log["signals"] = signal_meta
    if picks.empty:
        cycle_log["held"] = [p["ticker"] for p in pos_map.values()]
        cycle_log["capital"] = capital
        return cycle_log

    # ── Open new positions ───────────────────────────────────────────────────
    slots    = MAX_OPEN_POSITIONS - len(pos_map)
    new_rows = [r for _, r in picks.iterrows() if r["ticker"] not in pos_map]
    n_new    = min(len(new_rows), slots)
    if n_new == 0:
        cycle_log["held"] = [p["ticker"] for p in pos_map.values()]
        cycle_log["capital"] = capital
        return cycle_log

    capital_in_use    = sum(p["entry_price"] * p["qty"] for p in pos_map.values())
    max_exposure_cap  = max(0.0, capital * MAX_PORTFOLIO_EXPOSURE)
    available_capital = max(0.0, capital - capital_in_use)
    per_pos_cap       = available_capital / n_new
    opened            = 0

    for _, row in picks.iterrows():
        if opened >= slots:
            break
        ticker = row["ticker"]
        if ticker in pos_map:
            continue

        entry_price = float(row["close"])
        sig = 1 if row["signal"] == "BUY" else -1

        # ATR-based SL/TP
        sl_pct = SL_PCT
        tp_pct = TP_PCT
        atr    = None
        if not df_1h.empty:
            t1h = df_1h[df_1h["ticker"] == ticker].sort_index()
            if not t1h.empty and "atr" in t1h.columns:
                atr_val = float(t1h["atr"].iloc[-1])
                if atr_val > 0:
                    sl_pct = min(atr_val * ATR_SL_MULT / entry_price, MAX_SL_PCT)
                    tp_pct = atr_val * ATR_TP_MULT / entry_price
                    atr    = atr_val

        risk_amount    = capital * RISK_PER_TRADE
        risk_per_share = entry_price * sl_pct
        qty_by_risk    = max(MIN_SHARES, int(risk_amount / (risk_per_share + 1e-9)))
        qty_by_cap_raw = int(per_pos_cap / entry_price)
        if qty_by_cap_raw <= 0:
            continue
        qty_by_cap     = qty_by_cap_raw
        qty            = min(qty_by_risk, qty_by_cap)
        if qty < MIN_SHARES:
            continue

        invested = entry_price * qty
        if (capital_in_use + invested) > max_exposure_cap:
            continue

        sl = entry_price * (1 - sl_pct) if sig == 1 else entry_price * (1 + sl_pct)
        tp = entry_price * (1 + tp_pct) if sig == 1 else entry_price * (1 - tp_pct)

        pos_row = {
            "ticker":               ticker,
            "entry_price":          entry_price,
            "qty":                  qty,
            "direction":            sig,
            "sl":                   round(sl, 2),
            "tp":                   round(tp, 2),
            "atr":                  atr if atr else entry_price * sl_pct / ATR_SL_MULT,
            "highest_since_entry":  entry_price,
            "lowest_since_entry":   entry_price,
            "confidence":           float(row["confidence"]),
            "entry_time":           now_ist.isoformat(),
        }
        _upsert_position(bot_id, pos_row)
        capital_in_use += invested
        opened += 1

        cycle_log["new_entries"].append({
            "ticker":    ticker,
            "direction": "BUY" if sig == 1 else "SELL",
            "qty":       qty,
            "entry":     entry_price,
            "invested":  round(entry_price * qty, 2),
            "sl":        round(sl, 2),
            "tp":        round(tp, 2),
            "conf":      float(row["confidence"]),
        })

    cycle_log["held"]    = [p["ticker"] for p in pos_map.values()]
    cycle_log["capital"] = capital
    return cycle_log
