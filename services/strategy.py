from __future__ import annotations
import datetime as dt
from dataclasses import asdict
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from strategy_model import (
    FXPlanConfigV2,
    simulate_walk_forward_fx_plan_v2,
    _detect_regime_shift,
    _pick_mode_and_params,
    _compute_signals_online,
    _schedule_ratio,
    _cheap_score,
)

def cfg_from_form(form: Dict[str, Any]) -> FXPlanConfigV2:
    """Create FXPlanConfigV2 from a Flask form dict (strings). Missing => default."""
    cfg = FXPlanConfigV2()
    for k, v in form.items():
        if not hasattr(cfg, k):
            continue
        if v is None or str(v).strip() == "":
            continue
        cur = getattr(cfg, k)
        try:
            if isinstance(cur, bool):
                vv = str(v).lower() in ("1","true","yes","y","on")
            elif isinstance(cur, int):
                vv = int(float(v))
            else:
                vv = float(v)
            setattr(cfg, k, vv)
        except Exception:
            # ignore bad inputs; UI should validate but keep app running
            pass
    return cfg

def run_backtest(df_wide: pd.DataFrame, bank: str, n_test: int, cfg: FXPlanConfigV2) -> Dict[str, Any]:
    """df_wide columns: date, <bank>."""
    df = df_wide[["date", bank]].copy()
    df.columns = ["date", "rate"]
    df["date"] = pd.to_datetime(df["date"])
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna().sort_values("date")
    out = simulate_walk_forward_fx_plan_v2(df_rates=df, rate_col="rate", N_test=int(n_test), cfg=cfg, date_col="date")
    return out

def _compute_regime_state_sequential(rates: np.ndarray, dates: np.ndarray, cfg: FXPlanConfigV2) -> Tuple[int, Optional[dt.date]]:
    """Replicate regime_start_idx + cooldown across the whole sequence."""
    regime_start_idx = 0
    last_shift_date: Optional[dt.date] = None
    buf = np.array([], dtype=float)
    for r, d in zip(rates, dates):
        buf = np.append(buf, float(r))
        shift, _z = _detect_regime_shift(buf, cfg)
        can_trigger = True
        if last_shift_date is not None:
            if (d - last_shift_date).days < cfg.regime_cooldown_days:
                can_trigger = False
        if shift and can_trigger:
            regime_start_idx = max(0, buf.size - cfg.regime_reset_lookback)
            last_shift_date = d
    return regime_start_idx, last_shift_date

def recommend_today(
    df_wide: pd.DataFrame,
    bank: str,
    plan_start: str,
    plan_end: str,
    bought_so_far: float,
    num_trades: int,
    last_trade_date: Optional[str],
    cfg: FXPlanConfigV2,
) -> Dict[str, Any]:
    """Return today's recommendation following the V2 logic, but with fixed plan horizon."""
    df = df_wide[["date", bank]].copy()
    df.columns = ["date", "rate"]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna().sort_values("date")
    if len(df) < max(30, cfg.vol_short_window + 5):
        return {"ok": False, "error": "历史数据太少，无法稳定计算信号（至少建议几十个交易日以上）。"}

    today = df["date"].iloc[-1]
    today_rate = float(df["rate"].iloc[-1])

    start_d = dt.date.fromisoformat(plan_start)
    end_d = dt.date.fromisoformat(plan_end)
    if today < start_d:
        return {"ok": False, "error": "今日数据日期早于计划开始日期。"}
    H_total = (end_d - start_d).days + 1
    step_idx = (today - start_d).days
    if H_total <= 0:
        return {"ok": False, "error": "计划结束日期需晚于开始日期。"}
    if step_idx >= H_total:
        # plan already ended; treat as endgame
        step_idx = H_total - 1

    # Rates buffer up to today (inclusive)
    rates = df["rate"].to_numpy(dtype=float)
    dates = df["date"].to_numpy()
    dates = np.array([dt.date.fromisoformat(str(x)) for x in dates], dtype=object)

    regime_start_idx, last_shift_date = _compute_regime_state_sequential(rates, dates, cfg)
    eff_known = rates[regime_start_idx:]

    state = _pick_mode_and_params(eff_known, cfg)
    gamma = state["gamma"]
    corridor_halfwidth = state["corridor_halfwidth"]
    base_floor = state["base_floor"]
    base_cap = state["base_cap"]
    addon_max = state["addon_max"]
    mode = state["mode"]

    days_left = (H_total - 1) - step_idx
    endgame = (days_left <= cfg.last_k_days_force)

    if cfg.corridor_shrink_near_end:
        frac_left = (days_left + 1) / max(H_total, 1)
        mult = max(cfg.corridor_min_multiplier, frac_left)
        corridor_halfwidth = corridor_halfwidth * mult

    q, z = _compute_signals_online(eff_known, today_rate, cfg)

    target_ratio = _schedule_ratio(step_idx, H_total, gamma)
    lower_ratio = max(0.0, target_ratio - corridor_halfwidth)
    upper_ratio = min(1.0, target_ratio + corridor_halfwidth)

    lower_bought = cfg.total_eur * lower_ratio
    upper_bought = cfg.total_eur * upper_ratio
    mid_bought   = cfg.total_eur * target_ratio

    bought = float(bought_so_far)
    remaining = float(cfg.total_eur - bought)

    below_lower = (bought + 1e-9) < lower_bought
    above_upper = (bought - 1e-9) > upper_bought
    lag_to_mid = max(0.0, mid_bought - bought)

    # interval
    can_trade = True
    last_td: Optional[dt.date] = None
    if last_trade_date:
        try:
            last_td = dt.date.fromisoformat(last_trade_date)
        except Exception:
            last_td = None
    if last_td is not None:
        if (today - last_td).days < cfg.min_interval_days:
            can_trade = False

    if cfg.force_override_interval and (endgame or (below_lower and days_left <= cfg.override_interval_last_days)):
        can_trade = True

    forced = endgame or below_lower
    expensive = ((not np.isnan(q)) and (q >= cfg.skip_if_q_ge)) or ((not np.isnan(z)) and (z >= cfg.skip_if_z_ge))

    eur_amt = 0.0
    reasons = []

    if remaining <= 1e-9:
        reasons.append("already_completed")
    elif num_trades >= cfg.max_trades:
        reasons.append("max_trades_reached")
    elif not can_trade:
        reasons.append("cooldown_interval")
    else:
        if above_upper and (not endgame):
            reasons.append("pause_above_upper")
            eur_amt = 0.0
        else:
            if forced:
                base = max(base_floor, min(lag_to_mid, base_cap))
                reasons.append("endgame_force" if endgame else "below_lower_force")
            else:
                base = base_floor
                reasons.append("base_in_corridor")

            add = 0.0
            if (not expensive) and (not np.isnan(q) or not np.isnan(z)):
                score = _cheap_score(q, z, cfg)
                add = addon_max * (score ** cfg.score_power)
                if add > 1e-9:
                    reasons.append(f"addon(score={score:.2f})")

            if (not forced) and expensive:
                eur_amt = 0.0
                reasons.append("skip_expensive")
            else:
                eur_amt = base + add

            if eur_amt > 0:
                eur_amt = min(eur_amt, cfg.max_trade_eur)
                eur_amt = max(eur_amt, cfg.min_trade_eur)
                eur_amt = min(eur_amt, remaining)

    return {
        "ok": True,
        "today": today.isoformat(),
        "today_rate": today_rate,
        "recommend_eur": round(float(eur_amt), 2),
        "would_trade": bool(eur_amt > 0),
        "reasons": "+".join(reasons),
        "q": None if np.isnan(q) else round(float(q), 4),
        "z": None if np.isnan(z) else round(float(z), 4),
        "mode": mode,
        "bought_so_far": round(bought, 2),
        "remaining": round(remaining, 2),
        "plan_start": plan_start,
        "plan_end": plan_end,
        "H_total": int(H_total),
        "step_idx": int(step_idx),
        "days_left": int(days_left),
        "target_ratio": round(float(target_ratio), 4),
        "lower_ratio": round(float(lower_ratio), 4),
        "upper_ratio": round(float(upper_ratio), 4),
        "lower_bought": round(float(lower_bought), 2),
        "upper_bought": round(float(upper_bought), 2),
        "mid_bought": round(float(mid_bought), 2),
        "below_lower": bool(below_lower),
        "above_upper": bool(above_upper),
        "endgame": bool(endgame),
        "expensive": bool(expensive),
        "can_trade": bool(can_trade),
        "regime_start_idx": int(regime_start_idx),
        "last_regime_shift_date": last_shift_date.isoformat() if last_shift_date else None,
        "state": {
            "vol_ratio": None if np.isnan(state["vol_ratio"]) else round(float(state["vol_ratio"]), 4),
            "trend_slope_norm": None if np.isnan(state["trend_slope_norm"]) else round(float(state["trend_slope_norm"]), 6),
            "high_vol": bool(state["high_vol"]),
            "trend_up": bool(state["trend_up"]),
            "trend_down": bool(state["trend_down"]),
        }
    }

def cfg_to_dict(cfg: FXPlanConfigV2) -> Dict[str, Any]:
    return asdict(cfg)
