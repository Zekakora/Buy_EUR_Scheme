import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft Yahei'

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# =========================
# Config (V2)
# =========================
@dataclass
class FXPlanConfigV2:
    total_eur: float = 12600.0

    # trade constraints
    max_trades: int = 18
    min_interval_days: int = 5
    min_trade_eur: float = 200.0
    max_trade_eur: float = 2000.0

    # endgame
    last_k_days_force: int = 0
    force_override_interval: bool = True
    override_interval_last_days: int = 7

    # ---------- Vol / Trend state ----------
    vol_short_window: int = 20
    vol_long_window: int = 120
    vol_ratio_th: float = 1.35  # short_vol / long_vol > th => high vol

    trend_window: int = 20
    trend_slope_th: float = 0.00035  # normalized slope per day, ~0.035%/day

    # ---------- Regime shift detection ----------
    regime_short_mean_w: int = 20
    regime_long_mean_w: int = 120
    regime_shift_z_th: float = 1.8        # abs(mean_short-mean_long)/std_long > th
    regime_reset_lookback: int = 90       # after shift, only use last 90 obs as "effective history"
    regime_cooldown_days: int = 20        # avoid frequent retriggers

    # ---------- Online signals ----------
    lookback_days_for_quantile: int = 252
    z_window: int = 30
    z_min_periods: int = 10

    # ---------- Progress corridor + adaptive schedules ----------
    # Two modes: "progress" (high vol / trend up) vs "opportunity" (low vol)
    schedule_gamma_progress: float = 1.0      # closer to 1 => earlier completion
    schedule_gamma_opportunity: float = 1.45  # >1 => later completion, more waiting for dips

    corridor_halfwidth_progress: float = 0.06     # ±6% of total amount
    corridor_halfwidth_opportunity: float = 0.10  # ±10%

    slack_eur_progress: float = 300.0
    slack_eur_opportunity: float = 700.0

    base_floor_progress: float = 600.0
    base_cap_progress: float = 1400.0

    base_floor_opportunity: float = 300.0
    base_cap_opportunity: float = 1000.0

    # Optionally shrink corridor near end (helps avoid last-week cliff)
    corridor_shrink_near_end: bool = True
    corridor_min_multiplier: float = 0.35   # at very end, corridor_halfwidth *= max(mult, this)

    # ---------- Continuous add-on (smooth) ----------
    # cheap_score = wq*cheap_q + wz*cheap_z, then addon = addon_max * cheap_score^p
    addon_max_progress: float = 600.0
    addon_max_opportunity: float = 1400.0

    w_q: float = 0.65
    w_z: float = 0.35
    score_power: float = 1.25

    q_ref: float = 0.50   # q<=0.5 considered "not expensive"; cheaper as q->0
    z_ref: float = 1.5    # z=-1.5 => very cheap (scaled to 1)

    # expensive skip rule (only when not forced and not below corridor lower bound)
    skip_if_q_ge: float = 0.75
    skip_if_z_ge: float = 0.7


# =========================
# Helpers
# =========================
def _std_last(x: np.ndarray, w: int) -> float:
    if x.size < max(5, w // 3):
        return np.nan
    ww = min(w, x.size)
    return float(np.std(x[-ww:], ddof=0))

def _mean_last(x: np.ndarray, w: int) -> float:
    if x.size < max(5, w // 3):
        return np.nan
    ww = min(w, x.size)
    return float(np.mean(x[-ww:]))

def _schedule_ratio(step_idx: int, H: int, gamma: float) -> float:
    # "by end of today"
    if H <= 0:
        return 1.0
    x = (step_idx + 1) / H
    x = min(max(x, 0.0), 1.0)
    return x ** gamma

def _percentile_rank(window: np.ndarray, value: float) -> float:
    if window.size == 0:
        return np.nan
    return float(np.mean(window <= value))

def _trend_slope_norm(x: np.ndarray, w: int) -> float:
    """
    Normalized linear slope per day: slope / mean_level
    Positive => rate trending up (EUR more expensive), negative => down.
    """
    if x.size < max(6, w // 2):
        return np.nan
    ww = min(w, x.size)
    y = x[-ww:]
    mean_level = float(np.mean(y))
    if mean_level == 0:
        return np.nan
    t = np.arange(ww, dtype=float)
    slope = np.polyfit(t, y, 1)[0]  # units: rate per day
    return float(slope / mean_level)

def _detect_regime_shift(known: np.ndarray, cfg: FXPlanConfigV2) -> Tuple[bool, float]:
    """
    Return (shift, shift_z).
    shift_z = (mean_short - mean_long) / std_long
    """
    ms = _mean_last(known, cfg.regime_short_mean_w)
    ml = _mean_last(known, cfg.regime_long_mean_w)
    sl = _std_last(known, cfg.regime_long_mean_w)
    if np.isnan(ms) or np.isnan(ml) or np.isnan(sl) or sl == 0:
        return False, np.nan
    z = (ms - ml) / sl
    return (abs(z) >= cfg.regime_shift_z_th), float(z)

def _compute_signals_online(eff_known: np.ndarray, today_rate: float, cfg: FXPlanConfigV2) -> Tuple[float, float]:
    # quantile over last L in effective history
    L = min(cfg.lookback_days_for_quantile, eff_known.size)
    q_window = eff_known[-L:]
    q = _percentile_rank(q_window, today_rate)

    # z-score over last z_window
    z = np.nan
    if eff_known.size >= cfg.z_min_periods:
        W = min(cfg.z_window, eff_known.size)
        zz = eff_known[-W:]
        ma = float(np.mean(zz))
        sd = float(np.std(zz, ddof=0))
        if sd > 0:
            z = (today_rate - ma) / sd

    return q, z

def _cheap_score(q: float, z: float, cfg: FXPlanConfigV2) -> float:
    """
    cheap_q in [0,1]: q=0 ->1, q=q_ref ->0, q>q_ref ->0
    cheap_z in [0,1]: z=-z_ref ->1, z=0 ->0, z>0 ->0
    """
    cq = 0.0 if np.isnan(q) else max(0.0, (cfg.q_ref - q) / max(cfg.q_ref, 1e-9))
    cz = 0.0 if np.isnan(z) else max(0.0, (-z) / max(cfg.z_ref, 1e-9))
    cq = min(cq, 1.0)
    cz = min(cz, 1.0)
    s = cfg.w_q * cq + cfg.w_z * cz
    return float(min(max(s, 0.0), 1.0))

def _pick_mode_and_params(known: np.ndarray, cfg: FXPlanConfigV2) -> Dict[str, float]:
    """
    Decide mode (progress vs opportunity) using vol + trend.
    Return dict with gamma, corridor_halfwidth, slack, base_floor, base_cap, addon_max, flags.
    """
    vol_s = _std_last(known, cfg.vol_short_window)
    vol_l = _std_last(known, cfg.vol_long_window)
    vol_ratio = np.nan
    if not np.isnan(vol_s) and not np.isnan(vol_l) and vol_l > 0:
        vol_ratio = vol_s / vol_l

    slope_norm = _trend_slope_norm(known, cfg.trend_window)
    trend_up = (not np.isnan(slope_norm)) and (slope_norm >= cfg.trend_slope_th)
    trend_down = (not np.isnan(slope_norm)) and (slope_norm <= -cfg.trend_slope_th)

    high_vol = (not np.isnan(vol_ratio)) and (vol_ratio >= cfg.vol_ratio_th)

    # Mode rule: high vol OR trend up => prioritize progress
    mode = "progress" if (high_vol or trend_up) else "opportunity"

    if mode == "progress":
        gamma = cfg.schedule_gamma_progress
        corridor = cfg.corridor_halfwidth_progress
        slack = cfg.slack_eur_progress
        base_floor = cfg.base_floor_progress
        base_cap = cfg.base_cap_progress
        addon_max = cfg.addon_max_progress
    else:
        gamma = cfg.schedule_gamma_opportunity
        corridor = cfg.corridor_halfwidth_opportunity
        slack = cfg.slack_eur_opportunity
        base_floor = cfg.base_floor_opportunity
        base_cap = cfg.base_cap_opportunity
        addon_max = cfg.addon_max_opportunity

        # If strong downtrend (EUR getting cheaper), allow even more patience/opportunism a little
        if trend_down:
            slack *= 1.15
            corridor *= 1.10

    return {
        "mode": mode,
        "vol_short": float(vol_s) if not np.isnan(vol_s) else np.nan,
        "vol_long": float(vol_l) if not np.isnan(vol_l) else np.nan,
        "vol_ratio": float(vol_ratio) if not np.isnan(vol_ratio) else np.nan,
        "trend_slope_norm": float(slope_norm) if not np.isnan(slope_norm) else np.nan,
        "gamma": float(gamma),
        "corridor_halfwidth": float(corridor),
        "slack": float(slack),
        "base_floor": float(base_floor),
        "base_cap": float(base_cap),
        "addon_max": float(addon_max),
        "trend_up": bool(trend_up),
        "trend_down": bool(trend_down),
        "high_vol": bool(high_vol),
    }


# =========================
# Walk-forward simulator (V2)
# =========================
def simulate_walk_forward_fx_plan_v2(
    df_rates: pd.DataFrame,
    rate_col: str,
    N_test: int,
    cfg: FXPlanConfigV2,
    date_col: str = "date",
) -> Dict[str, object]:
    """
    Walk-forward online simulation with:
    - progress corridor
    - continuous add-on function
    - volatility/trend adaptive parameters
    - regime shift detection + effective-history reset
    """

    d = df_rates[[date_col, rate_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col).reset_index(drop=True)

    if len(d) < N_test + 2:
        raise ValueError(f"Not enough rows: got {len(d)}, need at least N_test+2 with N_test={N_test}")

    known_df = d.iloc[:-N_test].reset_index(drop=True)
    test_df  = d.iloc[-N_test:].reset_index(drop=True)

    known_rates_full = known_df[rate_col].to_numpy(dtype=float)

    # regime state: effective-history start index (in the full known_rates_full array)
    regime_start_idx = 0
    last_regime_shift_date: Optional[pd.Timestamp] = None

    bought = 0.0
    remaining = float(cfg.total_eur)
    trades = 0
    last_trade_date: Optional[pd.Timestamp] = None

    trade_records = []
    daily_records = []

    H = N_test

    for i in range(N_test):
        today = test_df.loc[i, date_col]
        today_rate = float(test_df.loc[i, rate_col])

        # Reveal today into full known buffer
        known_rates_full = np.append(known_rates_full, today_rate)

        # -------- regime shift detection (on full history) --------
        shift, shift_z = _detect_regime_shift(known_rates_full, cfg)
        can_trigger = True
        if last_regime_shift_date is not None:
            if (today - last_regime_shift_date).days < cfg.regime_cooldown_days:
                can_trigger = False

        regime_shifted_today = False
        if shift and can_trigger:
            # Reset effective history to recent lookback
            regime_start_idx = max(0, known_rates_full.size - cfg.regime_reset_lookback)
            last_regime_shift_date = today
            regime_shifted_today = True

        # Effective history for signal computation (avoid stale old regime)
        eff_known = known_rates_full[regime_start_idx:]

        # -------- volatility / trend adaptive params (use effective history) --------
        state = _pick_mode_and_params(eff_known, cfg)
        gamma = state["gamma"]
        corridor_halfwidth = state["corridor_halfwidth"]
        slack = state["slack"]
        base_floor = state["base_floor"]
        base_cap = state["base_cap"]
        addon_max = state["addon_max"]
        mode = state["mode"]

        days_left = (N_test - 1) - i
        endgame = (days_left <= cfg.last_k_days_force)

        # Corridor shrink near end (optional)
        if cfg.corridor_shrink_near_end:
            # multiplier goes from ~1 (early) down to >= corridor_min_multiplier (late)
            frac_left = (days_left + 1) / max(N_test, 1)
            mult = max(cfg.corridor_min_multiplier, frac_left)
            corridor_halfwidth = corridor_halfwidth * mult

        # -------- signals (online) --------
        q, z = _compute_signals_online(eff_known, today_rate, cfg)

        # -------- progress corridor --------
        target_ratio = _schedule_ratio(i, H, gamma)
        lower_ratio = max(0.0, target_ratio - corridor_halfwidth)
        upper_ratio = min(1.0, target_ratio + corridor_halfwidth)

        lower_bought = cfg.total_eur * lower_ratio
        upper_bought = cfg.total_eur * upper_ratio
        mid_bought   = cfg.total_eur * target_ratio

        below_lower = (bought + 1e-9) < lower_bought
        above_upper = (bought - 1e-9) > upper_bought

        # How far behind the midline
        lag_to_mid = max(0.0, mid_bought - bought)

        # -------- interval constraint (with optional override) --------
        can_trade = True
        if last_trade_date is not None:
            if (today - last_trade_date).days < cfg.min_interval_days:
                can_trade = False

        if cfg.force_override_interval and (endgame or (below_lower and days_left <= cfg.override_interval_last_days)):
            can_trade = True

        # Forced logic:
        # - If below lower corridor OR endgame, we must buy (progress priority)
        forced = endgame or below_lower

        # Expensive skip:
        expensive = ((not np.isnan(q)) and (q >= cfg.skip_if_q_ge)) or ((not np.isnan(z)) and (z >= cfg.skip_if_z_ge))

        eur_amt = 0.0
        reason_parts = []

        if remaining <= 1e-9:
            # done
            pass
        elif trades >= cfg.max_trades:
            # reached trade cap
            pass
        elif not can_trade:
            reason_parts.append("cooldown_interval")
        else:
            if above_upper and (not endgame):
                # already ahead of corridor: pause
                eur_amt = 0.0
                reason_parts.append("pause_above_upper")
            else:
                if forced:
                    # Base aims to pull you back toward midline, but bounded
                    base = max(base_floor, min(lag_to_mid, base_cap))
                    reason_parts.append("endgame_force" if endgame else "below_lower_force")
                else:
                    # In corridor: base can be small; set base_floor_opportunity=0 if你想纯择机
                    base = base_floor
                    reason_parts.append("base_in_corridor")

                # Continuous add-on only when not expensive (or when forced you may still allow add-on if cheap)
                add = 0.0
                if (not expensive) and (not np.isnan(q) or not np.isnan(z)):
                    score = _cheap_score(q, z, cfg)
                    add = addon_max * (score ** cfg.score_power)
                    if add > 1e-9:
                        reason_parts.append(f"addon(score={score:.2f})")

                # If not forced and expensive: allow skipping entirely
                if (not forced) and expensive:
                    eur_amt = 0.0
                    reason_parts.append("skip_expensive")
                else:
                    eur_amt = base + add

                # bounds + remaining
                if eur_amt > 0:
                    eur_amt = min(eur_amt, cfg.max_trade_eur)
                    eur_amt = max(eur_amt, cfg.min_trade_eur)
                    eur_amt = min(eur_amt, remaining)

        # Execute
        if eur_amt > 0:
            bought += eur_amt
            remaining -= eur_amt
            trades += 1
            last_trade_date = today

            trade_records.append({
                "date": today,
                "eur": round(eur_amt, 2),
                "rate": today_rate,
                "q": None if np.isnan(q) else round(q, 4),
                "z": None if np.isnan(z) else round(z, 4),
                "mode": mode,
                "reason": "+".join(reason_parts),
                "remaining_eur_after": round(remaining, 2),
                "cny_cost": round(eur_amt * today_rate, 2),
                "target_ratio": round(target_ratio, 4),
                "lower_ratio": round(lower_ratio, 4),
                "upper_ratio": round(upper_ratio, 4),
                "regime_start_idx": int(regime_start_idx),
                "regime_shifted_today": bool(regime_shifted_today),
                "regime_shift_z": None if np.isnan(shift_z) else round(shift_z, 4),
                "vol_ratio": None if np.isnan(state["vol_ratio"]) else round(state["vol_ratio"], 4),
                "trend_slope_norm": None if np.isnan(state["trend_slope_norm"]) else round(state["trend_slope_norm"], 6),
            })

        daily_records.append({
            "date": today,
            "rate": today_rate,
            "q": None if np.isnan(q) else round(q, 4),
            "z": None if np.isnan(z) else round(z, 4),
            "mode": mode,
            "bought_so_far": round(bought, 2),
            "remaining": round(remaining, 2),
            "target_ratio": round(target_ratio, 4),
            "lower_ratio": round(lower_ratio, 4),
            "upper_ratio": round(upper_ratio, 4),
            "lower_bought": round(lower_bought, 2),
            "upper_bought": round(upper_bought, 2),
            "mid_bought": round(mid_bought, 2),
            "below_lower": bool(below_lower),
            "above_upper": bool(above_upper),
            "forced": bool(forced),
            "expensive": bool(expensive),
            "can_trade": bool(can_trade),
            "trade_eur": round(float(eur_amt), 2),
            "did_trade": bool(eur_amt > 0),
            "regime_start_idx": int(regime_start_idx),
            "regime_shifted_today": bool(regime_shifted_today),
            "regime_shift_z": None if np.isnan(shift_z) else round(shift_z, 4),
            "vol_short": None if np.isnan(state["vol_short"]) else round(state["vol_short"], 6),
            "vol_long": None if np.isnan(state["vol_long"]) else round(state["vol_long"], 6),
            "vol_ratio": None if np.isnan(state["vol_ratio"]) else round(state["vol_ratio"], 4),
            "trend_slope_norm": None if np.isnan(state["trend_slope_norm"]) else round(state["trend_slope_norm"], 6),
        })

        if remaining <= 1e-9:
            break

    trades_df = pd.DataFrame(trade_records)
    daily_df = pd.DataFrame(daily_records)

    if len(trades_df) > 0:
        avg_rate = float((trades_df["eur"] * trades_df["rate"]).sum() / trades_df["eur"].sum())
        total_cny = float(trades_df["cny_cost"].sum())
    else:
        avg_rate = np.nan
        total_cny = 0.0

    return {
        "trades": trades_df,
        "daily": daily_df,
        "completed": (remaining <= 1e-9),
        "bought_total_eur": float(cfg.total_eur - remaining),
        "remaining_eur": float(remaining),
        "num_trades": int(trades),
        "avg_rate": avg_rate,
        "total_cny_cost": total_cny,
        "test_start": test_df[date_col].iloc[0],
        "test_end": daily_df["date"].iloc[-1] if len(daily_df) else test_df[date_col].iloc[0],
    }




import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def plot_rate_with_trades(
    df_rates: pd.DataFrame,
    out: dict,
    rate_col: str,
    date_col: str = "date",
    window: str = "test",   # "test" or "all"
    annotate: bool = True,
    annotate_offset: float = 0.0,  # 文字在y轴方向偏移（以汇率单位计）
    size_min: float = 80.0,        # 圆圈最小面积
    size_max: float = 1200.0,      # 圆圈最大面积
):
    """
    画汇率走势 + 购汇点（圆圈大小=购汇金额，数字标注=金额）。
    out: simulate_walk_forward_fx_plan 的输出 dict（含 out["trades"], out["daily"], out["test_start"], out["test_end"]）
    """

    d = df_rates[[date_col, rate_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col).reset_index(drop=True)

    trades = out.get("trades", pd.DataFrame()).copy()
    if len(trades) > 0:
        trades["date"] = pd.to_datetime(trades["date"])

    # 选择绘图区间
    if window == "test" and ("test_start" in out) and ("test_end" in out):
        t0 = pd.to_datetime(out["test_start"])
        t1 = pd.to_datetime(out["test_end"])
        mask = (d[date_col] >= t0) & (d[date_col] <= t1)
        d_plot = d.loc[mask].copy()
    else:
        d_plot = d

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(d_plot[date_col], d_plot[rate_col], linewidth=1.5)

    # 叠加交易点
    if len(trades) > 0:
        # 只画在当前可视区间内的交易
        if window == "test" and ("test_start" in out) and ("test_end" in out):
            trades_plot = trades[(trades["date"] >= t0) & (trades["date"] <= t1)].copy()
        else:
            trades_plot = trades

        if len(trades_plot) > 0:
            # 交易点的y坐标：用当天汇率（从 df_rates 中匹配），匹配不到就用 trade 里记录的 rate
            # 先做一个日期->汇率的映射（按天匹配）
            rate_map = d.set_index(date_col)[rate_col]
            y = []
            for dt, r in zip(trades_plot["date"], trades_plot["rate"]):
                y.append(float(rate_map.get(dt, r)))
            y = np.array(y, dtype=float)

            eur = trades_plot["eur"].to_numpy(dtype=float)
            eur_max = max(eur.max(), 1.0)

            # scatter 的 s 是面积 (points^2)，做线性缩放
            sizes = size_min + (eur / eur_max) * (size_max - size_min)

            ax.scatter(trades_plot["date"], y, s=sizes, alpha=0.6)

            if annotate:
                # 文字略微上移，避免压在圆圈中心
                # 如果你觉得太挤，可以把 annotate_offset 调大一点，比如 0.02~0.05（取决于汇率量级）
                for dt, yy, amt in zip(trades_plot["date"], y, eur):
                    ax.annotate(
                        f"{amt:.0f}",
                        (dt, yy + annotate_offset),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

    ax.set_xlabel("Date")
    ax.set_ylabel(rate_col)
    ax.set_title("FX rate & purchases (circle size = EUR amount, label = EUR)")

    # 日期轴格式更清晰
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()

# NOTE: Example usage removed for library import.
