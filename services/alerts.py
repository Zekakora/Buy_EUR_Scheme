from __future__ import annotations

import datetime as dt
from typing import Optional, Dict, Any

import pandas as pd

from services.db import (
    list_email_alerts,
    get_rates_wide,
    list_purchases,
    get_plan,
    upsert_plan,
    mark_email_alert_sent,
)
from services.strategy import recommend_today, FXPlanConfigV2, cfg_from_form, cfg_to_dict
from services.mailer import send_email_smtp


def _parse_iso_dt(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None


def _rate_match(rate: float, threshold: float, comparator: str) -> bool:
    comparator = (comparator or "lte").strip().lower()
    if comparator == "gte":
        return rate >= threshold
    return rate <= threshold


def _make_email(reco: Dict[str, Any], bank: str, threshold: float, comparator: str) -> Dict[str, str]:
    sign = "≤" if (comparator or "lte").lower() != "gte" else "≥"
    subject = f"购汇提醒｜{bank} {reco['today']} 汇率 {reco['today_rate']:.4f} {sign} {threshold:.4f}"
    text = (
        f"银行: {bank}\n"
        f"日期: {reco['today']}\n"
        f"最新汇率(CNY/EUR): {reco['today_rate']:.6f}\n"
        f"策略建议: {'买入' if reco['would_trade'] else '不买'}\n"
        f"建议买入(EUR): {reco['recommend_eur']}\n"
        f"原因: {reco.get('reasons','')}\n"
        f"q={reco.get('q')}  z={reco.get('z')}\n"
        f"进度: target={reco.get('target_ratio')}  corridor={reco.get('lower_ratio')}~{reco.get('upper_ratio')}\n"
        f"已购(EUR): {reco.get('bought_so_far')}  剩余(EUR): {reco.get('remaining')}\n"
        f"计划: {reco.get('plan_start')} -> {reco.get('plan_end')}\n"
    )

    html = f"""
    <h3>购汇提醒</h3>
    <p><b>银行</b>: {bank}</p>
    <p><b>日期</b>: {reco['today']}</p>
    <p><b>最新汇率(CNY/EUR)</b>: {reco['today_rate']:.6f} （阈值: {sign} {threshold:.6f}）</p>
    <p><b>策略建议</b>: {'买入' if reco['would_trade'] else '不买'}<br/>
       <b>建议买入(EUR)</b>: {reco['recommend_eur']}</p>
    <p><b>原因</b>: {reco.get('reasons','')}</p>
    <p><small>q={reco.get('q')} / z={reco.get('z')}<br/>
       进度 target={reco.get('target_ratio')}（{reco.get('lower_ratio')}~{reco.get('upper_ratio')}）<br/>
       已购 {reco.get('bought_so_far')} EUR, 剩余 {reco.get('remaining')} EUR<br/>
       计划 {reco.get('plan_start')} → {reco.get('plan_end')}</small></p>
    """
    return {"subject": subject, "text": text, "html": html}


def check_and_send_alerts(db_path: str, banks: Optional[list[str]] = None) -> Dict[str, Any]:
    """Check all enabled alerts and send email if:
    - strategy recommends trading today, and
    - rate meets the user threshold, and
    - not already sent within cooldown_hours.
    """
    alerts = list_email_alerts(db_path)
    if len(alerts) == 0:
        return {"sent": 0, "checked": 0}

    sent = 0
    checked = 0

    for row in alerts.itertuples(index=False):
        bank = str(row.bank)
        if banks and bank not in banks:
            continue
        if int(row.enabled) != 1:
            continue

        checked += 1

        # respect cooldown
        last_sent = _parse_iso_dt(getattr(row, "last_sent_at", None))
        cooldown_hours = int(getattr(row, "cooldown_hours", 24) or 24)
        if last_sent is not None:
            if (dt.datetime.now() - last_sent) < dt.timedelta(hours=cooldown_hours):
                continue

        # ensure plan exists (same defaults as realtime page)
        plan = get_plan(db_path, bank)
        if plan is None:
            today = dt.date.today()
            start = today.isoformat()
            end = (today + dt.timedelta(days=180)).isoformat()
            cfg0 = FXPlanConfigV2()
            upsert_plan(db_path, bank, start, end, cfg_to_dict(cfg0))
            plan = get_plan(db_path, bank)
        assert plan is not None

        # purchases so far
        purchases = list_purchases(db_path, bank=bank)
        bought_so_far = float(purchases["eur"].sum()) if len(purchases) else 0.0
        num_trades = int(len(purchases)) if len(purchases) else 0
        last_trade_date = str(purchases["date"].iloc[0]) if len(purchases) else None

        # use plan cfg (stored) for recommendation
        cfg = cfg_from_form(plan["cfg"])

        # latest rates
        df_wide = get_rates_wide(db_path, banks=[bank])
        if len(df_wide) == 0:
            continue

        reco = recommend_today(
            df_wide=df_wide,
            bank=bank,
            plan_start=plan["start_date"],
            plan_end=plan["end_date"],
            bought_so_far=bought_so_far,
            num_trades=num_trades,
            last_trade_date=last_trade_date,
            cfg=cfg,
        )
        if not reco.get("ok"):
            continue

        if not bool(reco.get("would_trade")):
            continue

        min_reco = float(getattr(row, "min_reco_eur", 0) or 0)
        if float(reco.get("recommend_eur", 0)) < min_reco:
            continue

        today_rate = float(reco["today_rate"])
        thr = float(row.threshold_rate)
        comp = str(row.comparator)
        if not _rate_match(today_rate, thr, comp):
            continue

        # send
        pkg = _make_email(reco, bank=bank, threshold=thr, comparator=comp)
        send_email_smtp(to_email=str(row.email), subject=pkg["subject"], text_body=pkg["text"], html_body=pkg["html"])

        sent_at = dt.datetime.now().isoformat(timespec="seconds")
        mark_email_alert_sent(
            db_path,
            bank=bank,
            email=str(row.email),
            sent_at_iso=sent_at,
            rate=today_rate,
            recommend_eur=float(reco.get("recommend_eur", 0)),
            reasons=str(reco.get("reasons", "")),
        )
        sent += 1

    return {"sent": sent, "checked": checked}
