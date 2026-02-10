from __future__ import annotations

import os
import datetime as dt
import json
from typing import Dict, Any, Optional, List

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import plotly.graph_objects as go

from services.db import (
    init_db, upsert_rates, get_rates_wide, get_latest_date,
    insert_purchase, list_purchases, upsert_plan, get_plan,
    get_purchase, update_purchase, delete_purchase,
    upsert_email_alert, list_email_alerts, delete_email_alert, set_email_alert_enabled
)
from services.crawler import DEFAULT_BANKS, fetch_bank_history, today_shanghai, compute_next_date
from services.strategy import cfg_from_form, cfg_to_dict, run_backtest, recommend_today, FXPlanConfigV2
from services.alerts import check_and_send_alerts
from services.dates import normalize_iso_date
from services.dates import normalize_iso_date

APP_TITLE = "人民币兑欧元购汇策略"

def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")

    db_path = os.environ.get("FX_DB_PATH", os.path.join(os.path.dirname(__file__), "data", "fx.db"))
    app.config["FX_DB_PATH"] = db_path
    init_db(db_path)

    @app.route("/")
    def index():
        return render_template("index.html", title=APP_TITLE, banks=DEFAULT_BANKS)

    @app.route("/admin/update", methods=["GET"])
    def admin_update():
        """Manually trigger crawler update for all banks (simple admin action)."""
        db_path = app.config["FX_DB_PATH"]
        total_rows = 0
        errors: List[str] = []
        dateto = today_shanghai()

        for bank in DEFAULT_BANKS:
            try:
                last = get_latest_date(db_path, bank)
                if last:
                    datefrom = compute_next_date(last)
                else:
                    # first time: fetch ~400 days
                    datefrom = (dt.date.fromisoformat(dateto) - dt.timedelta(days=400)).isoformat()
                df = fetch_bank_history(bank, datefrom=datefrom, dateto=dateto)
                if len(df) == 0:
                    continue
                df_long = df.copy()
                df_long.insert(0, "bank", bank)
                total_rows += upsert_rates(db_path, df_long)
            except Exception as e:
                errors.append(f"{bank}: {e}")

        if errors:
            flash("更新完成但有错误：" + " | ".join(errors))
        else:
            flash(f"更新成功：写入/更新 {total_rows} 行汇率数据。")

        # After updating data, optionally run email alerts
        try:
            res = check_and_send_alerts(db_path)
            if res.get("sent", 0) > 0:
                flash(f"已发送邮件提醒：{res.get('sent')} 条（检查 {res.get('checked')} 条规则）。")
        except Exception as e:
            # do not break the update action
            flash(f"邮件提醒检查失败：{e}")

        return redirect(url_for("index"))

    def _plot_rate_with_markers(df: pd.DataFrame, trades: pd.DataFrame, title: str) -> str:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rate"],
            mode="lines",
            name="汇率"
        ))
        if trades is not None and len(trades) > 0:
            eur = trades["eur"].astype(float).to_numpy()
            s = 8 + 22 * (eur / max(eur.max(), 1.0))  # marker size (px)
            fig.add_trace(go.Scatter(
                x=trades["date"], y=trades["rate"],
                mode="markers+text",
                name="购汇",
                marker=dict(size=s, opacity=0.65),
                text=[f"{v:.0f}" for v in eur],
                textposition="top center"
            ))

        fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="CNY / EUR (现汇卖出)",
            margin=dict(l=30, r=20, t=50, b=40),
            height=520,
        )

        fig.update_xaxes(
            type="date",
            rangeslider=dict(visible=True)
        )

        # let front-end enable scrollZoom
        return fig.to_json()

    @app.route("/backtest", methods=["GET", "POST"])
    def backtest():
        db_path = app.config["FX_DB_PATH"]
        banks = DEFAULT_BANKS
        bank = request.values.get("bank", "boc")
        n_test = int(float(request.values.get("n_test", 90)))
        cfg = cfg_from_form(request.values)
        out = None
        plot_json = None
        trades_html = None
        daily_html = None
        kpi = None

        # Load data (wide)
        df_wide = get_rates_wide(db_path, banks=[bank])
        if len(df_wide) == 0:
            flash("数据库暂无汇率数据，请先点首页的“更新数据”。")
            return render_template("backtest.html", title=APP_TITLE, banks=banks, bank=bank, n_test=n_test, cfg=cfg_to_dict(cfg))

        try:
            out = run_backtest(df_wide, bank=bank, n_test=n_test, cfg=cfg)
            trades = out["trades"].copy()
            daily = out["daily"].copy()

            # For plotting: use df in out.daily (has date, rate)
            # df_plot = daily[["date", "rate"]].copy()
            # df_plot["date"] = pd.to_datetime(df_plot["date"])
            df_plot = df_wide[["date", bank]].copy()
            df_plot.columns = ["date", "rate"]
            df_plot = df_plot.dropna()
            df_plot["date"] = pd.to_datetime(df_plot["date"])
            trades_plot = trades[["date", "eur", "rate"]].copy() if len(trades) else trades
            if len(trades_plot):
                trades_plot["date"] = pd.to_datetime(trades_plot["date"])

            plot_json = _plot_rate_with_markers(df_plot, trades_plot, title=f"回测 - {bank}（全量数据）")

            kpi = {
                "completed": out["completed"],
                "bought_total_eur": round(float(out["bought_total_eur"]), 2),
                "remaining_eur": round(float(out["remaining_eur"]), 2),
                "avg_rate": None if pd.isna(out["avg_rate"]) else round(float(out["avg_rate"]), 6),
                "total_cny_cost": round(float(out["total_cny_cost"]), 2),
                "num_trades": int(out["num_trades"]),
                "test_start": str(pd.to_datetime(out["test_start"]).date()),
                "test_end": str(pd.to_datetime(out["test_end"]).date()),
            }

            # tables
            if len(trades):
                trades_show = trades.copy()
                trades_show["date"] = pd.to_datetime(trades_show["date"]).dt.date.astype(str)
                trades_html = trades_show.to_html(index=False, classes="display", table_id="tradesTable")
            daily_show = daily.tail(180).copy()
            daily_show["date"] = pd.to_datetime(daily_show["date"]).dt.date.astype(str)
            daily_html = daily_show.to_html(index=False, classes="display", table_id="dailyTable")
        except Exception as e:
            flash(f"回测失败：{e}")

        return render_template(
            "backtest.html",
            title=APP_TITLE,
            banks=banks,
            bank=bank,
            n_test=n_test,
            cfg=cfg_to_dict(cfg),
            plot_json=plot_json,
            trades_html=trades_html,
            daily_html=daily_html,
            kpi=kpi,
        )

    @app.route("/alerts", methods=["GET", "POST"])
    def alerts_page():
        """Manage email alerts (email + target rate threshold)."""
        db_path = app.config["FX_DB_PATH"]
        banks = DEFAULT_BANKS
        bank = request.values.get("bank", "boc")

        action = request.form.get("action", "")
        if request.method == "POST" and action == "save_alert":
            email = (request.form.get("alert_email", "") or "").strip()
            comparator = (request.form.get("alert_comparator", "lte") or "lte").strip()
            try:
                threshold_rate = float(request.form.get("alert_threshold", 0) or 0)
                min_reco_eur = float(request.form.get("alert_min_reco_eur", 0) or 0)
                cooldown_hours = int(float(request.form.get("alert_cooldown_hours", 24) or 24))
            except Exception:
                threshold_rate, min_reco_eur, cooldown_hours = 0.0, 0.0, 24
            enabled = (request.form.get("alert_enabled", "on") or "").lower() in ("1","true","yes","y","on")
            if not email:
                flash("邮箱不能为空。")
            elif threshold_rate <= 0:
                flash("阈值汇率必须 > 0。")
            else:
                upsert_email_alert(
                    db_path,
                    bank=bank,
                    email=email,
                    threshold_rate=threshold_rate,
                    comparator=comparator,
                    min_reco_eur=min_reco_eur,
                    enabled=enabled,
                    cooldown_hours=cooldown_hours,
                )
                flash("提醒已保存。")
            return redirect(url_for("alerts_page", bank=bank))

        if request.method == "POST" and action == "delete_alert":
            email = (request.form.get("alert_email", "") or "").strip()
            if email:
                delete_email_alert(db_path, bank=bank, email=email)
                flash("已删除该提醒。")
            return redirect(url_for("alerts_page", bank=bank))

        if request.method == "POST" and action == "toggle_alert":
            email = (request.form.get("alert_email", "") or "").strip()
            enabled = (request.form.get("enabled", "0") or "0") in ("1","true","yes","y","on")
            if email:
                set_email_alert_enabled(db_path, bank=bank, email=email, enabled=enabled)
                flash("已更新提醒开关。")
            return redirect(url_for("alerts_page", bank=bank))

        alerts_df = list_email_alerts(db_path, bank=bank)
        alerts = []
        if len(alerts_df):
            alerts = alerts_df.to_dict(orient="records")

        return render_template(
            "alerts.html",
            title=APP_TITLE,
            banks=banks,
            bank=bank,
            alerts=alerts,
        )

    @app.route("/realtime", methods=["GET", "POST"])
    def realtime():
        db_path = app.config["FX_DB_PATH"]
        banks = DEFAULT_BANKS
        bank = request.values.get("bank", "boc")

        # Load / init plan
        plan = get_plan(db_path, bank)
        if plan is None:
            # default plan: 180 days
            today = dt.date.today()
            start = today.isoformat()
            end = (today + dt.timedelta(days=180)).isoformat()
            cfg = FXPlanConfigV2()
            upsert_plan(db_path, bank, start, end, cfg_to_dict(cfg))
            plan = get_plan(db_path, bank)

        assert plan is not None
        cfg = cfg_from_form({**plan["cfg"], **request.values})

        action = request.form.get("action", "")
        if request.method == "POST" and action == "save_plan":
            start_raw = request.form.get("start_date", plan["start_date"])
            end_raw = request.form.get("end_date", plan["end_date"])
            try:
                start_date, start_changed = normalize_iso_date(start_raw)
                end_date, end_changed = normalize_iso_date(end_raw)
                if start_date > end_date:
                    raise ValueError("开始日期不能晚于结束日期")
                upsert_plan(db_path, bank, start_date, end_date, cfg_to_dict(cfg))
                plan = get_plan(db_path, bank)
                if start_changed or end_changed:
                    flash("计划日期已规范化为 YYYY-MM-DD。")
                flash("计划已保存。")
            except Exception as e:
                flash(f"计划日期无效：{e}")

        if request.method == "POST" and action == "record_purchase":
            date_raw = request.form.get("purchase_date", today_shanghai())
            note = request.form.get("note", "")
            try:
                date, changed = normalize_iso_date(date_raw)
            except Exception as e:
                flash(f"日期格式不正确：{e}")
                date = None
                changed = False

            try:
                eur = float(request.form.get("purchase_eur", 0))
                rate = float(request.form.get("purchase_rate", 0))
            except Exception:
                eur, rate = 0.0, 0.0

            if date and eur > 0 and rate > 0:
                insert_purchase(db_path, bank, date, eur, rate, note)
                if changed:
                    flash("购汇日期已规范化为 YYYY-MM-DD。")
                flash("已记录本次购汇。")
            elif date:
                flash("购入欧元金额与汇率必须为正数。")

        # Purchase edit/delete actions are handled by dedicated routes.

        # Data
        df_wide = get_rates_wide(db_path, banks=[bank])
        purchases = list_purchases(db_path, bank=bank)
        alerts = list_email_alerts(db_path, bank=bank)
        bought_so_far = float(purchases["eur"].sum()) if len(purchases) else 0.0
        num_trades = int(len(purchases)) if len(purchases) else 0
        last_trade_date = str(purchases["date"].iloc[0]) if len(purchases) else None  # sorted desc

        reco = None
        plot_json = None
        latest = None

        if len(df_wide) > 0:
            # latest rate
            latest_row = df_wide.dropna(subset=[bank]).tail(1)
            if len(latest_row):
                latest = {
                    "date": str(latest_row["date"].iloc[0]),
                    "rate": float(latest_row[bank].iloc[0]),
                }

            reco = recommend_today(
                df_wide=df_wide,
                bank=bank,
                plan_start=plan["start_date"],
                plan_end=plan["end_date"],
                bought_so_far=bought_so_far,
                num_trades=num_trades,
                last_trade_date=last_trade_date,
                cfg=cfg
            )

            # plot: last 220 days
            d = df_wide[["date", bank]].copy()
            d.columns = ["date", "rate"]
            d = d.dropna()
            d["date"] = pd.to_datetime(d["date"])
            # d = d.tail(220)

            trades_plot = purchases.copy()
            if len(trades_plot):
                trades_plot = trades_plot[["date","eur","rate"]].copy()
                trades_plot["date"] = pd.to_datetime(trades_plot["date"])
            plot_json = _plot_rate_with_markers(d, trades_plot, title=f"实时监控 - {bank}（最近数据）")
        else:
            flash("数据库暂无汇率数据，请先点首页的“更新数据”。")

        purchases_rows = []
        if len(purchases):
            purchases_show = purchases.copy()
            # Ensure date is rendered as ISO string
            purchases_show["date"] = purchases_show["date"].astype(str)
            purchases_rows = purchases_show.to_dict(orient="records")

        alerts_html = None
        if len(alerts):
            a = alerts.copy()
            # nicer display
            a["enabled"] = a["enabled"].apply(lambda x: "on" if int(x)==1 else "off")
            alerts_html = a.to_html(index=False, classes="display", table_id="alertsTable")

        return render_template(
            "realtime.html",
            title=APP_TITLE,
            banks=banks,
            bank=bank,
            plan=plan,
            cfg=cfg_to_dict(cfg),
            latest=latest,
            reco=reco,
            plot_json=plot_json,
            purchases_rows=purchases_rows,
            alerts_html=alerts_html
        )

    @app.route("/purchase/edit/<int:purchase_id>", methods=["GET", "POST"])
    def purchase_edit(purchase_id: int):
        db_path = app.config["FX_DB_PATH"]
        bank = request.values.get("bank", "")

        p = get_purchase(db_path, purchase_id)
        if not p:
            flash("未找到该购汇记录。")
            return redirect(url_for("realtime", bank=bank or "boc"))

        # bank fallback from record
        bank = bank or p.get("bank") or "boc"

        if request.method == "POST":
            date_raw = request.form.get("date", p["date"])
            note = request.form.get("note", "")
            try:
                date, changed = normalize_iso_date(date_raw)
                eur = float(request.form.get("eur", p["eur"]))
                rate = float(request.form.get("rate", p["rate"]))
                if eur <= 0 or rate <= 0:
                    raise ValueError("购入欧元金额与汇率必须为正数")
                update_purchase(db_path, purchase_id, date, eur, rate, note)
                if changed:
                    flash("日期已规范化为 YYYY-MM-DD。")
                flash("购汇记录已更新。")
                return redirect(url_for("realtime", bank=bank))
            except Exception as e:
                flash(f"保存失败：{e}")

        return render_template(
            "purchase_edit.html",
            title=APP_TITLE,
            bank=bank,
            purchase=p,
        )

    @app.route("/purchase/delete/<int:purchase_id>", methods=["POST"])
    def purchase_delete(purchase_id: int):
        db_path = app.config["FX_DB_PATH"]
        bank = request.values.get("bank", "boc")
        try:
            delete_purchase(db_path, purchase_id)
            flash("已删除购汇记录。")
        except Exception as e:
            flash(f"删除失败：{e}")
        return redirect(url_for("realtime", bank=bank))

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
