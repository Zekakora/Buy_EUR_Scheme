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
    insert_purchase, list_purchases, upsert_plan, get_plan
)
from services.crawler import DEFAULT_BANKS, fetch_bank_history, today_shanghai, compute_next_date
from services.strategy import cfg_from_form, cfg_to_dict, run_backtest, recommend_today, FXPlanConfigV2

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
            start_date = request.form.get("start_date", plan["start_date"])
            end_date = request.form.get("end_date", plan["end_date"])
            upsert_plan(db_path, bank, start_date, end_date, cfg_to_dict(cfg))
            plan = get_plan(db_path, bank)
            flash("计划已保存。")

        if request.method == "POST" and action == "record_purchase":
            date = request.form.get("purchase_date", today_shanghai())
            eur = float(request.form.get("purchase_eur", 0))
            rate = float(request.form.get("purchase_rate", 0))
            note = request.form.get("note", "")
            if eur > 0 and rate > 0:
                insert_purchase(db_path, bank, date, eur, rate, note)
                flash("已记录本次购汇。")

        # Data
        df_wide = get_rates_wide(db_path, banks=[bank])
        purchases = list_purchases(db_path, bank=bank)
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

        purchases_html = None
        if len(purchases):
            purchases_show = purchases.copy()
            purchases_html = purchases_show.to_html(index=False, classes="display", table_id="purchasesTable")

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
            purchases_html=purchases_html
        )

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
