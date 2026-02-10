import sqlite3
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import json
import pandas as pd

from .dates import normalize_iso_date

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rates (
  bank TEXT NOT NULL,
  date TEXT NOT NULL,         -- ISO yyyy-mm-dd
  rate REAL NOT NULL,
  PRIMARY KEY (bank, date)
);

CREATE TABLE IF NOT EXISTS purchases (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  bank TEXT NOT NULL,
  date TEXT NOT NULL,         -- ISO yyyy-mm-dd
  eur REAL NOT NULL,
  rate REAL NOT NULL,
  cny_cost REAL NOT NULL,
  note TEXT
);

CREATE TABLE IF NOT EXISTS plans (
  bank TEXT PRIMARY KEY,
  start_date TEXT NOT NULL,   -- ISO
  end_date TEXT NOT NULL,     -- ISO
  cfg_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS email_alerts (
  bank TEXT NOT NULL,
  email TEXT NOT NULL,
  threshold_rate REAL NOT NULL,
  comparator TEXT NOT NULL DEFAULT 'lte',   -- lte / gte
  min_reco_eur REAL NOT NULL DEFAULT 0,
  enabled INTEGER NOT NULL DEFAULT 1,
  cooldown_hours INTEGER NOT NULL DEFAULT 24,
  last_sent_at TEXT,
  PRIMARY KEY (bank, email)
);

CREATE TABLE IF NOT EXISTS email_alert_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  bank TEXT NOT NULL,
  email TEXT NOT NULL,
  sent_at TEXT NOT NULL,
  rate REAL NOT NULL,
  recommend_eur REAL NOT NULL,
  reasons TEXT
);
"""

def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    con.row_factory = sqlite3.Row
    return con

def init_db(db_path: str) -> None:
    con = connect(db_path)
    try:
        con.executescript(SCHEMA_SQL)
        con.commit()
    finally:
        con.close()

def upsert_rates(db_path: str, df_long: pd.DataFrame) -> int:
    """df_long columns: bank, date (yyyy-mm-dd), rate (float)"""
    if df_long is None or len(df_long) == 0:
        return 0
    con = connect(db_path)
    try:
        # Normalize dates to ISO (YYYY-MM-DD)
        rows = []
        for r in df_long.itertuples(index=False):
            iso, _ = normalize_iso_date(str(r.date))
            rows.append((str(r.bank), iso, float(r.rate)))
        con.executemany(
            "INSERT OR REPLACE INTO rates(bank,date,rate) VALUES(?,?,?)",
            rows
        )
        con.commit()
        return len(rows)
    finally:
        con.close()

def get_rates_wide(db_path: str, banks: List[str], date_min: Optional[str]=None, date_max: Optional[str]=None) -> pd.DataFrame:
    """Return DataFrame with columns: date + banks... (wide). Missing values kept as NaN."""
    con = connect(db_path)
    try:
        q = "SELECT bank, date, rate FROM rates WHERE bank IN ({})".format(",".join(["?"]*len(banks)))
        params: List[Any] = list(banks)
        if date_min:
            q += " AND date >= ?"
            params.append(date_min)
        if date_max:
            q += " AND date <= ?"
            params.append(date_max)
        q += " ORDER BY date ASC"
        rows = con.execute(q, params).fetchall()
        if not rows:
            return pd.DataFrame(columns=["date"]+banks)
        df = pd.DataFrame(rows, columns=["bank","date","rate"])
        wide = df.pivot(index="date", columns="bank", values="rate").reset_index()
        # ensure all banks columns exist
        for b in banks:
            if b not in wide.columns:
                wide[b] = pd.NA
        wide = wide[["date"] + banks]
        return wide
    finally:
        con.close()

def get_latest_date(db_path: str, bank: str) -> Optional[str]:
    con = connect(db_path)
    try:
        row = con.execute("SELECT MAX(date) AS d FROM rates WHERE bank=?", (bank,)).fetchone()
        if row and row["d"]:
            return str(row["d"])
        return None
    finally:
        con.close()

def insert_purchase(db_path: str, bank: str, date: str, eur: float, rate: float, note: str="") -> None:
    iso, _ = normalize_iso_date(date)
    con = connect(db_path)
    try:
        cny_cost = float(eur) * float(rate)
        con.execute(
            "INSERT INTO purchases(bank,date,eur,rate,cny_cost,note) VALUES(?,?,?,?,?,?)",
            (bank, iso, float(eur), float(rate), float(cny_cost), note)
        )
        con.commit()
    finally:
        con.close()


def get_purchase(db_path: str, purchase_id: int) -> Optional[Dict[str, Any]]:
    con = connect(db_path)
    try:
        row = con.execute(
            "SELECT id, bank, date, eur, rate, cny_cost, note FROM purchases WHERE id=?",
            (int(purchase_id),),
        ).fetchone()
        if not row:
            return None
        return {
            "id": int(row["id"]),
            "bank": row["bank"],
            "date": row["date"],
            "eur": float(row["eur"]),
            "rate": float(row["rate"]),
            "cny_cost": float(row["cny_cost"]),
            "note": row["note"] or "",
        }
    finally:
        con.close()


def update_purchase(db_path: str, purchase_id: int, date: str, eur: float, rate: float, note: str="") -> None:
    iso, _ = normalize_iso_date(date)
    con = connect(db_path)
    try:
        cny_cost = float(eur) * float(rate)
        con.execute(
            "UPDATE purchases SET date=?, eur=?, rate=?, cny_cost=?, note=? WHERE id=?",
            (iso, float(eur), float(rate), float(cny_cost), note, int(purchase_id)),
        )
        con.commit()
    finally:
        con.close()


def delete_purchase(db_path: str, purchase_id: int) -> None:
    con = connect(db_path)
    try:
        con.execute("DELETE FROM purchases WHERE id=?", (int(purchase_id),))
        con.commit()
    finally:
        con.close()

def list_purchases(db_path: str, bank: Optional[str]=None) -> pd.DataFrame:
    con = connect(db_path)
    try:
        if bank:
            rows = con.execute(
                "SELECT id, bank, date, eur, rate, cny_cost, note FROM purchases WHERE bank=? ORDER BY date DESC, id DESC",
                (bank,)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT id, bank, date, eur, rate, cny_cost, note FROM purchases ORDER BY date DESC, id DESC"
            ).fetchall()
        df = pd.DataFrame(rows, columns=["id","bank","date","eur","rate","cny_cost","note"])
        return df
    finally:
        con.close()

def upsert_plan(db_path: str, bank: str, start_date: str, end_date: str, cfg: Dict[str, Any]) -> None:
    start_iso, _ = normalize_iso_date(start_date)
    end_iso, _ = normalize_iso_date(end_date)
    con = connect(db_path)
    try:
        con.execute(
            "INSERT OR REPLACE INTO plans(bank,start_date,end_date,cfg_json) VALUES(?,?,?,?)",
            (bank, start_iso, end_iso, json.dumps(cfg, ensure_ascii=False))
        )
        con.commit()
    finally:
        con.close()

def get_plan(db_path: str, bank: str) -> Optional[Dict[str, Any]]:
    con = connect(db_path)
    try:
        row = con.execute("SELECT bank, start_date, end_date, cfg_json FROM plans WHERE bank=?", (bank,)).fetchone()
        if not row:
            return None
        return {
            "bank": row["bank"],
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "cfg": json.loads(row["cfg_json"])
        }
    finally:
        con.close()


def upsert_email_alert(
    db_path: str,
    bank: str,
    email: str,
    threshold_rate: float,
    comparator: str = "lte",
    min_reco_eur: float = 0.0,
    enabled: bool = True,
    cooldown_hours: int = 24,
) -> None:
    """Create/update a single alert per (bank,email). Keeps last_sent_at on updates."""
    comparator = (comparator or "lte").strip().lower()
    if comparator not in ("lte", "gte"):
        comparator = "lte"
    con = connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO email_alerts(bank,email,threshold_rate,comparator,min_reco_eur,enabled,cooldown_hours)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(bank,email) DO UPDATE SET
              threshold_rate=excluded.threshold_rate,
              comparator=excluded.comparator,
              min_reco_eur=excluded.min_reco_eur,
              enabled=excluded.enabled,
              cooldown_hours=excluded.cooldown_hours
            """,
            (
                bank,
                email,
                float(threshold_rate),
                comparator,
                float(min_reco_eur),
                1 if enabled else 0,
                int(cooldown_hours),
            ),
        )
        con.commit()
    finally:
        con.close()


def list_email_alerts(db_path: str, bank: Optional[str] = None) -> pd.DataFrame:
    con = connect(db_path)
    try:
        if bank:
            rows = con.execute(
                "SELECT bank,email,threshold_rate,comparator,min_reco_eur,enabled,cooldown_hours,last_sent_at FROM email_alerts WHERE bank=? ORDER BY email ASC",
                (bank,),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT bank,email,threshold_rate,comparator,min_reco_eur,enabled,cooldown_hours,last_sent_at FROM email_alerts ORDER BY bank ASC, email ASC"
            ).fetchall()
        return pd.DataFrame(
            rows,
            columns=[
                "bank",
                "email",
                "threshold_rate",
                "comparator",
                "min_reco_eur",
                "enabled",
                "cooldown_hours",
                "last_sent_at",
            ],
        )
    finally:
        con.close()


def set_email_alert_enabled(db_path: str, bank: str, email: str, enabled: bool) -> None:
    con = connect(db_path)
    try:
        con.execute(
            "UPDATE email_alerts SET enabled=? WHERE bank=? AND email=?",
            (1 if enabled else 0, bank, email),
        )
        con.commit()
    finally:
        con.close()


def delete_email_alert(db_path: str, bank: str, email: str) -> None:
    con = connect(db_path)
    try:
        con.execute("DELETE FROM email_alerts WHERE bank=? AND email=?", (bank, email))
        con.commit()
    finally:
        con.close()


def mark_email_alert_sent(
    db_path: str,
    bank: str,
    email: str,
    sent_at_iso: str,
    rate: float,
    recommend_eur: float,
    reasons: str = "",
) -> None:
    """Update last_sent_at and append an audit row."""
    con = connect(db_path)
    try:
        con.execute(
            "UPDATE email_alerts SET last_sent_at=? WHERE bank=? AND email=?",
            (sent_at_iso, bank, email),
        )
        con.execute(
            "INSERT INTO email_alert_events(bank,email,sent_at,rate,recommend_eur,reasons) VALUES(?,?,?,?,?,?)",
            (bank, email, sent_at_iso, float(rate), float(recommend_eur), reasons or ""),
        )
        con.commit()
    finally:
        con.close()
