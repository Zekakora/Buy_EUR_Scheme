import sqlite3
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import json
import pandas as pd

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
        rows = [(str(r.bank), str(r.date), float(r.rate)) for r in df_long.itertuples(index=False)]
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
    con = connect(db_path)
    try:
        cny_cost = float(eur) * float(rate)
        con.execute(
            "INSERT INTO purchases(bank,date,eur,rate,cny_cost,note) VALUES(?,?,?,?,?,?)",
            (bank, date, float(eur), float(rate), float(cny_cost), note)
        )
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
    con = connect(db_path)
    try:
        con.execute(
            "INSERT OR REPLACE INTO plans(bank,start_date,end_date,cfg_json) VALUES(?,?,?,?)",
            (bank, start_date, end_date, json.dumps(cfg, ensure_ascii=False))
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
