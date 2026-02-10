import os
import pandas as pd

from services.db import init_db, upsert_rates

def main():
    db_path = os.environ.get("FX_DB_PATH", os.path.join(os.path.dirname(__file__), "data", "fx.db"))
    csv_path = os.environ.get("FX_CSV_PATH", os.path.join(os.path.dirname(__file__), "data.csv"))

    init_db(db_path)
    df = pd.read_csv(csv_path)

    # normalize date column
    if "date" not in df.columns and "日期" in df.columns:
        df = df.rename(columns={"日期": "date"})

    if "date" not in df.columns:
        raise RuntimeError(f"Cannot find date column. columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

    banks = [c for c in df.columns if c != "date"]
    rows = []
    for b in banks:
        d = df[["date", b]].copy()
        d = d.rename(columns={b: "rate"})
        d["bank"] = b
        d["rate"] = pd.to_numeric(d["rate"], errors="coerce")
        d = d.dropna(subset=["rate"])
        rows.append(d[["bank","date","rate"]])

    df_long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["bank","date","rate"])
    n = upsert_rates(db_path, df_long)
    print(f"imported {n} rows into {db_path}")

if __name__ == "__main__":
    main()
