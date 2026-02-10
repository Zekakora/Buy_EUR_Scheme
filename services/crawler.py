from __future__ import annotations
import datetime as dt
from typing import List, Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.kylc.com/huilv/d-{bank}-eur.html?datefrom={datefrom}&dateto={dateto}"

DEFAULT_BANKS = ["boc", "icbc", "ccb", "cib", "bocm", "citic", "cmb"]

def parse_html_table(table) -> pd.DataFrame:
    # headers
    headers = []
    thead = table.find("thead")
    if thead:
        headers = [th.get_text(strip=True) for th in thead.select("th")]
    else:
        first_tr = table.find("tr")
        if first_tr:
            ths = first_tr.find_all("th")
            if ths:
                headers = [th.get_text(strip=True) for th in ths]

    # rows
    rows_data = []
    tbody = table.find("tbody")
    trs = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]
    for tr in trs:
        tds = tr.find_all(["td", "th"])
        row = [td.get_text(strip=True) for td in tds]
        if row:
            rows_data.append(row)

    df = pd.DataFrame(rows_data, columns=headers if headers else None)
    return df

def fetch_bank_history(bank: str, datefrom: str, dateto: str, timeout: int=15) -> pd.DataFrame:
    """Return DataFrame columns: date (yyyy-mm-dd), rate (float)."""
    url = BASE_URL.format(bank=bank, datefrom=datefrom, dateto=dateto)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; fx-bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.select_one("table.ui-responsive.table-stripe.bank_huilv_table.text-nowrap")
    if table is None:
        raise RuntimeError(f"Cannot find target table in HTML for bank={bank}")
    df = parse_html_table(table)

    # keep needed columns (site uses Chinese headers)
    if "日期" not in df.columns:
        # sometimes columns may differ; fail loudly to adjust parser
        raise RuntimeError(f"Unexpected columns for bank={bank}: {list(df.columns)[:10]}")
    # Prefer 现汇卖出价; if absent, try other common names
    rate_col = None
    for c in ["现汇卖出价", "现汇卖出", "卖出价", "现汇卖出价(100外币)"]:
        if c in df.columns:
            rate_col = c
            break
    if rate_col is None:
        raise RuntimeError(f"Cannot locate rate column in {list(df.columns)}")

    out = df[["日期", rate_col]].copy()
    out.columns = ["date", "rate"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)

    # numeric cleanup
    out["rate"] = (out["rate"]
                  .astype(str)
                  .str.replace(",", "", regex=False)
                  .str.replace("--", "", regex=False))
    out["rate"] = pd.to_numeric(out["rate"], errors="coerce")
    out = out.dropna(subset=["date", "rate"]).drop_duplicates(subset=["date"]).sort_values("date")
    return out

def fetch_banks_history(banks: List[str], datefrom: str, dateto: str) -> pd.DataFrame:
    """Wide DataFrame: date + each bank column."""
    wide = None
    for bank in banks:
        df = fetch_bank_history(bank, datefrom, dateto)
        df = df.rename(columns={"rate": bank})
        if wide is None:
            wide = df
        else:
            wide = wide.merge(df, on="date", how="outer")
    if wide is None:
        return pd.DataFrame(columns=["date"] + banks)
    wide = wide.sort_values("date").reset_index(drop=True)
    return wide

def today_shanghai() -> str:
    # Simple: use local date. In production, you may want zoneinfo("Asia/Shanghai")
    return dt.date.today().isoformat()

def compute_next_date(d: str) -> str:
    dd = dt.date.fromisoformat(d)
    return (dd + dt.timedelta(days=1)).isoformat()
