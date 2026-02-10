import os
import datetime as dt

from services.db import init_db, upsert_rates, get_latest_date
from services.crawler import DEFAULT_BANKS, fetch_bank_history, today_shanghai, compute_next_date
from services.alerts import check_and_send_alerts

def main():
    db_path = os.environ.get("FX_DB_PATH", os.path.join(os.path.dirname(__file__), "data", "fx.db"))
    init_db(db_path)
    dateto = today_shanghai()

    total_rows = 0
    errors = []
    for bank in DEFAULT_BANKS:
        try:
            last = get_latest_date(db_path, bank)
            if last:
                datefrom = compute_next_date(last)
            else:
                datefrom = (dt.date.fromisoformat(dateto) - dt.timedelta(days=400)).isoformat()
            df = fetch_bank_history(bank, datefrom=datefrom, dateto=dateto)
            if len(df) == 0:
                continue
            df.insert(0, "bank", bank)
            total_rows += upsert_rates(db_path, df)
        except Exception as e:
            errors.append(f"{bank}: {e}")

    print(f"done. rows={total_rows}")
    if errors:
        print("errors:")
        for e in errors:
            print(" - ", e)

    # Optional: check alert rules and send email notifications
    try:
        res = check_and_send_alerts(db_path)
        print(f"alerts checked={res.get('checked')} sent={res.get('sent')}")
    except Exception as e:
        print(f"alert check failed: {e}")

if __name__ == "__main__":
    main()
