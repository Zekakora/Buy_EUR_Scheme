"""Date parsing and normalization utilities.

The site stores dates in the database as ISO strings: YYYY-MM-DD.
We accept a few common variants from user inputs and normalize them.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Tuple


_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_iso_date(s: str) -> Tuple[str, bool]:
    """Return (iso_date, changed).

    - If s is already ISO (YYYY-MM-DD) and valid, returns it with changed=False.
    - If s is parseable in common formats, returns ISO with changed=True.
    - Otherwise raises ValueError.
    """
    raw = (s or "").strip()
    if not raw:
        raise ValueError("日期不能为空，请使用 YYYY-MM-DD")

    if _ISO_RE.match(raw):
        # Validate actual calendar date
        dt.date.fromisoformat(raw)
        return raw, False

    # Common variants
    candidates = [
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y%m%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y.%m.%d %H:%M:%S",
    ]
    for fmt in candidates:
        try:
            d = dt.datetime.strptime(raw, fmt).date()
            return d.isoformat(), True
        except ValueError:
            pass

    # Last resort: pandas (handles a wide range; still returns ISO)
    try:
        import pandas as pd

        d = pd.to_datetime(raw, errors="raise").date()
        return d.isoformat(), True
    except Exception as e:
        raise ValueError("日期格式不正确，请使用 YYYY-MM-DD") from e
