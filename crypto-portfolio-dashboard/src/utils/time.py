from __future__ import annotations
from datetime import datetime, timezone, date
import pandas as pd

def today_utc() -> datetime:
    return datetime.now(tz=timezone.utc)

def to_utc_index(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Ensure df has a UTC DatetimeIndex named 'date'."""
    out = df.copy()
    if col in out.columns:
        out["date"] = pd.to_datetime(out[col], utc=True)
        out = out.drop(columns=[c for c in [col] if c != "date"])
    elif isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.tz_convert("UTC") if out.index.tz else out.index.tz_localize("UTC")
        out.index.name = "date"
        return out
    out = out.set_index("date").sort_index()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    return out

def date_range_days(start: str | date, end: str | date) -> pd.DatetimeIndex:
    return pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D", tz="UTC")
