from __future__ import annotations
from pathlib import Path
import pandas as pd


def get_vix_series(start: str | None = None, end: str | None = None) -> pd.Series:
    """Fetch ^VIX from yfinance when available; else fall back to data/factors/vix.csv.
    Returns a daily close series named 'VIX'.
    """
    try:
        import yfinance as yf  # optional dependency
        df = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)
        s = df["Adj Close"].rename("VIX").dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception:
        csv = Path("data/factors/vix.csv")
        if csv.exists():
            d = pd.read_csv(csv, parse_dates=[0])
            s = d.set_index(d.columns[0])[d.columns[1]].astype(float).rename("VIX")
            s.index = s.index.tz_localize(None)
            if start or end:
                s = s.loc[start:end]
            return s
        raise RuntimeError("VIX not available: install yfinance or add data/factors/vix.csv with columns [date,vix]")