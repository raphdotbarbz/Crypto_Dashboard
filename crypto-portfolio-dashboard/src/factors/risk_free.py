from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_risk_free_series(csv_path: str | Path) -> pd.Series:
    """Load daily risk-free (annualized pct) CSV and convert to daily decimal rate.

    Expected columns:
    - date: YYYY-MM-DD
    - rf_annual_pct: e.g., 5.25 means 5.25% annualized

    Returns a Series indexed by UTC date with column name 'rf_daily' (decimal).
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date")
    # Convert annualized percentage to daily decimal (simple 365-day basis for now)
    daily = (df["rf_annual_pct"] / 100.0) / 365.0
    s = pd.Series(daily.values, index=df["date"], name="rf_daily")
    return s
