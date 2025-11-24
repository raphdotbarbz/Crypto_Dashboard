from __future__ import annotations
import pandas as pd
import numpy as np

def horizon_return(totals: pd.DataFrame, days: int) -> float | np.nan:
    """Return over a lookback (e.g., 30/180/365) using last available value prior to the start threshold."""
    t = totals.dropna(subset=["total_value"]).sort_values("date")
    if t.empty:
        return np.nan
    end_date = t.iloc[-1]["date"]
    start_cut = pd.to_datetime(end_date) - pd.Timedelta(days=days)
    # pick the latest value on/before start_cut
    t_start = t[t["date"] <= start_cut]
    if t_start.empty:
        return np.nan
    start_val = float(t_start.iloc[-1]["total_value"])
    end_val = float(t.iloc[-1]["total_value"])
    if start_val == 0:
        return np.nan
    return end_val / start_val - 1.0

# stubs for later slices
def time_weighted_return(*_args, **_kwargs):  # Slice 3+ will make this flow-neutral
    return np.nan

def money_weighted_return(*_args, **_kwargs):  # Slice 3+ (IRR/XIRR)
    return np.nan
