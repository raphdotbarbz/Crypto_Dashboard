from __future__ import annotations
import numpy as np
import pandas as pd
from .drawdown import max_drawdown

# Use 365 for crypto daily series (changeable via config later)
ANNUAL_DAYS = 365

def _daily_returns(totals: pd.DataFrame) -> pd.Series:
    t = totals.dropna(subset=["total_value"]).sort_values("date")
    return t["total_value"].pct_change().dropna()

def _ann_return(daily: pd.Series) -> float:
    if daily.empty: return np.nan
    n = len(daily)
    g = (1.0 + daily).prod()
    return g ** (ANNUAL_DAYS / n) - 1.0

def _ann_vol(daily: pd.Series) -> float:
    if daily.empty: return np.nan
    return daily.std(ddof=1) * np.sqrt(ANNUAL_DAYS)

def _downside_dev(daily: pd.Series) -> float:
    if daily.empty: return np.nan
    neg = daily[daily < 0]
    if neg.empty: return 0.0
    return neg.std(ddof=1) * np.sqrt(ANNUAL_DAYS)

def sharpe(totals: pd.DataFrame, rf_annual: float = 0.0) -> float:
    r = _daily_returns(totals)
    ar = _ann_return(r)
    av = _ann_vol(r)
    if np.isnan(ar) or np.isnan(av) or av == 0: return np.nan
    return (ar - rf_annual) / av

def sortino(totals: pd.DataFrame, rf_annual: float = 0.0) -> float:
    r = _daily_returns(totals)
    ar = _ann_return(r)
    dd = _downside_dev(r)
    if np.isnan(ar) or dd == 0 or np.isnan(dd): return np.nan
    return (ar - rf_annual) / dd

def calmar(totals: pd.DataFrame) -> float:
    r = _daily_returns(totals)
    ar = _ann_return(r)
    mdd = abs(max_drawdown(totals))
    if np.isnan(ar) or mdd == 0 or np.isnan(mdd): return np.nan
    return ar / mdd

def trailing_slice(totals: pd.DataFrame, days: int) -> pd.DataFrame:
    t = totals.dropna(subset=["total_value"]).sort_values("date")
    if t.empty: return t
    end = t["date"].iloc[-1]
    start = pd.to_datetime(end) - pd.Timedelta(days=days)
    return t[t["date"] >= start]
