from __future__ import annotations
import pandas as pd

def drawdown_series(totals: pd.DataFrame) -> pd.DataFrame:
    """
    Input: totals [date, total_value] (UTC)
    Output: [date, total_value, peak, dd]
      - peak: running max of total_value
      - dd: drawdown (total_value/peak - 1)
    """
    t = totals.dropna(subset=["total_value"]).sort_values("date").copy()
    t["peak"] = t["total_value"].cummax()
    t["dd"] = t["total_value"] / t["peak"] - 1.0
    return t

def max_drawdown(totals: pd.DataFrame) -> float:
    t = drawdown_series(totals)
    return float(t["dd"].min()) if not t.empty else float("nan")

