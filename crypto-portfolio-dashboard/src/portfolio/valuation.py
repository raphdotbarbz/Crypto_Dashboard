from __future__ import annotations
import pandas as pd

def market_value(positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Merge positions and prices to compute market value per symbol and in total.

    positions: columns [date, symbol, quantity]
    prices: columns [date, symbol, price]

    returns: DataFrame with [date, symbol, quantity, price, value]
    """
    p = positions.copy()
    px = prices.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True)
    px["date"] = pd.to_datetime(px["date"], utc=True)
    merged = pd.merge_asof(
        p.sort_values("date"),
        px.sort_values("date"),
        by="symbol",
        on="date",
        direction="backward"
    )
    merged["value"] = merged["quantity"] * merged["price"]
    return merged

def weights_on_date(mkt_values: pd.DataFrame, dt) -> pd.DataFrame:
    """Return holdings snapshot with weights on a given date."""
    snap = mkt_values[mkt_values["date"] == pd.to_datetime(dt, utc=True)]
    tot = snap["value"].sum()
    if pd.isna(tot) or tot == 0:
        return snap.assign(weight=0.0)[["symbol","value","weight"]].sort_values("value", ascending=False)
    return (snap.assign(weight=snap["value"] / tot)
                 [["symbol","value","weight"]]
                 .sort_values("weight", ascending=False))


def total_value_series(mkt_values: pd.DataFrame) -> pd.DataFrame:
    """Keep NaN on days with no valuation (so we don’t show fake 0)."""
    tot = (mkt_values.groupby("date", as_index=False)["value"]
           .sum(min_count=1)
           .rename(columns={"value":"total_value"})
           .sort_values("date"))
    tot["ret_1d"] = tot["total_value"].pct_change(1)
    tot["ret_7d"] = tot["total_value"].pct_change(7)
    return tot
def merge_with_prices_daily(daily_pos: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    As-of merge positions with prices (backward) to get [date, symbol, quantity, price, value].
    """
    p = daily_pos.copy()
    px = prices.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True)
    px["date"] = pd.to_datetime(px["date"], utc=True)
    merged = pd.merge_asof(
        p.sort_values("date"),
        px.sort_values("date"),
        by="symbol",
        on="date",
        direction="backward"
    )
    merged["value"] = merged["quantity"] * merged["price"]
    return merged

