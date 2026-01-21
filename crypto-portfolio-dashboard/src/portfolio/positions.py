from __future__ import annotations
import pandas as pd

def transactions_to_positions(txn: pd.DataFrame) -> pd.DataFrame:
    """Compute running positions by symbol from a transactions DataFrame.

    Expected columns: date, symbol, quantity (signed), price (optional), fees (optional)
    Returns: DataFrame with columns: date, symbol, quantity
    """
    df = txn.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["symbol"] = df["symbol"].str.upper()
    df = df.sort_values(["symbol", "date"])
    df["quantity"] = df["quantity"].astype(float)
    df["quantity"] = df.groupby("symbol")["quantity"].cumsum()
    out = df[["date", "symbol", "quantity"]]
    return out



def daily_positions(positions: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Expand trade-date cumulative positions to daily positions (one row per day & symbol).
    Keys:
      - floor timestamps to daily first (avoid losing intraday rows)
      - forward-fill within each symbol over a full daily index
    """
    p = positions.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.floor("D")

    # last cumulative qty for (symbol, day)
    by_day = (p.sort_values(["symbol", "date"])
                .groupby(["symbol", "date"], as_index=False)["quantity"]
                .last())

    start_dt = pd.to_datetime(start, utc=True).floor("D") if start else by_day["date"].min()
    end_dt   = pd.to_datetime(end,   utc=True).floor("D") if end   else pd.Timestamp.utcnow().floor("D")
    days = pd.date_range(start_dt, end_dt, freq="D", tz="UTC")

    # wide → reindex to daily → ffill → back to long
    grid = (by_day.pivot(index="date", columns="symbol", values="quantity")
                 .reindex(days)
                 .ffill()
                 .fillna(0.0))

    out = grid.stack().rename("quantity").reset_index()
    out.columns = ["date", "symbol", "quantity"]
    return out