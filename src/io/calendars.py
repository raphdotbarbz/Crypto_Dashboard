import pandas as pd

def crypto_day_index(start, end) -> pd.DatetimeIndex:
    """24/7 daily calendar for crypto markets (UTC)."""
    return pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D", tz="UTC")
