from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# (optional) logger, won’t break if not present
try:
    from src.utils.logging import get_logger  # type: ignore
except Exception:  # pragma: no cover
    import logging
    def get_logger(name):  # type: ignore
        return logging.getLogger(name)

log = get_logger(__name__)

# ---------------------------------------------------------------------
# Existing helpers (kept)
# ---------------------------------------------------------------------
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

def time_weighted_return(*_args, **_kwargs):  # Slice 3+ will make this flow-neutral
    return np.nan

def money_weighted_return(*_args, **_kwargs):  # Slice 3+ (IRR/XIRR)
    return np.nan

# ---------------------------------------------------------------------
# New: portfolio returns loader (used by /risk/vares)
# ---------------------------------------------------------------------
def _to_returns(prices_tidy: pd.DataFrame) -> pd.DataFrame:
    """
    Convert tidy prices ['date','symbol','price'] -> wide daily returns (T x N).
    """
    if prices_tidy.empty:
        return pd.DataFrame()
    wide = (
        prices_tidy
        .pivot(index="date", columns="symbol", values="price")
        .sort_index()
        .ffill()  # guard against per-coin gaps
    )
    rets = wide.pct_change().dropna(how="all")
    return rets

def load_portfolio_returns(window_days: int = 180, *, fresh: bool = False) -> pd.DataFrame:
    """
    Build a T×N matrix of daily returns for the current portfolio symbols.

    - Universe/order comes from current_weights()
    - Prices loaded via PriceStore for last `window_days + 5` days
    - Returns are wide (columns = symbols), UTC index
    """
    # local imports to avoid import-time cycles
    from src.io.loaders import PriceStore
    from src.portfolio.positions import current_weights

    w = current_weights()  # Series indexed by SYMBOL, sum ~ 1
    symbols = list(w.index)
    if not symbols:
        log.warning("current_weights() returned empty universe.")
        return pd.DataFrame()

    store = PriceStore(data_path=Path("data"), auto_backfill=True)
    now_utc = pd.Timestamp.now(tz="UTC")  # tz-safe
    start = (now_utc - pd.Timedelta(days=window_days + 5)).isoformat()

    prices = store.get_many(symbols, start=start, end=None, fresh=fresh)  # tidy ['date','symbol','price']
    rets = _to_returns(prices)

    # Ensure column order matches weights; drop all-NaN assets
    rets = rets.reindex(columns=symbols).dropna(how="all", axis=1)

    # Trim strictly to window_days
    if len(rets) > window_days:
        rets = rets.iloc[-window_days:]

    return rets
