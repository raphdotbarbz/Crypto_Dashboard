from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
from pathlib import Path

import pandas as pd

from .vendor_coingecko import ensure_price_cache  # resolves CG id + fetches + writes parquet
from src.utils.logging import get_logger

log = get_logger(__name__)

# Where vendor parquet files live by convention
PRICE_DIR = Path("data/prices/coingecko")


def load_coin_map(path: str | Path) -> pd.DataFrame:
    """Load symbol→id coin mapping (CSV with columns: symbol, coingecko_id, name)."""
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].str.upper()
    return df


def _df_to_series(df: pd.DataFrame) -> pd.Series:
    """
    Normalize any of our known parquet layouts into a daily Series named 'close',
    indexed by UTC-normalized date.

    Supported inputs:
      1) Vendor style: index = dates, column 'close'
      2) Store style: columns 'date' and 'price'
      3) Generic: single numeric column with a DatetimeIndex
    """
    # Case 1: vendor (index=datetime, column 'close')
    if "close" in df.columns and "date" not in df.columns:
        idx = pd.to_datetime(df.index, utc=True).normalize()
        s = pd.Series(df["close"].astype(float).values, index=idx, name="close")
        return s.sort_index()

    # Case 2: store (explicit 'date' + 'price')
    if "date" in df.columns and ("price" in df.columns or "close" in df.columns):
        col = "price" if "price" in df.columns else "close"
        dt = pd.to_datetime(df["date"], utc=True).dt.normalize()
        s = pd.Series(pd.to_numeric(df[col], errors="coerce").astype(float).values, index=dt, name="close")
        return s.sort_index()

    # Case 3: fallback — single column with datetime index
    idx = pd.to_datetime(df.index, utc=True).normalize()
    vals = pd.to_numeric(df.squeeze(), errors="coerce").astype(float)
    s = pd.Series(vals.values, index=idx, name="close")
    return s.sort_index()


@dataclass
class PriceStore:
    """
    Simple parquet-backed price store.

    Schema on disk can be either:
      - vendor: <index=date>, columns: ['close']
      - store:  columns: ['date','price']

    Files: data/prices/<vendor>/<SYMBOL>.parquet
    """
    data_path: Path
    vendor: str = "coingecko"
    auto_backfill: bool = True  # fetch from CoinGecko if file missing

    def __post_init__(self):
        self.base = Path(self.data_path) / "prices" / self.vendor
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.base / f"{key.upper().strip()}.parquet"

    def ensure(self, symbol: str) -> Path:
        """
        Ensure a parquet exists for the symbol. If missing and auto_backfill=True,
        fetch via CoinGecko and write it.
        """
        p = self._path(symbol)
        if not p.exists() and self.auto_backfill:
            try:
                # use vendor helper pointing to the canonical vendor dir
                ensure_price_cache(symbol)
            except Exception as e:
                log.warning("Auto-backfill failed for %s: %s", symbol, e)
        return self._path(symbol)

    def put(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Persist a DataFrame with either:
          - columns ['date','price'], or
          - index datetime and a column 'close'
        """
        out = df.copy()

        # normalize to ['date','price'] on disk for our own puts
        if "date" in out.columns and ("price" in out.columns or "close" in out.columns):
            if "close" in out.columns and "price" not in out.columns:
                out = out.rename(columns={"close": "price"})
            out["date"] = pd.to_datetime(out["date"], utc=True)
            out = out.sort_values("date")[["date", "price"]]
        else:
            # assume vendor style: index datetime + 'close'
            if "close" not in out.columns:
                raise ValueError("Expected a 'close' column or ['date','price'] columns")
            out = out.copy()
            out = out.rename(columns={"close": "price"})
            out = out.reset_index().rename(columns={"index": "date"})
            out["date"] = pd.to_datetime(out["date"], utc=True)
            out = out.sort_values("date")[["date", "price"]]

        out.to_parquet(self._path(symbol))

    def get(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        Return a DataFrame with columns ['date','price'] for the requested symbol.
        Applies optional start/end filters (inclusive).
        """
        sym = symbol.upper().strip()
        p = self._path(sym)
        if not p.exists():
            log.warning("No cached prices for %s at %s", sym, p)
            if self.auto_backfill:
                self.ensure(sym)
            if not p.exists():
                # still missing → empty frame
                return pd.DataFrame(columns=["date", "price"])

        raw = pd.read_parquet(p)
        s = _df_to_series(raw)
        df = s.to_frame(name="price").reset_index().rename(columns={"index": "date"})

        if start:
            df = df[df["date"] >= pd.to_datetime(start, utc=True)]
        if end:
            df = df[df["date"] <= pd.to_datetime(end, utc=True)]
        return df

    def get_many(self, symbols: Iterable[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        Concatenate multiple symbols into one tidy frame: ['date','symbol','price'].
        Missing symbols auto-backfill if enabled.
        """
        frames: list[pd.DataFrame] = []
        for s in symbols:
            d = self.get(s, start, end)
            if not d.empty:
                frames.append(d.assign(symbol=s.upper().strip()))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "symbol", "price"])


def load_price_series(symbol: str, *, auto_backfill: bool = True) -> pd.Series:
    """
    Load a **daily close** Series for a single symbol. If missing and `auto_backfill` is True,
    fetch from CoinGecko first. Returns a Series named 'close' with a UTC-normalized DatetimeIndex.
    """
    sym = symbol.upper().strip()
    path = PRICE_DIR / f"{sym}.parquet"
    if not path.exists():
        if auto_backfill:
            try:
                path = ensure_price_cache(sym)
            except Exception as e:
                log.warning("Auto-backfill failed for %s: %s", sym, e)
        if not path.exists():
            raise FileNotFoundError(f"Missing price cache for {sym}: {path}")

    df = pd.read_parquet(path)
    s = _df_to_series(df)
    return s

