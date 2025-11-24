from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

def load_coin_map(path: str | Path) -> pd.DataFrame:
    """Load symbol→id coin mapping (CSV with columns: symbol,id,name)."""
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].str.upper()
    return df

@dataclass
class PriceStore:
    """Simple parquet-backed price store.

    Schema: date (UTC, daily), symbol (str), price (float in base currency).
    Files: data/prices/<vendor>/<SYMBOL>.parquet
    """
    data_path: Path
    vendor: str = "coingecko"

    def __post_init__(self):
        self.base = Path(self.data_path) / "prices" / self.vendor
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.base / f"{key.upper()}.parquet"

    def put(self, symbol: str, df: pd.DataFrame) -> None:
        out = df.copy()
        if "date" not in out.columns:
            raise ValueError("Expected a 'date' column")
        if "price" not in out.columns:
            raise ValueError("Expected a 'price' column")
        out["date"] = pd.to_datetime(out["date"], utc=True)
        out = out.sort_values("date")[["date", "price"]]
        out.to_parquet(self._path(symbol))

    def get(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        p = self._path(symbol)
        if not p.exists():
            log.warning("No cached prices for %s at %s", symbol, p)
            return pd.DataFrame(columns=["date", "price"])
        df = pd.read_parquet(p)
        if start:
            df = df[df["date"] >= pd.to_datetime(start, utc=True)]
        if end:
            df = df[df["date"] <= pd.to_datetime(end, utc=True)]
        return df

    def get_many(self, symbols: Iterable[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        frames = []
        for s in symbols:
            d = self.get(s, start, end)
            if not d.empty:
                d = d.assign(symbol=s.upper())
                frames.append(d)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date","symbol","price"])
