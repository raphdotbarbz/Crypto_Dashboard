# src/api/prices_router.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from typing import Optional
import pandas as pd
import numpy as np
from src.io.loaders import load_price_series

router = APIRouter(prefix="/prices", tags=["prices"])

def _to_price_df(ser_or_df: Optional[pd.Series | pd.DataFrame]) -> pd.DataFrame:
    if ser_or_df is None or len(ser_or_df) == 0:
        return pd.DataFrame(columns=["date", "price"])

    if isinstance(ser_or_df, pd.Series):
        df = ser_or_df.to_frame(name="price").reset_index()
        df = df.rename(columns={df.columns[0]: "date"})
    else:
        df = ser_or_df.copy()
        if "date" not in df.columns:
            if "timestamp" in df.columns:
                df = df.rename(columns={"timestamp": "date"})
            else:
                df = df.reset_index().rename(columns={df.columns[0]: "date"})
        if "price" not in df.columns:
            for c in ("close", "value", "Price", "adj_close"):
                if c in df.columns:
                    df = df.rename(columns={c: "price"})
                    break

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date")
    return df[["date", "price"]]

@router.get("/series")
def prices_series(symbol: str, days: int = 365):
    sym = symbol.upper().strip()
    ser = load_price_series(sym, auto_backfill=True)
    if ser is None or len(ser) == 0:
        raise HTTPException(404, f"No prices for {sym}")
    df = _to_price_df(ser)
    if days and days > 0: df = df.tail(days)
    ts_ms = (df["date"].astype("int64") // 10**6).astype("int64").tolist()
    return {"symbol": sym, "ts": ts_ms, "price": df["price"].astype(float).tolist()}

@router.get("/history")
def prices_history(symbol: str, days: int = 365):
    sym = symbol.upper().strip()
    ser = load_price_series(sym, auto_backfill=True)
    if ser is None or len(ser) == 0:
        raise HTTPException(404, f"No prices for {sym}")
    df = _to_price_df(ser)
    if days and days > 0: df = df.tail(days)
    items = [{"date": d.strftime("%Y-%m-%d"), "price": float(p)} for d, p in zip(df["date"].dt.date, df["price"])]
    return {"symbol": sym, "items": items}

# optional convenience: /prices → /prices/series
@router.get("/")
def prices_default(symbol: str, days: int = 365):
    return prices_series(symbol=symbol, days=days)