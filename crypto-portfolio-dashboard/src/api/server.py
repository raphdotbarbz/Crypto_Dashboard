# src/api/server.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
# from fastapi.middleware.cors import CORSMiddleware  # uncomment & configure if you add a web UI

from src.io.loaders import PriceStore
from src.api.schemas import (
    PortfolioOverview,
    HoldingWeight,
    PortfolioMetrics,
    RiskTiles,
    TransactionIn,
)
from src.portfolio.positions import transactions_to_positions, daily_positions
from src.portfolio.valuation import (
    merge_with_prices_daily,
    total_value_series,
    weights_on_date,
)
from src.portfolio.returns import horizon_return
from src.risk.drawdown import max_drawdown
from src.risk.ratios import sharpe, sortino, calmar, trailing_slice

# add imports
from sqlmodel import select
from src.api.db import init_db, get_session
from src.api.models import User, Portfolio, Tx

# -----------------------------------------------------------------------------
# App & settings
# -----------------------------------------------------------------------------

app = FastAPI(title="Crypto Portfolio API", version="0.1.0")

# Optional CORS (if you front this with a browser UI)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],    # tighten later
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.on_event("startup")
def _startup():
    init_db()


def _tx_df_from_db(session, portfolio_id: int) -> pd.DataFrame:
    rows = session.exec(select(Tx).where(Tx.portfolio_id == portfolio_id).order_by(Tx.date)).all()
    if not rows:
        return pd.DataFrame(columns=["date","symbol","quantity","price","fees","notes"])
    df = pd.DataFrame([{
        "date": r.date,
        "symbol": r.symbol.upper().strip(),
        "quantity": r.quantity,
        "price": r.price,
        "fees": r.fees,
        "notes": r.notes
    } for r in rows])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df

# --- create user ---
@app.post("/users")
def create_user(email: str):
    with get_session() as s:
        if s.exec(select(User).where(User.email==email)).first():
            return {"ok": True, "id": s.exec(select(User).where(User.email==email)).first().id}
        u = User(email=email)
        s.add(u); s.commit(); s.refresh(u)
        return {"ok": True, "id": u.id}


# --- create portfolio ---
@app.post("/portfolios")
def create_portfolio(owner_id: int, name: str):
    with get_session() as s:
        owner = s.get(User, owner_id)
        if not owner: raise HTTPException(400, "owner_id not found")
        p = Portfolio(name=name, owner_id=owner_id)
        s.add(p); s.commit(); s.refresh(p)
        return {"ok": True, "id": p.id}


# --- list portfolios for a user ---
@app.get("/portfolios")
def list_portfolios(owner_id: int):
    with get_session() as s:
        rows = s.exec(select(Portfolio).where(Portfolio.owner_id==owner_id)).all()
        return [{"id": p.id, "name": p.name, "created_at": p.created_at.isoformat()} for p in rows]



# --- append tx to DB ---
@app.post("/transactions/db/append")
def transactions_append_db(portfolio_id: int, items: List[TransactionIn]):
    with get_session() as s:
        if not s.get(Portfolio, portfolio_id):
            raise HTTPException(400, "portfolio_id not found")
        for i in items:
            s.add(Tx(
                portfolio_id=portfolio_id,
                date=pd.to_datetime(i.date, utc=True).to_pydatetime(),
                symbol=i.symbol.upper().strip(),
                quantity=float(i.quantity),
                price=(float(i.price) if i.price is not None else None),
                fees=(float(i.fees) if i.fees is not None else 0.0),
                notes=i.notes or ""
            ))
        s.commit()
    return {"ok": True}



def load_settings() -> dict:
    path = Path("config/settings.yaml")
    if not path.exists():
        return {"data_path": "data", "timezone": "UTC", "vendors": {"prices": "coingecko"}}
    return yaml.safe_load(path.read_text()) or {}

SETTINGS = load_settings()
DATA_PATH = Path(SETTINGS.get("data_path", "data"))
VENDOR = (SETTINGS.get("vendors", {}) or {}).get("prices", "coingecko")
STORE = PriceStore(DATA_PATH, vendor=VENDOR)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _parse_dates_iso(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """
    Robustly parse date strings into UTC datetimes.
    - Drops rows where date is empty/nan/nat/none/null (string or NaN).
    - Parses ISO strings w/ microseconds, Z, +00:00, mixed formats.
    - If remaining bad strings exist, raises 400 with examples.
    """
    raw = df[col].astype(str).str.strip()

    # Drop placeholders
    droppable = raw.str.lower().isin(["", "nan", "nat", "none", "null"])
    if droppable.any():
        df = df[~droppable].copy()
        raw = raw[~droppable]

    # Try mixed ISO parse first (handles microseconds & offsets)
    parsed = pd.to_datetime(raw, format="mixed", utc=True, errors="coerce")

    # Fallback pass
    bad = parsed.isna()
    if bad.any():
        parsed.loc[bad] = pd.to_datetime(raw[bad], utc=True, errors="coerce")

    # Still bad? Surface examples (as strings) in a 400
    still_bad = parsed.isna()
    if still_bad.any():
        examples = raw[still_bad].head(3).tolist()
        raise HTTPException(
            status_code=400,
            detail={"msg": "Unparseable dates in transactions.csv", "examples": examples},
        )

    df[col] = parsed
    return df


def _load_transactions_csv() -> pd.DataFrame:
    tx_path = Path("data/transactions.csv")
    if not tx_path.exists():
        raise HTTPException(status_code=400, detail="Missing data/transactions.csv")
    tx = pd.read_csv(tx_path)
    if tx.empty:
        raise HTTPException(status_code=400, detail="transactions.csv is empty")
    tx = _parse_dates_iso(tx, "date")
    tx["symbol"] = tx["symbol"].astype(str).str.upper().str.strip()
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce")
    if tx["quantity"].isna().any():
        raise HTTPException(status_code=400, detail="Non-numeric quantities detected in transactions.csv")
    # price/fees optional
    if "price" in tx:
        tx["price"] = pd.to_numeric(tx["price"], errors="coerce")
    if "fees" in tx:
        tx["fees"] = pd.to_numeric(tx["fees"], errors="coerce").fillna(0.0)
    return tx

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "data_path": str(DATA_PATH),
        "vendor": VENDOR,
    }

@app.get("/config")
def config():
    return SETTINGS

@app.get("/portfolio/overview", response_model=PortfolioOverview)
def portfolio_overview(portfolio_id: int | None = None):
    if portfolio_id is not None:
        with get_session() as s:
            tx = _tx_df_from_db(s, portfolio_id)
            if tx.empty: raise HTTPException(400, "No transactions in this portfolio")
    else:
        tx = _load_transactions_csv()

    pos = transactions_to_positions(tx)
    dpos = daily_positions(pos)

    syms = sorted(dpos["symbol"].unique())
    prices = STORE.get_many(syms)
    if prices.empty:
        raise HTTPException(status_code=400, detail="No cached prices; run ingester.")

    mv = merge_with_prices_daily(dpos, prices)   # [date,symbol,quantity,price,value]
    totals = total_value_series(mv)              # [date,total_value,ret_1d,ret_7d]
    valid = totals.dropna(subset=["total_value"])
    if valid.empty:
        raise HTTPException(status_code=400, detail="No valued days produced. Check coin_map IDs or ingestion window.")

    last = valid.iloc[-1]
    w = weights_on_date(mv, last["date"]).head(10)

    return PortfolioOverview(
        as_of=last["date"].to_pydatetime(),
        total_value=float(last["total_value"]),
        ret_1d=float(last["ret_1d"]) if pd.notna(last["ret_1d"]) else None,
        ret_7d=float(last["ret_7d"]) if pd.notna(last["ret_7d"]) else None,
        ret_30d=float(horizon_return(valid, 30))  if len(valid) > 30  else None,
        ret_180d=float(horizon_return(valid, 180)) if len(valid) > 180 else None,
        ret_365d=float(horizon_return(valid, 365)) if len(valid) > 365 else None,
        top_holdings=[HoldingWeight(**r) for r in w.to_dict("records")]
    )

@app.get("/portfolio/metrics", response_model=PortfolioMetrics)
def portfolio_metrics(lookbacks: str = "90,180,365", rf_annual: float = 0.0, portfolio_id: int | None = None):
    # --- load transactions for CSV or DB portfolio ---
    if portfolio_id is not None:
        with get_session() as s:
            tx = _tx_df_from_db(s, portfolio_id)
            if tx.empty:
                raise HTTPException(status_code=400, detail="No transactions in this portfolio")
    else:
        tx = _load_transactions_csv()

    # --- positions -> daily positions ---
    pos = transactions_to_positions(tx)
    dpos = daily_positions(pos)

    # --- prices & valuation ---
    syms = sorted(dpos["symbol"].unique())
    prices = STORE.get_many(syms)
    if prices.empty:
        raise HTTPException(status_code=400, detail="No cached prices; run ingester.")

    mv = merge_with_prices_daily(dpos, prices)
    totals = total_value_series(mv).dropna(subset=["total_value"])
    if totals.empty:
        raise HTTPException(status_code=400, detail="No valued days produced.")

    as_of = totals["date"].iloc[-1].to_pydatetime()

    # --- build tiles ---
    out_tiles = []
    for tok in [t.strip() for t in lookbacks.split(",") if t.strip()]:
        try:
            lb = int(tok)
        except ValueError:
            continue
        t = trailing_slice(totals, lb)
        if t.empty:
            out_tiles.append(RiskTiles(lookback_days=lb))
            continue
        out_tiles.append(RiskTiles(
            lookback_days=lb,
            max_drawdown=float(max_drawdown(t)),
            sharpe=float(sharpe(t, rf_annual)) if len(t) > 2 else None,
            sortino=float(sortino(t, rf_annual)) if len(t) > 2 else None,
            calmar=float(calmar(t)) if len(t) > 2 else None,
        ))

    # --- explicit, JSON-serializable payload (prevents 'None' returns) ---
    payload = {
        "as_of": as_of,  # datetime; FastAPI will serialize
        "tiles": [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in out_tiles],
    }
    return payload


@app.post("/transactions/append")
def transactions_append(items: List[TransactionIn]):
    if not items:
        raise HTTPException(status_code=400, detail="No items provided.")

    tx_path = Path("data/transactions.csv")

    # Read existing or create fresh with correct dtypes
    if tx_path.exists():
        existing = pd.read_csv(tx_path)
    else:
        existing = pd.DataFrame(columns=["date", "symbol", "quantity", "price", "fees", "notes"])

    # Normalize existing
    if not existing.empty:
        existing = _parse_dates_iso(existing, "date")
        existing["symbol"] = existing["symbol"].astype(str).str.upper().str.strip()
        existing["quantity"] = pd.to_numeric(existing["quantity"], errors="coerce")
        existing["price"] = pd.to_numeric(existing.get("price", np.nan), errors="coerce")
        existing["fees"] = pd.to_numeric(existing.get("fees", 0.0), errors="coerce").fillna(0.0)
        if "notes" not in existing:
            existing["notes"] = ""
    else:
        existing = pd.DataFrame({
            "date": pd.Series([], dtype="datetime64[ns, UTC]"),
            "symbol": pd.Series([], dtype="string"),
            "quantity": pd.Series([], dtype="float"),
            "price": pd.Series([], dtype="float"),
            "fees": pd.Series([], dtype="float"),
            "notes": pd.Series([], dtype="string"),
        })

    # Normalize incoming
    new = pd.DataFrame([{
        "date": pd.to_datetime(i.date, utc=True),
        "symbol": i.symbol.upper().strip(),
        "quantity": float(i.quantity),
        "price": float(i.price) if i.price is not None else np.nan,
        "fees": float(i.fees) if i.fees is not None else 0.0,
        "notes": i.notes or "",
    } for i in items])

    # De-dup key: ISO timestamp + normalized symbol/qty/price (no fragile int casting)
    def _key(df: pd.DataFrame) -> pd.Series:
        iso = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        sym = df["symbol"].astype(str).str.upper().str.strip()
        qty = pd.to_numeric(df["quantity"], errors="coerce").round(10).astype(str)
        prc = pd.to_numeric(df["price"], errors="coerce").fillna(-0.0).round(10).astype(str)
        return iso + "|" + sym + "|" + qty + "|" + prc

    existing["_k"] = _key(existing) if not existing.empty else pd.Series([], dtype="string")
    new["_k"] = _key(new)

    out = (pd.concat([existing, new], ignore_index=True)
           .drop_duplicates("_k")
           .drop(columns="_k")
           .sort_values("date"))

    out.to_csv(tx_path, index=False)
    return {"ok": True, "rows": int(len(out))}

@app.get("/transactions")
def list_transactions(limit: int = 100):
    p = Path("data/transactions.csv")
    if not p.exists():
        return []

    df = pd.read_csv(p)
    df = _parse_dates_iso(df, "date")

    # Sort & limit
    df = df.sort_values("date", ascending=False).head(limit)

    # Make it JSON-safe:
    # - convert datetime to ISO strings
    # - replace NaN/NaT with None (JSON null)
    df["date"] = df["date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.where(pd.notnull(df), None)

    return df.to_dict(orient="records")


@app.get("/transactions/db")
def list_transactions_db(portfolio_id: int, limit: int = 200):
    with get_session() as s:
        rows = s.exec(select(Tx).where(Tx.portfolio_id == portfolio_id).order_by(Tx.date.desc())).all()
        out = [{
            "id": r.id,
            "date": pd.to_datetime(r.date).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": r.symbol,
            "quantity": r.quantity,
            "price": (None if r.price is None else float(r.price)),
            "fees": float(r.fees),
            "notes": r.notes or ""
        } for r in rows[:limit]]
        return out


@app.get("/users")
def list_users():
    with get_session() as s:
        rows = s.exec(select(User).order_by(User.id)).all()
        return [{"id": u.id, "email": u.email, "created_at": u.created_at.isoformat()} for u in rows]


@app.get("/portfolio/totals")
def portfolio_totals(portfolio_id: int | None = None):
    # reuse the same helper logic you use in /portfolio/overview
    if portfolio_id is not None:
        with get_session() as s:
            tx = _tx_df_from_db(s, portfolio_id)
            if tx.empty:
                raise HTTPException(400, "No transactions in this portfolio")
    else:
        tx = _load_transactions_csv()

    pos = transactions_to_positions(tx)
    dpos = daily_positions(pos)
    syms = sorted(dpos["symbol"].unique())
    prices = STORE.get_many(syms)
    if prices.empty:
        raise HTTPException(400, "No cached prices; run ingester.")

    mv = merge_with_prices_daily(dpos, prices)
    totals = total_value_series(mv).dropna(subset=["total_value"]).sort_values("date")
    # JSON-safe: ISO date + floats; include returns if present
    out = []
    for _, r in totals.iterrows():
        out.append({
            "date": pd.to_datetime(r["date"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_value": float(r["total_value"]),
            "ret_1d": (None if pd.isna(r.get("ret_1d")) else float(r["ret_1d"])) if "ret_1d" in totals else None,
            "ret_7d": (None if pd.isna(r.get("ret_7d")) else float(r["ret_7d"])) if "ret_7d" in totals else None,
        })
    return out

