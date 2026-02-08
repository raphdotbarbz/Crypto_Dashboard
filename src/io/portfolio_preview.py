#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import yaml

from src.io.loaders import PriceStore, load_coin_map
from src.portfolio.positions import transactions_to_positions, daily_positions
from src.portfolio.valuation import merge_with_prices_daily, total_value_series

def main():
    ap = argparse.ArgumentParser(description="Preview portfolio value + 1d/7d returns")
    ap.add_argument("--transactions", default="data/transactions.csv")
    ap.add_argument("--config", default="config/settings.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_path = Path(cfg.get("data_path","data"))
    store = PriceStore(data_path, vendor=cfg.get("vendors",{}).get("prices","coingecko"))

    # 1) load transactions
    txn = pd.read_csv(args.transactions)
    txn["date"] = pd.to_datetime(txn["date"], utc=True)
    txn["symbol"] = txn["symbol"].str.upper()
    txn["quantity"] = txn["quantity"].astype(float)

    # 2) positions
    pos = transactions_to_positions(txn)
    dpos = daily_positions(pos, start=args.start, end=args.end)

    # 3) prices for all symbols in txns
    symbols = sorted(dpos["symbol"].unique())
    prices = store.get_many(symbols, start=args.start, end=args.end)
    if prices.empty:
        raise SystemExit("No prices in store. Ingest first with scripts/ingest_prices.py")

    # 4) valuation & returns
    mv = merge_with_prices_daily(dpos, prices)
    totals = total_value_series(mv)

    # 5) print a small summary
    last = totals.dropna().iloc[-1]
    print(f"Last date: {last['date']:%Y-%m-%d}")
    print(f"Total value: {last['total_value']:,.2f}")
    print(f"1d return: {last['ret_1d']*100:,.2f}%")
    print(f"7d return: {last['ret_7d']*100:,.2f}%")

    # optional: save artifacts
    outdir = Path("artifacts"); outdir.mkdir(exist_ok=True)
    mv.to_parquet(outdir / "market_values.parquet", index=False)
    totals.to_parquet(outdir / "totals.parquet", index=False)
    print("Saved artifacts/market_values.parquet and artifacts/totals.parquet")

if __name__ == "__main__":
    main()
