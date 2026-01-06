#!/usr/bin/env python
from __future__ import annotations

# --- allow running this file directly (python scripts/portfolio_preview.py) ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------------

from pathlib import Path
import argparse
import pandas as pd
import yaml

from src.io.loaders import PriceStore
from src.portfolio.positions import transactions_to_positions, daily_positions
from src.portfolio.valuation import merge_with_prices_daily, total_value_series

def main():
    ap = argparse.ArgumentParser(description="Preview portfolio value + 1d/7d returns")
    ap.add_argument("--transactions", default="data/transactions.csv")
    ap.add_argument("--config", default="config/settings.yaml")
    ap.add_argument("--start", default=None)  # YYYY-MM-DD
    ap.add_argument("--end", default=None)    # YYYY-MM-DD
    args = ap.parse_args()

    # Load config & store
    cfg = yaml.safe_load(Path(args.config).read_text())
    base = Path(cfg.get("data_path","data"))
    store = PriceStore(base)

    # Load transactions
    tx = pd.read_csv(args.transactions)
    tx["date"] = pd.to_datetime(tx["date"], utc=True)
    tx["symbol"] = tx["symbol"].str.upper()
    tx["quantity"] = tx["quantity"].astype(float)

    # Positions (trade → cumulative → daily)
    pos = transactions_to_positions(tx)
    dpos = daily_positions(pos, args.start, args.end)

    # Load prices for all symbols present in daily positions
    symbols = sorted(dpos["symbol"].unique())
    prices = store.get_many(symbols, start=args.start, end=args.end)
    if prices.empty:
        raise SystemExit(
            "No cached prices found for any symbols in this window.\n"
            "→ Ingest first (or adjust --start/--end):\n"
            "  python -m src.io.ingest_prices --days 730 --api-key $COINGECKO_API_KEY"
        )

    # Merge & compute totals
    mv = merge_with_prices_daily(dpos, prices)              # [date,symbol,quantity,price,value]
    totals = total_value_series(mv)                         # [date,total_value,ret_1d,ret_7d]

    # Keep rows that actually have a value; returns may legitimately be NaN at the start
    valid = totals[totals["total_value"].notna()].copy()
    if valid.empty:
        # Print diagnostics to help you adjust the window or mapping
        print("No valued days were produced. Diagnostics:")
        print(f"  Positions date range: {dpos['date'].min()} → {dpos['date'].max()}")
        if not prices.empty:
            rng = prices.groupby("symbol")["date"].agg(["min","max"]).sort_index()
            print("  Price coverage by symbol:")
            print(rng.to_string())
        print("\nHints:\n"
              "- Check coin_map IDs for every symbol in transactions.\n"
              "- Ingest the missing symbols or widen/narrow the --start/--end window.\n")
        raise SystemExit(1)

    # Last point & safe returns (don’t break if we have <7 days)
    last = valid.iloc[-1]
    # compute fallback returns if NaN or not enough history
    if pd.isna(last.get("ret_1d")) and len(valid) >= 2:
        valid["ret_1d"] = valid["total_value"].pct_change(1)
        last["ret_1d"] = valid["ret_1d"].iloc[-1]
    if pd.isna(last.get("ret_7d")) and len(valid) >= 8:
        valid["ret_7d"] = valid["total_value"].pct_change(7)
        last["ret_7d"] = valid["ret_7d"].iloc[-1]

    print(f"Last date:   {last['date']:%Y-%m-%d}")
    print(f"Total value: {last['total_value']:,.2f}")
    print(f"1d return:   { (last['ret_1d']*100):,.2f}%"
          if pd.notna(last.get('ret_1d')) else "1d return:   n/a")
    print(f"7d return:   { (last['ret_7d']*100):,.2f}%"
          if pd.notna(last.get('ret_7d')) else "7d return:   n/a")

    # Save artifacts for dashboards
    out = Path("artifacts"); out.mkdir(exist_ok=True)
    mv.to_parquet(out / "market_values.parquet", index=False)
    valid.to_parquet(out / "totals.parquet", index=False)
    print("Saved artifacts/market_values.parquet and artifacts/totals.parquet")

if __name__ == "__main__":
    main()
