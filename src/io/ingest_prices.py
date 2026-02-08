#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from src.utils.config import load_settings
from src.io.loaders import PriceStore, load_coin_map
from src.io.vendor_coingecko import fetch_coingecko_daily_pro

def main():
    ap = argparse.ArgumentParser(description="Ingest prices (CoinGecko Pro)")
    ap.add_argument("--coin-map", default="data/coin_map.csv")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--days", default="max")  # ignored when start/end are set
    ap.add_argument("--start", default="2024-01-01")  # set your desired default window
    ap.add_argument("--end", default="2025-11-18")
    ap.add_argument("--api-key", default=None)  # CLI override if needed
    args = ap.parse_args()

    cfg = load_settings()
    data_path = Path(cfg.get("data_path", "data"))
    base_ccy = cfg.get("base_currency", "USD").lower()
    cg_cfg = (cfg.get("vendors", {}) or {}).get("coingecko", {}) or {}
    api_key = args.api_key or cg_cfg.get("api_key")
    if not api_key:
        raise SystemExit("No CoinGecko API key. Set COINGECKO_API_KEY or settings.local.yaml.")

    store = PriceStore(data_path, vendor="coingecko")
    cm = load_coin_map(args.coin_map)
    if args.symbols:
        keep = {s.upper() for s in args.symbols}
        cm = cm[cm["symbol"].isin(keep)]
        if cm.empty:
            raise SystemExit("No matching symbols in coin_map.")

    for _, row in cm.iterrows():
        sym, cid = row["symbol"], row["id"]
        print(f"Fetching {sym} ({cid}) …")
        df = fetch_coingecko_daily_pro(
            cid, vs=base_ccy, api_key=api_key, start=args.start, end=args.end, days=args.days
        )
        store.put(sym, df)
        print(f"  -> stored {len(df):,} rows")
    print("Done.")

if __name__ == "__main__":
    main()
