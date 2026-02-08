#!/usr/bin/env python
import os, sys, requests

BASE = "https://pro-api.coingecko.com/api/v3"
KEY = "CG-kPEhoZTjArPXKVEiJuUqkbpP"

def search_ids(query: str):
    r = requests.get(f"{BASE}/search", params={"query": query},
                     headers={"x-cg-pro-api-key": KEY}, timeout=20)
    r.raise_for_status()
    return [(c["symbol"].upper(), c["id"], c["name"]) for c in r.json().get("coins", [])]

def verify_id(cid: str):
    r = requests.get(f"{BASE}/coins/{cid}",
                     params={"localization":"false","tickers":"false","market_data":"false"},
                     headers={"x-cg-pro-api-key": KEY}, timeout=20)
    return r.status_code == 200

if __name__ == "__main__":
    for q in sys.argv[1:]:
        hits = search_ids(q)
        print(f"\n=== {q} ===")
        for sym, cid, name in hits[:8]:
            ok = "✅" if verify_id(cid) else "❌"
            print(f"{ok} {sym:6s}  {cid:30s}  {name}")


