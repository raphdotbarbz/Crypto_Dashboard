# Crypto Portfolio Dashboard — Skeleton (Slice 1)

This is a ready-to-run skeleton for your dashboard. It aligns with the plan:
1) Skeleton & data → 2) Portfolio core → 3) Horizons → 4) Drawdowns & ratios → ...

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional: set your timezone (defaults to Asia/Singapore)
export TZ=Asia/Singapore

# Run basic tests
PYTHONPATH=. pytest -q

# Start API (health endpoint)
python -m uvicorn src.api.server:app --reload
# -> visit http://127.0.0.1:8000/health
```

## What you have in Slice 1
- `config/settings.yaml` – paths, horizons, base currency, RF source.
- `data/` – local store for parquet/duckdb (gitignored).
- `src/io/loaders.py` – vendor loaders + `PriceStore` (parquet-backed).
- `src/io/calendars.py` – 24/7 crypto calendar helpers.
- `src/factors/risk_free.py` – loads daily RF series from CSV.
- `src/utils/*` – logging, typing, time helpers.
- `src/api/server.py` + `schemas.py` – FastAPI with `/health` and `/config`.
- `tests/` – sanity tests for config, RF loader, and price store.

### Next (Slice 2 preview)
- `portfolio/positions.py` → holdings from transactions.
- `portfolio/valuation.py` → daily portfolio value.
- `portfolio/returns.py` → TWR/MWR stubs to fill.

## Notes
- `data/` is ignored by git. Parquet cache will live in `data/prices/<vendor>/<coin>.parquet`.
- Sample `coin_map.csv` and `risk_free.csv` are included to get you moving.
