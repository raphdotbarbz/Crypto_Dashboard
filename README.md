# Crypto Portfolio Dashboard

FastAPI + Streamlit app for crypto portfolios: ingest prices, track holdings, returns, and risk tiles; multi-portfolio via SQLite; scenario testing (range/event windows).

## Features
- CoinGecko Pro price ingestion (cached Parquet in `data/prices/`).
- Portfolio overview (value, 1d/7d/30d/180d/365 returns, top holdings).
- Risk tiles: Max DD, Sharpe, Sortino, Calmar (configurable lookbacks).
- Multi-portfolio via SQLite (`data/app.db`) + CSV fallback.
- Streamlit dashboard (tiles, charts, donut holdings, append tx).
- Scenario API (range & event windows) – UI coming next.

## Quickstart
```bash
git clone https://github.com/raphdotbarbz/Crypto_Dashboard.git
cd Crypto_Dashboard

# Python 3.12 recommended
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Set your API key (or put it in .env you load yourself)
export COINGECKO_API_KEY="YOUR_KEY"
