from fastapi import APIRouter, Query
from ..io.loaders import load_price_series

router = APIRouter()

@router.get("/prices/{coin_id}")
def prices(coin_id: str, fresh: bool = Query(True)):
    df = load_price_series(coin_id, fresh=fresh)
    # Convert tz-aware timestamps to ms epoch for frontends
    ts_ms = (df["timestamp"].astype("int64") // 10**6).astype("int64").tolist()
    return {
        "coin_id": coin_id,
        "rows": int(len(df)),
        "data": {
            "timestamp": ts_ms,
            "price": df["price"].astype(float).round(8).tolist(),
        },
    }

