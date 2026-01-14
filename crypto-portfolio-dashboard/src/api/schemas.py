from __future__ import annotations
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class PricePoint(BaseModel):
    date: datetime
    symbol: str
    price: float

class PriceSeries(BaseModel):
    symbol: str
    points: List[PricePoint]

class HoldingWeight(BaseModel):
    symbol: str
    value: float
    weight: float

class PortfolioOverview(BaseModel):
    as_of: datetime
    total_value: float
    ret_1d: Optional[float] = None
    ret_7d: Optional[float] = None
    ret_30d: Optional[float] = None
    ret_180d: Optional[float] = None
    ret_365d: Optional[float] = None
    top_holdings: List[HoldingWeight]

# Resolve forward references under postponed annotations
PriceSeries.model_rebuild()
PortfolioOverview.model_rebuild()

class TransactionIn(BaseModel):
    date: datetime
    symbol: str
    quantity: float
    price: float | None = None
    fees: float | None = 0.0
    notes: str | None = ""


class RiskTiles(BaseModel):
    lookback_days: int
    max_drawdown: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    calmar: float | None = None

class PortfolioMetrics(BaseModel):
    as_of: datetime
    tiles: list[RiskTiles]


class BackfillRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    days: Optional[str] = "max"  # or an int like "365"