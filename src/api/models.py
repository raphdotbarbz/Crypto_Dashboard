from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # use typing.List[...] for relationships (not builtin list)
    portfolios: List["Portfolio"] = Relationship(back_populates="owner")

class Portfolio(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    owner_id: int = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    owner: Optional["User"] = Relationship(back_populates="portfolios")
    transactions: List["Tx"] = Relationship(back_populates="portfolio")

class Tx(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    portfolio_id: int = Field(foreign_key="portfolio.id", index=True)
    date: datetime
    symbol: str
    quantity: float
    price: Optional[float] = None
    fees: float = 0.0
    notes: Optional[str] = ""

    portfolio: Optional["Portfolio"] = Relationship(back_populates="transactions")
