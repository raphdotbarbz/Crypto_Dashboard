from __future__ import annotations
from sqlmodel import SQLModel, Session, create_engine
from pathlib import Path

DB_PATH = Path("data/app.db")
DB_PATH.parent.mkdir(exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

def init_db():
    from .models import User, Portfolio, Tx  # noqa
    SQLModel.metadata.create_all(engine)

def get_session():
    return Session(engine)

