"""
Database Module
===============
SQLite-backed storage via SQLAlchemy for raw and cleaned flight data.
Designed so historical records accumulate over repeated collection runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "flights.db"


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


class RawFlight(Base):
    __tablename__ = "raw_flights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_date = Column(String, nullable=False)
    departure_date = Column(String, nullable=False)
    origin = Column(String(5), nullable=False)
    destination = Column(String(5), nullable=False)
    airline = Column(String(50))
    departure_time = Column(String)
    arrival_time = Column(String)
    duration = Column(String)
    stops = Column(Integer)
    stopover_airports = Column(String)
    price = Column(Float, nullable=False)


class CleanedFlight(Base):
    __tablename__ = "cleaned_flights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_date = Column(String)
    departure_date = Column(String)
    origin = Column(String(5))
    destination = Column(String(5))
    airline = Column(String(50))
    departure_time = Column(String)
    arrival_time = Column(String)
    duration = Column(String)
    stops = Column(Integer)
    stopover_airports = Column(String)
    price = Column(Float)
    # Engineered features stored for convenience
    days_until_departure = Column(Integer)
    departure_day_of_week = Column(Integer)
    departure_month = Column(Integer)
    is_weekend = Column(Integer)
    is_holiday = Column(Integer)
    flight_duration_minutes = Column(Float)


# ---------------------------------------------------------------------------
# Engine / session helpers
# ---------------------------------------------------------------------------
def get_engine(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they don't exist."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)


def get_session(db_path: Path = DB_PATH) -> Session:
    engine = get_engine(db_path)
    return sessionmaker(bind=engine)()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------
def insert_raw_flights(df: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    """Append a DataFrame of raw flights. Returns number of rows inserted."""
    engine = get_engine(db_path)
    init_db(db_path)
    df = df.drop(columns=["id"], errors="ignore")  # let SQLite auto-generate IDs
    rows = df.to_dict(orient="records")
    with Session(engine) as session:
        session.execute(RawFlight.__table__.insert(), rows)
        session.commit()
    return len(rows)


def insert_cleaned_flights(df: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    engine = get_engine(db_path)
    init_db(db_path)
    df = df.drop(columns=["id"], errors="ignore")  # let SQLite auto-generate IDs
    rows = df.to_dict(orient="records")
    with Session(engine) as session:
        session.execute(CleanedFlight.__table__.insert(), rows)
        session.commit()
    return len(rows)


def _fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert SQLAlchemy quoted_name columns to plain str."""
    df.columns = [str(c) for c in df.columns]
    return df


def load_raw_flights(db_path: Path = DB_PATH) -> pd.DataFrame:
    engine = get_engine(db_path)
    init_db(db_path)
    return _fix_columns(pd.read_sql_table("raw_flights", engine))


def load_cleaned_flights(db_path: Path = DB_PATH) -> pd.DataFrame:
    engine = get_engine(db_path)
    init_db(db_path)
    return _fix_columns(pd.read_sql_table("cleaned_flights", engine))


def row_count(table: str, db_path: Path = DB_PATH) -> int:
    engine = get_engine(db_path)
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {DB_PATH}")

    from src.data_collection import generate_mock_data

    df = generate_mock_data(n_records=5000)
    n = insert_raw_flights(df)
    print(f"Inserted {n} raw flight records.")
