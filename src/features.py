"""
Feature Engineering Module
==========================
Cleans raw flight data and creates features for the ML model.
"""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

# US federal holidays (month, day) — extend as needed
US_HOLIDAYS = {
    (1, 1), (1, 20), (2, 17), (5, 26), (6, 19),
    (7, 4), (9, 1), (10, 13), (11, 11), (11, 27),
    (12, 25),
}


def _parse_duration_minutes(dur: str) -> float:
    """Convert '3h 45m' or 'PT3H45M' into total minutes."""
    if pd.isna(dur):
        return np.nan
    # Try 'Xh Ym' format
    m = re.match(r"(\d+)h\s*(\d+)m", str(dur), re.IGNORECASE)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    # Try ISO 8601 'PTxHyM'
    m = re.match(r"PT(\d+)H(\d+)M", str(dur), re.IGNORECASE)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return np.nan


def _is_holiday(date: datetime) -> int:
    return int((date.month, date.day) in US_HOLIDAYS)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept a raw-flights DataFrame and return a cleaned copy with
    engineered features ready for modelling.
    """
    df = df.copy()

    # --- type coercions ---
    df["search_date"] = pd.to_datetime(df["search_date"], errors="coerce")
    df["departure_date"] = pd.to_datetime(df["departure_date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["stops"] = pd.to_numeric(df["stops"], errors="coerce").fillna(0).astype(int)

    # drop rows missing critical fields
    df.dropna(subset=["search_date", "departure_date", "price", "origin", "destination"], inplace=True)
    df = df[df["price"] > 0].copy()

    # --- engineered features ---
    df["days_until_departure"] = (df["departure_date"] - df["search_date"]).dt.days
    df["departure_day_of_week"] = df["departure_date"].dt.dayofweek  # 0=Mon
    df["departure_month"] = df["departure_date"].dt.month
    df["is_weekend"] = (df["departure_day_of_week"] >= 5).astype(int)
    df["is_holiday"] = df["departure_date"].apply(_is_holiday)
    df["flight_duration_minutes"] = df["duration"].apply(_parse_duration_minutes)

    # fill missing durations with route median
    median_dur = df.groupby(["origin", "destination"])["flight_duration_minutes"].transform("median")
    df["flight_duration_minutes"] = df["flight_duration_minutes"].fillna(median_dur)
    df["flight_duration_minutes"] = df["flight_duration_minutes"].fillna(df["flight_duration_minutes"].median())

    # convert dates back to strings for DB storage
    df["search_date"] = df["search_date"].dt.strftime("%Y-%m-%d")
    df["departure_date"] = df["departure_date"].dt.strftime("%Y-%m-%d")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode categorical columns for tree-based models.
    Returns a copy with new `*_encoded` columns and drops the originals.
    """
    df = df.copy()
    cat_cols = ["origin", "destination", "airline"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
            df[f"{col}_encoded"] = df[col].cat.codes
    return df


FEATURE_COLUMNS = [
    "origin_encoded",
    "destination_encoded",
    "airline_encoded",
    "stops",
    "days_until_departure",
    "departure_day_of_week",
    "departure_month",
    "is_weekend",
    "is_holiday",
    "flight_duration_minutes",
]

TARGET_COLUMN = "price"


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_collection import generate_mock_data

    raw = generate_mock_data(n_records=500)
    cleaned = clean_and_engineer(raw)
    encoded = encode_categoricals(cleaned)
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Encoded columns: {list(encoded.columns)}")
    print(encoded[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().round(2))
