"""
Machine Learning Model Module
==============================
Train, evaluate, tune, save, and load a GradientBoosting regression model
for flight price prediction.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_and_engineer,
    encode_categoricals,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "gb_flight_price.joblib"
ENCODERS_PATH = MODEL_DIR / "category_maps.joblib"


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Clean, engineer features, encode categoricals.
    Returns (X, y, category_maps) where category_maps can reconstruct
    codes at inference time.
    """
    cleaned = clean_and_engineer(df)
    encoded = encode_categoricals(cleaned)

    # Persist category → code mapping for inference
    cat_maps: dict[str, dict] = {}
    for col in ["origin", "destination", "airline"]:
        cat_type = encoded[col].cat
        cat_maps[col] = dict(zip(cat_type.categories, range(len(cat_type.categories))))

    X = encoded[FEATURE_COLUMNS].copy()
    X.columns = [str(c) for c in X.columns]  # force plain str (SQLAlchemy quoted_name fix)
    y = encoded[TARGET_COLUMN].copy()
    return X, y, cat_maps


def split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=seed)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> GradientBoostingRegressor:
    defaults = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    if params:
        defaults.update(params)
    model = GradientBoostingRegressor(**defaults)
    model.fit(X_train, y_train)
    return model


def tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict | None = None,
    cv: int = 3,
) -> GradientBoostingRegressor:
    if param_grid is None:
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
        }
    base = GradientBoostingRegressor(subsample=0.8, random_state=42)
    search = GridSearchCV(
        base, param_grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    return {"mae": mae, "rmse": rmse}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(model, cat_maps: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(cat_maps, ENCODERS_PATH)
    print(f"Model saved → {MODEL_PATH}")


def load_model() -> tuple:
    model = joblib.load(MODEL_PATH)
    cat_maps = joblib.load(ENCODERS_PATH)
    return model, cat_maps


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def predict_price(
    model,
    cat_maps: dict,
    origin: str,
    destination: str,
    airline: str,
    stops: int,
    days_until_departure: int,
    departure_day_of_week: int,
    departure_month: int,
    is_weekend: int,
    is_holiday: int,
    flight_duration_minutes: float,
) -> float:
    """Return a single price prediction."""
    row = pd.DataFrame([{
        "origin_encoded": cat_maps["origin"].get(origin, -1),
        "destination_encoded": cat_maps["destination"].get(destination, -1),
        "airline_encoded": cat_maps["airline"].get(airline, -1),
        "stops": stops,
        "days_until_departure": days_until_departure,
        "departure_day_of_week": departure_day_of_week,
        "departure_month": departure_month,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "flight_duration_minutes": flight_duration_minutes,
    }])
    return float(model.predict(row[FEATURE_COLUMNS])[0])


# ---------------------------------------------------------------------------
# CLI – full train pipeline (reads from collected data in the database)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from src.database import init_db, load_raw_flights, insert_cleaned_flights

    parser = argparse.ArgumentParser(description="Train flight price model")
    parser.add_argument(
        "--csv", default=None,
        help="Path to a CSV file to use instead of database (optional)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run grid-search hyperparameter tuning (slower)",
    )
    args = parser.parse_args()

    init_db()

    # --- Load data ---
    if args.csv:
        print(f"=== Loading data from {args.csv} ===")
        raw = pd.read_csv(args.csv)
    else:
        print("=== Loading collected data from database ===")
        raw = load_raw_flights()

    if raw.empty:
        print(
            "ERROR: No data found. Run data collection first:\n"
            "  python -m src.data_collection --key YOUR_SERPAPI_KEY\n"
        )
        raise SystemExit(1)

    print(f"  ↳ {len(raw)} records loaded")

    print("=== Preparing features ===")
    X, y, cat_maps = prepare_dataset(raw)
    print(f"  ↳ {len(X)} samples, {len(FEATURE_COLUMNS)} features")

    print("=== Splitting (80/20) ===")
    X_train, X_test, y_train, y_test = split(X, y)

    if args.tune:
        print("=== Hyperparameter tuning ===")
        model = tune(X_train, y_train)
    else:
        print("=== Training GradientBoosting ===")
        model = train(X_train, y_train)

    print("=== Evaluation ===")
    metrics = evaluate(model, X_test, y_test)

    print("=== Saving model ===")
    save_model(model, cat_maps)

    # Store cleaned data for the Streamlit UI's historical charts
    cleaned = clean_and_engineer(raw)
    insert_cleaned_flights(cleaned)

    print("Done — model ready for inference.")
