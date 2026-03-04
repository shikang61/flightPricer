# Flight Price Analysis & Forecasting System

End-to-end pipeline: collect real flight data from Google Flights (via SerpApi), engineer features, train an XGBoost price-prediction model, and serve predictions through a Streamlit UI.

## Project Structure

```
flightPricer/
├── app.py                  # Streamlit web application
├── requirements.txt
├── README.md
├── data/                   # SQLite DB and CSV exports
├── models/                 # Saved model artefacts
├── notebooks/              # Exploration notebooks
└── src/
    ├── __init__.py
    ├── data_collection.py  # SerpApi Google Flights client + batch collector
    ├── database.py         # SQLAlchemy / SQLite storage
    ├── features.py         # Cleaning & feature engineering
    └── model.py            # XGBoost training, tuning, evaluation
```

## Prerequisites

1. **Python 3.10+**
2. **SerpApi account** — sign up at [serpapi.com](https://serpapi.com) and grab your API key (free tier = 100 searches/month).

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your SerpApi key
export SERPAPI_KEY="your_key_here"

# 4. Collect flight data (customise routes, dates, etc.)
python -m src.data_collection \
    --origins JFK LAX ORD \
    --destinations SFO MIA SEA \
    --days-ahead 60 \
    --date-step 7

# 5. Train the model on collected data
python -m src.model

# 6. Launch the Streamlit app
streamlit run app.py
```

## Module Details

### Data Collection (`src/data_collection.py`)

**`GoogleFlightsClient`** — searches Google Flights via SerpApi.

```python
from src.data_collection import GoogleFlightsClient

client = GoogleFlightsClient(api_key="YOUR_KEY")
df = client.search("JFK", "LAX", "2026-06-15")
```

**`collect_routes()`** — batch sweep across multiple route × date combinations and stores results in SQLite.

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--key` | `$SERPAPI_KEY` | SerpApi key |
| `--origins` | JFK LAX ORD | Origin airports |
| `--destinations` | SFO MIA SEA | Destination airports |
| `--days-ahead` | 30 | How many days into the future to search |
| `--date-step` | 7 | Day interval between departure dates |
| `--stops` | 0 | 0=any, 1=nonstop only |
| `--delay` | 1.5 | Seconds between API calls |

### Database (`src/database.py`)

SQLite via SQLAlchemy with two tables:

| Table             | Purpose                          |
|-------------------|----------------------------------|
| `raw_flights`     | Unprocessed records as collected |
| `cleaned_flights` | Feature-engineered records       |

### Feature Engineering (`src/features.py`)

`clean_and_engineer(df)` produces:

| Feature                  | Description                              |
|--------------------------|------------------------------------------|
| `days_until_departure`   | Search date → departure date gap         |
| `departure_day_of_week`  | 0 = Monday … 6 = Sunday                 |
| `departure_month`        | 1–12                                     |
| `is_weekend`             | Sat/Sun flag                             |
| `is_holiday`             | US federal holiday flag                  |
| `flight_duration_minutes`| Parsed from duration string              |

### Model (`src/model.py`)

```bash
# Train with default XGBoost params using database data
python -m src.model

# Train from a CSV file instead
python -m src.model --csv data/collected_flights.csv

# Run hyperparameter grid search (slower, better results)
python -m src.model --tune
```

Outputs MAE and RMSE on the held-out test set.

### Streamlit App (`app.py`)

- Select origin, destination, airline, stops, and departure date.
- Click **Predict Price** to get the model's forecast.
- View historical price trend line chart and airline box plot for the route.

## Collecting More Data Over Time

Run the collector on a schedule (e.g. weekly cron) to build up historical data — the database accumulates records across runs:

```bash
# Example cron entry (every Sunday at 8 AM)
0 8 * * 0 cd /path/to/flightPricer && .venv/bin/python -m src.data_collection
```

Then retrain periodically:

```bash
python -m src.model
```
