"""
Data Collection Module
======================
Primary data source: SerpApi Google Flights API.

Provides:
  1. GoogleFlightsClient – fetches real flight offers from Google Flights via SerpApi.
  2. collect_routes()    – batch collector that sweeps multiple routes × dates and
                           stores results in the SQLite database.

Set your SerpApi key via the SERPAPI_KEY environment variable or pass it directly.
"""

from __future__ import annotations

import os
import time
import logging
from datetime import date, datetime, timedelta
from itertools import product
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

SERPAPI_ENDPOINT = "https://serpapi.com/search"


# ---------------------------------------------------------------------------
# SerpApi Google Flights Client
# ---------------------------------------------------------------------------
class GoogleFlightsClient:
    """Wrapper around SerpApi's ``google_flights`` engine."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        if not self.api_key:
            raise ValueError(
                "SerpApi key required. Set SERPAPI_KEY env var or pass api_key=."
            )

    # ---- low-level request ------------------------------------------------
    def _request(self, params: dict) -> dict:
        params = {
            "engine": "google_flights",
            "api_key": self.api_key,
            "currency": "USD",
            "hl": "en",
            "gl": "us",
            **params,
        }
        resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()

    # ---- public search ----------------------------------------------------
    def search(
        self,
        origin: str,
        destination: str,
        outbound_date: str,
        *,
        return_date: str | None = None,
        stops: int = 0,
        adults: int = 1,
        travel_class: int = 1,
        sort_by: int = 2,
    ) -> pd.DataFrame:
        """
        Search Google Flights for one-way (or round-trip) offers.

        Parameters
        ----------
        origin, destination : IATA airport codes (e.g. "JFK", "LAX").
        outbound_date       : "YYYY-MM-DD"
        return_date         : If given, search round-trip (type=1).
        stops               : 0 = any, 1 = non-stop, 2 = ≤1, 3 = ≤2
        adults              : Number of adult passengers.
        travel_class        : 1=Economy, 2=Premium economy, 3=Business, 4=First
        sort_by             : 1=Top, 2=Price, 3=Departure, 4=Arrival, 5=Duration

        Returns
        -------
        pd.DataFrame with normalised flight records.
        """
        params: dict = {
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": outbound_date,
            "adults": adults,
            "stops": stops,
            "travel_class": travel_class,
            "sort_by": sort_by,
            "type": 1 if return_date else 2,  # 1=round-trip, 2=one-way
        }
        if return_date:
            params["return_date"] = return_date

        payload = self._request(params)
        return self._parse(payload, origin, destination, outbound_date)

    # ---- parsing ----------------------------------------------------------
    @staticmethod
    def _parse(
        payload: dict,
        origin: str,
        destination: str,
        outbound_date: str,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        search_date = datetime.utcnow().strftime("%Y-%m-%d")

        all_flights = payload.get("best_flights", []) + payload.get("other_flights", [])

        for offer in all_flights:
            segments = offer.get("flights", [])
            if not segments:
                continue

            dep_airport = segments[0].get("departure_airport", {})
            arr_airport = segments[-1].get("arrival_airport", {})

            # Airline – may be multi-carrier; take the first leg
            airline = segments[0].get("airline", "Unknown")

            # Stopovers from layovers list
            layovers = offer.get("layovers", [])
            stopover_ids = [l.get("id", "") for l in layovers]

            total_duration = offer.get("total_duration", 0)  # minutes
            price = offer.get("price")
            if price is None:
                continue

            rows.append({
                "search_date": search_date,
                "departure_date": outbound_date,
                "origin": dep_airport.get("id", origin),
                "destination": arr_airport.get("id", destination),
                "airline": airline,
                "departure_time": dep_airport.get("time", ""),
                "arrival_time": arr_airport.get("time", ""),
                "duration": f"{total_duration // 60}h {total_duration % 60}m",
                "stops": len(layovers),
                "stopover_airports": ",".join(stopover_ids),
                "price": float(price),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("No flights found for %s → %s on %s", origin, destination, outbound_date)
        return df

    # ---- convenience: get price insights ----------------------------------
    def price_insights(
        self,
        origin: str,
        destination: str,
        outbound_date: str,
        return_date: str | None = None,
    ) -> dict:
        """Return the price_insights block (lowest, typical range, history)."""
        params: dict = {
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": outbound_date,
            "type": 1 if return_date else 2,
        }
        if return_date:
            params["return_date"] = return_date
        payload = self._request(params)
        return payload.get("price_insights", {})


# ---------------------------------------------------------------------------
# Batch route collector
# ---------------------------------------------------------------------------
def collect_routes(
    client: GoogleFlightsClient,
    routes: list[tuple[str, str]],
    dates: list[str],
    *,
    stops: int = 0,
    delay: float = 1.5,
    store: bool = True,
) -> pd.DataFrame:
    """
    Sweep *routes* × *dates* and return a combined DataFrame.

    Parameters
    ----------
    client : GoogleFlightsClient instance.
    routes : List of (origin, destination) IATA tuples.
    dates  : List of "YYYY-MM-DD" departure dates.
    stops  : SerpApi stops filter (0=any, 1=nonstop, 2=≤1, 3=≤2).
    delay  : Seconds between API calls (respect rate limits).
    store  : If True, insert results into the SQLite database.

    Returns
    -------
    Combined pd.DataFrame of all results.
    """
    frames: list[pd.DataFrame] = []

    for (orig, dest), dep_date in product(routes, dates):
        logger.info("Fetching %s → %s on %s …", orig, dest, dep_date)
        try:
            df = client.search(orig, dest, dep_date, stops=stops)
            if not df.empty:
                frames.append(df)
                logger.info("  ↳ %d flights found", len(df))
            else:
                logger.info("  ↳ no results")
        except requests.HTTPError as exc:
            logger.error("  ↳ HTTP error: %s", exc)
        except Exception as exc:
            logger.error("  ↳ Unexpected error: %s", exc)

        time.sleep(delay)

    if not frames:
        logger.warning("No flight data collected across all routes/dates.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if store:
        from src.database import init_db, insert_raw_flights

        init_db()
        n = insert_raw_flights(combined)
        logger.info("Stored %d raw records in the database.", n)

    return combined


# ---------------------------------------------------------------------------
# Helper: generate a date range list
# ---------------------------------------------------------------------------
def date_range(start: date, days: int, step: int = 1) -> list[str]:
    """Return a list of 'YYYY-MM-DD' strings from *start* for *days* days."""
    return [
        (start + timedelta(days=d)).isoformat()
        for d in range(0, days, step)
    ]


# ---------------------------------------------------------------------------
# CLI – example collection run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Collect Google Flights data via SerpApi")
    parser.add_argument("--key", default=None, help="SerpApi API key (or set SERPAPI_KEY env var)")
    parser.add_argument("--origins", nargs="+", default=["JFK", "LAX", "ORD"],
                        help="Origin airports")
    parser.add_argument("--destinations", nargs="+", default=["SFO", "MIA", "SEA"],
                        help="Destination airports")
    parser.add_argument("--days-ahead", type=int, default=30,
                        help="How many days ahead to search")
    parser.add_argument("--date-step", type=int, default=7,
                        help="Day interval between departure dates")
    parser.add_argument("--stops", type=int, default=0,
                        help="SerpApi stops filter (0=any, 1=nonstop)")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds between API calls")
    args = parser.parse_args()

    client = GoogleFlightsClient(api_key=args.key)

    routes = [
        (o, d)
        for o in args.origins
        for d in args.destinations
        if o != d
    ]
    dates = date_range(date.today() + timedelta(days=1), args.days_ahead, step=args.date_step)

    print(f"Collecting {len(routes)} routes × {len(dates)} dates = {len(routes) * len(dates)} queries")
    df = collect_routes(client, routes, dates, stops=args.stops, delay=args.delay, store=True)
    print(f"\nTotal records collected: {len(df)}")
    if not df.empty:
        csv_path = "data/collected_flights.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV → {csv_path}")
        print(df.head(10).to_string(index=False))
