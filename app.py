"""
Flight Price Forecaster — Dashboard
====================================
Two inputs (origin + destination) → rich multi-dimensional visualizations
showing predicted prices across all airlines, stop counts, and the next 90 days.

Launch with:  streamlit run app.py
"""

from __future__ import annotations

from datetime import date, timedelta
from itertools import product as iterproduct

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.database import load_cleaned_flights, DB_PATH
from src.features import FEATURE_COLUMNS, US_HOLIDAYS
from src.model import load_model, MODEL_PATH

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flight Price Forecaster",
    page_icon="flight_departure",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    div[data-testid="stMetric"] label { font-size: 0.85rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def cached_load_model():
    return load_model()


@st.cache_data
def cached_load_flights():
    if DB_PATH.exists():
        return load_cleaned_flights()
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Guard: model must exist
# ---------------------------------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        "No trained model found. Run the pipeline first:\n\n"
        "```\npython -m src.data_collection\npython -m src.model\n```"
    )
    st.stop()

model, cat_maps = cached_load_model()
hist_df = cached_load_flights()


# ---------------------------------------------------------------------------
# Batch prediction engine
# ---------------------------------------------------------------------------
@st.cache_data
def generate_predictions(_model, cat_maps, origin, destination, _hist_df):
    """Predict prices for every (airline × stops × date) combination."""
    airlines = sorted(cat_maps["airline"].keys())
    stops_options = [0, 1, 2]

    # Sample dates: daily for 30 days, then every 3 days to 90
    sampled_days = sorted(set(
        list(range(1, 31)) + list(range(30, 91, 3))
    ))

    # Flight duration estimate from historical data
    dur_est = 240.0
    if _hist_df is not None and not _hist_df.empty:
        route = _hist_df[
            (_hist_df["origin"] == origin) & (_hist_df["destination"] == destination)
        ]
        if not route.empty and "flight_duration_minutes" in route.columns:
            med = route["flight_duration_minutes"].median()
            if pd.notna(med):
                dur_est = med

    today = date.today()
    rows = []
    for days, airline, stops in iterproduct(sampled_days, airlines, stops_options):
        dep_date = today + timedelta(days=days)
        dow = dep_date.weekday()
        rows.append({
            # Model features
            "origin_encoded": cat_maps["origin"].get(origin, -1),
            "destination_encoded": cat_maps["destination"].get(destination, -1),
            "airline_encoded": cat_maps["airline"].get(airline, -1),
            "stops": stops,
            "days_until_departure": days,
            "departure_day_of_week": dow,
            "departure_month": dep_date.month,
            "is_weekend": int(dow >= 5),
            "is_holiday": int((dep_date.month, dep_date.day) in US_HOLIDAYS),
            "flight_duration_minutes": dur_est,
            # Display metadata
            "_airline": airline,
            "_departure_date": dep_date,
        })

    df = pd.DataFrame(rows)
    feature_df = df[FEATURE_COLUMNS].copy()
    feature_df.columns = [str(c) for c in feature_df.columns]
    df["predicted_price"] = _model.predict(feature_df)

    return df[
        ["days_until_departure", "_departure_date", "_airline", "stops", "predicted_price"]
    ].rename(columns={"_airline": "airline", "_departure_date": "departure_date"})


# ---------------------------------------------------------------------------
# Header & inputs
# ---------------------------------------------------------------------------
st.markdown("## :airplane: Flight Price Forecaster")
st.caption(
    "Select a route to explore predicted prices across every airline, "
    "stop option, and travel date for the next 90 days."
)

col_orig, col_dest, col_go = st.columns([2, 2, 1])
with col_orig:
    origin = st.selectbox(
        "From", sorted(cat_maps["origin"].keys()),
        index=None, placeholder="Origin airport…",
    )
with col_dest:
    dest_opts = (
        [d for d in sorted(cat_maps["destination"].keys()) if d != origin]
        if origin else sorted(cat_maps["destination"].keys())
    )
    destination = st.selectbox(
        "To", dest_opts,
        index=None, placeholder="Destination airport…",
    )
with col_go:
    st.markdown("<br>", unsafe_allow_html=True)
    search = st.button("Explore Prices", type="primary", use_container_width=True)

# Persist across reruns
if search and origin and destination:
    st.session_state["preds"] = generate_predictions(
        model, cat_maps, origin, destination, hist_df,
    )
    st.session_state["route"] = (origin, destination)

if "preds" not in st.session_state:
    st.info("Pick an origin and destination above, then click **Explore Prices**.")
    st.stop()

df = st.session_state["preds"]
route_origin, route_dest = st.session_state["route"]

st.markdown("---")
st.subheader(f"Route: {route_origin} :arrow_right: {route_dest}")

# ---------------------------------------------------------------------------
# Best-deal summary
# ---------------------------------------------------------------------------
best = df.loc[df["predicted_price"].idxmin()]
avg_price = df["predicted_price"].mean()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(":trophy: Best Price", f"${best['predicted_price']:,.0f}")
m2.metric(":airplane: Airline", best["airline"])
m3.metric(":calendar: Date", best["departure_date"].strftime("%b %d, %Y"))
m4.metric(":arrows_counterclockwise: Stops", int(best["stops"]))
m5.metric(":bar_chart: Avg Price", f"${avg_price:,.0f}")

st.markdown("")

# =========================================================================
# CHART A — 90-Day Price Forecast (hero)
# =========================================================================
daily = df.groupby("days_until_departure")["predicted_price"].agg(
    ["min", "max", "mean"]
).reset_index()

fig_ts = go.Figure()

# Shaded band (max → min)
fig_ts.add_trace(go.Scatter(
    x=daily["days_until_departure"], y=daily["max"],
    mode="lines", line=dict(width=0),
    showlegend=False, hoverinfo="skip",
))
fig_ts.add_trace(go.Scatter(
    x=daily["days_until_departure"], y=daily["min"],
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(0, 212, 170, 0.12)",
    showlegend=False, hoverinfo="skip",
))

# Average line
fig_ts.add_trace(go.Scatter(
    x=daily["days_until_departure"], y=daily["mean"],
    mode="lines", line=dict(color="#00d4aa", width=2.5),
    name="Average",
))
# Cheapest line
fig_ts.add_trace(go.Scatter(
    x=daily["days_until_departure"], y=daily["min"],
    mode="lines", line=dict(color="#ffa726", width=2, dash="dot"),
    name="Cheapest",
))
# Most expensive line
fig_ts.add_trace(go.Scatter(
    x=daily["days_until_departure"], y=daily["max"],
    mode="lines", line=dict(color="#ef5350", width=1.5, dash="dash"),
    name="Most Expensive",
))

fig_ts.update_layout(
    title="90-Day Price Forecast (all airlines & stop options)",
    xaxis_title="Days Until Departure",
    yaxis_title="Predicted Price ($)",
    template="plotly_dark",
    height=420,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_ts, use_container_width=True)

# =========================================================================
# CHART B (left) — Heatmap: Airline × Booking Window
# CHART C (right) — Stops Impact by Airline
# =========================================================================
left_col, right_col = st.columns(2)

# --- Chart B: Heatmap ---------------------------------------------------
with left_col:
    nonstop = df[df["stops"] == 0]
    if nonstop.empty:
        nonstop = df  # fallback if no nonstop data

    heatmap_data = nonstop.pivot_table(
        index="airline",
        columns="days_until_departure",
        values="predicted_price",
        aggfunc="mean",
    )

    fig_hm = px.imshow(
        heatmap_data,
        labels=dict(x="Days Until Departure", y="Airline", color="Price ($)"),
        aspect="auto",
        color_continuous_scale="Turbo",
    )
    fig_hm.update_layout(
        title="Price by Airline & Booking Window (Nonstop)",
        template="plotly_dark",
        height=450,
        coloraxis_colorbar=dict(title="Price $"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# --- Chart C: Stops Impact -----------------------------------------------
with right_col:
    stops_agg = (
        df.groupby(["airline", "stops"])["predicted_price"]
        .mean()
        .reset_index()
    )
    stops_agg["stops"] = stops_agg["stops"].map(
        {0: "Nonstop", 1: "1 Stop", 2: "2 Stops"}
    )

    fig_bar = px.bar(
        stops_agg,
        x="airline", y="predicted_price", color="stops",
        barmode="group",
        labels={
            "predicted_price": "Avg Price ($)",
            "airline": "Airline",
            "stops": "Stops",
        },
        color_discrete_sequence=["#00d4aa", "#ffa726", "#ef5350"],
    )
    fig_bar.update_layout(
        title="How Stops Affect Price by Airline",
        template="plotly_dark",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# =========================================================================
# CHART D — Price Calendar (best day to fly)
# =========================================================================
cheapest_day = (
    df.groupby("departure_date")["predicted_price"]
    .min()
    .reset_index()
)
cheapest_day["departure_date"] = pd.to_datetime(cheapest_day["departure_date"])
cheapest_day["dow"] = cheapest_day["departure_date"].dt.dayofweek
cheapest_day["week"] = (
    cheapest_day["departure_date"].dt.isocalendar().week.astype(int)
)
cheapest_day["date_label"] = cheapest_day["departure_date"].dt.strftime("%b %d")

# Pivot into calendar grid
cal_pivot = cheapest_day.pivot_table(
    index="dow", columns="week", values="predicted_price", aggfunc="min",
)
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
cal_pivot.index = [day_names[i] for i in cal_pivot.index]

# Build hover text with date labels
date_pivot = cheapest_day.pivot_table(
    index="dow", columns="week", values="date_label", aggfunc="first",
)
date_pivot.index = [day_names[i] for i in date_pivot.index]
hover_text = date_pivot.fillna("").values

fig_cal = go.Figure(data=go.Heatmap(
    z=cal_pivot.values,
    x=[f"W{w}" for w in cal_pivot.columns],
    y=cal_pivot.index.tolist(),
    text=hover_text,
    hovertemplate="<b>%{text}</b><br>%{y}<br>Cheapest: $%{z:,.0f}<extra></extra>",
    colorscale="RdYlGn_r",
    colorbar=dict(title="Price $"),
))
fig_cal.update_layout(
    title="Price Calendar — Cheapest Available Flight Per Day",
    template="plotly_dark",
    height=320,
    yaxis=dict(autorange="reversed"),
    xaxis_title="Week of Year",
)
st.plotly_chart(fig_cal, use_container_width=True)

st.caption(
    f"Based on {len(df):,} predictions across "
    f"{df['airline'].nunique()} airlines, 3 stop options, "
    f"and {df['departure_date'].nunique()} dates."
)
