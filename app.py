# app.py
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Global Urban Temperatures (1743–2013)", layout="wide")
DATA_DIR = Path("data")

DOT_SIZE = 12
TIME_STEP = "Yearly"
USE_GLOBAL_COLOR_SCALE = True


@st.cache_data(show_spinner=True)
def load_data():
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No .csv files found in {DATA_DIR}. Put your dataset CSV in ./data/.")
    path = csvs[0]

    df = pd.read_csv(path)

    expected = {
        "dt",
        "AverageTemperature",
        "AverageTemperatureUncertainty",
        "City",
        "Country",
        "Latitude",
        "Longitude",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df["AverageTemperature"] = pd.to_numeric(df["AverageTemperature"], errors="coerce")
    df["AverageTemperatureUncertainty"] = pd.to_numeric(df["AverageTemperatureUncertainty"], errors="coerce")
    df = df.dropna(subset=["dt", "AverageTemperature", "Latitude", "Longitude", "City", "Country"])

    def _to_float(x):
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip()
        mult = 1.0
        if s and s[-1] in "NSEW":
            if s[-1] in "SW":
                mult = -1.0
            s = s[:-1]
        return float(s) * mult

    df["Latitude"] = df["Latitude"].apply(_to_float)
    df["Longitude"] = df["Longitude"].apply(_to_float)

    df = df.sort_values("dt").reset_index(drop=True)
    return df, path.name


st.title("Global Temperature Records by Major City (1743–2013)")
st.caption("Click Start to animate the map. Click Stop to freeze it.")

with st.spinner("Loading dataset..."):
    df, fname = load_data()

# Clamp to assignment range
df = df[(df["dt"] >= "1743-01-01") & (df["dt"] <= "2013-12-31")].copy()
if df.empty:
    st.error("No data in required range.")
    st.stop()

# Time grouping
if TIME_STEP == "Yearly":
    df["t"] = df["dt"].dt.to_period("Y").dt.to_timestamp()
    df["t_label"] = df["t"].dt.year.astype(str)
else:
    df["t"] = df["dt"].dt.to_period("M").dt.to_timestamp()
    df["t_label"] = df["t"].dt.strftime("%Y-%m")

# Aggregate
agg = (
    df.groupby(["t_label", "City", "Country"], as_index=False)
      .agg(
          AverageTemperature=("AverageTemperature", "mean"),
          AverageTemperatureUncertainty=("AverageTemperatureUncertainty", "mean"),
          Latitude=("Latitude", "mean"),
          Longitude=("Longitude", "mean"),
      )
)

agg["AverageTemperature"] = agg["AverageTemperature"].round(1)
agg["AverageTemperatureUncertainty"] = agg["AverageTemperatureUncertainty"].round(1)

if agg.empty:
    st.error("No aggregated data available.")
    st.stop()

# Color scale
if USE_GLOBAL_COLOR_SCALE:
    range_color = (
        float(agg["AverageTemperature"].min()),
        float(agg["AverageTemperature"].max())
    )
else:
    range_color = None

# Build animated figure
fig = px.scatter_geo(
    agg,
    lat="Latitude",
    lon="Longitude",
    color="AverageTemperature",
    animation_frame="t_label",
    hover_name="City",
    hover_data={
        "Country": True,
        "AverageTemperature": ":.1f",
        "AverageTemperatureUncertainty": ":.1f",
        "Latitude": ":.2f",
        "Longitude": ":.2f",
    },
    color_continuous_scale="RdBu_r",
    range_color=range_color,
)

fig.update_traces(marker=dict(size=DOT_SIZE, opacity=0.85))
fig.update_geos(showcountries=True, showcoastlines=True, projection_type="natural earth")
fig.update_layout(
    height=650,
    margin=dict(l=10, r=10, t=50, b=10),
    coloraxis_colorbar=dict(title="Temp (°C)")
)

# Add custom Start / Stop buttons
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Start",
                     method="animate",
                     args=[None, {"frame": {"duration": 300, "redraw": True},
                                  "fromcurrent": True}]),
                dict(label="Stop",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}])
            ],
            x=0.1,
            y=1.15
        )
    ]
)

st.plotly_chart(fig, use_container_width=True)
