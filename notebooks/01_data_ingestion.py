# %% [markdown]
# # Notebook 01: Multi-Source Data Ingestion & DuckDB Lakehouse Setup
#
# ### Dynamic Pricing Elasticity Engine for Short-Term Rentals
#
# This notebook demonstrates the automated data ingestion, validation, and schema unification
# across three heterogeneous data streams:
# 1. **Inside Airbnb**: 7M+ record daily calendar panel, listing attributes, and municipal GeoJSON boundaries.
# 2. **NOAA Climate Data Online**: Meteorological feeds (TMAX, TMIN, PRCP, AWND) for weather cost shocks.
# 3. **Eventbrite Demand Shocks**: Curated high-attendance civic, entertainment, and sporting events.
#
# We also initialize an embedded **DuckDB OLAP engine** to execute analytical queries over parquet files.

# %%
import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb

# Add project root and scripts directory to sys.path
PROJECT_ROOT = Path(".").resolve().parent if Path(".").resolve().name == "notebooks" else Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils import setup_logger, load_config, DuckDBManager
from data_collection import InsideAirbnbCollector, NOAAWeatherCollector, EventbriteCollector, SyntheticDataGenerator

logger = setup_logger("nb01_ingestion")
config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
print(f"Loaded Project: {config['project']['name']} (Market: {config['project']['market']})")

# %% [markdown]
# ## 1. Ingestion / Data Lake Synthesis
# We verify if raw data is already staged in `data/raw`. If absent, we initialize the deterministic
# synthetic generator matching the full schema of Inside Airbnb, NOAA, and Eventbrite.

# %%
raw_airbnb_dir = PROJECT_ROOT / config["paths"]["raw_dir"] / "airbnb"
raw_noaa_dir = PROJECT_ROOT / config["paths"]["raw_dir"] / "noaa"
raw_events_dir = PROJECT_ROOT / config["paths"]["raw_dir"] / "events"

listings_raw_file = raw_airbnb_dir / "listings.csv"
calendar_raw_file = raw_airbnb_dir / "calendar.csv.gz"

if not listings_raw_file.exists() or not calendar_raw_file.exists():
    print("Raw files not detected. Generating reproducible benchmark market dataset...")
    gen = SyntheticDataGenerator(config, n_listings=1200, n_days=730)
    gen.generate()
    noaa = NOAAWeatherCollector(config)
    noaa.fetch_daily_weather()
    eb = EventbriteCollector(config)
    eb.fetch_events()
    print("Market synthesis complete.")
else:
    print(f"Raw data verified in {raw_airbnb_dir}")

# %% [markdown]
# ## 2. Inspection of Raw Data Feeds
# Let's inspect the structural schema and sample rows from listings, calendar, NOAA weather, and events.

# %%
# Inspect Listings
df_listings = pd.read_csv(listings_raw_file, nrows=500)
print(f"Listings Schema Overview ({len(df_listings.columns)} columns):")
print(df_listings[["id", "name", "neighbourhood_cleansed", "room_type", "price", "accommodates", "review_scores_rating"]].head())

# %%
# Inspect Daily Calendar
df_cal = pd.read_csv(calendar_raw_file, nrows=1000)
print("\nDaily Calendar Panel Sample:")
print(df_cal.head())

# %%
# Inspect NOAA Weather
df_weather = pd.read_csv(raw_noaa_dir / "noaa_daily_weather.csv")
print(f"\nNOAA Daily Weather Observations ({len(df_weather)} days):")
print(df_weather.head())

# %%
# Inspect Eventbrite Demand Shocks
df_events = pd.read_csv(raw_events_dir / "austin_events_calendar.csv")
print(f"\nCurated Demand Shock Events ({len(df_events)} events):")
print(df_events[["name", "start_date", "end_date", "expected_attendance", "category"]])

# %% [markdown]
# ## 3. DuckDB Embedded Lakehouse Initialization
# DuckDB delivers columnar vectorized querying over Parquet/CSV formats with zero database overhead.

# %%
duckdb_path = str(PROJECT_ROOT / config["paths"]["duckdb_path"])
os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)
conn = duckdb.connect(duckdb_path)

# Register tables directly from raw files
conn.execute(f"CREATE OR REPLACE VIEW raw_listings AS SELECT * FROM read_csv_auto('{listings_raw_file}');")
conn.execute(f"CREATE OR REPLACE VIEW raw_weather AS SELECT * FROM read_csv_auto('{raw_noaa_dir / 'noaa_daily_weather.csv'}');")
conn.execute(f"CREATE OR REPLACE VIEW raw_events AS SELECT * FROM read_csv_auto('{raw_events_dir / 'austin_events_calendar.csv'}');")

print("DuckDB views created successfully:")
views = conn.execute("SHOW TABLES;").fetchall()
for v in views:
    print(f" - {v[0]}")

# %%
# Query Summary Statistics using DuckDB SQL
res = conn.execute("""
    SELECT 
        room_type,
        COUNT(*) AS listing_count,
        ROUND(AVG(TRY_CAST(REPLACE(REPLACE(price, '$', ''), ',', '') AS DOUBLE)), 2) AS avg_raw_price,
        ROUND(AVG(accommodates), 1) AS avg_accommodates
    FROM raw_listings
    GROUP BY room_type
    ORDER BY listing_count DESC;
""").df()

print("Summary by Room Type:")
print(res)

# %%
conn.close()
print("Notebook 01 execution finished cleanly.")
