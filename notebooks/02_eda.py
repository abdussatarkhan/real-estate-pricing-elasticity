# %% [markdown]
# # Notebook 02: Exploratory Data Analysis & Empirical Market Dynamics
#
# ### Dynamic Pricing Elasticity Engine for Short-Term Rentals
#
# In this notebook, we perform an in-depth empirical exploration of short-term rental market data:
# 1. **Price Distributions**: Skewness, long-tail dynamics, and log-normal properties across property types.
# 2. **Geospatial & Neighborhood Dispersion**: Submarket price gradients from downtown Austin outward.
# 3. **Temporal Seasonality**: Day-of-week weekend premiums, annual harmonic cycles, and event spikes (SXSW, F1).
# 4. **Occupancy & Demand Correlation**: Naive relationship between price and booking probability (highlighting simultaneity).

# %%
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(".").resolve().parent if Path(".").resolve().name == "notebooks" else Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils import setup_logger, load_config

config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
proc_dir = PROJECT_ROOT / config["paths"]["processed_dir"]

# Load cleaned listings and sample calendar panel
listings_path = proc_dir / "listings_cleaned.parquet"
cal_path = proc_dir / "calendar_panel_cleaned.parquet"

if not listings_path.exists():
    print("Preprocessing datasets first...")
    from preprocessing import PreprocessingPipeline
    pipeline = PreprocessingPipeline(config)
    df_listings = pipeline.clean_listings()
    df_cal = pipeline.clean_calendar_and_merge(df_listings)
else:
    df_listings = pd.read_parquet(listings_path)
    df_cal = pd.read_parquet(cal_path)

print(f"Loaded {len(df_listings):,} listings and {len(df_cal):,} calendar records.")

# %% [markdown]
# ## 1. Price Distribution & Log-Transformation
# Real estate short-term rental prices typically exhibit strong right-skewness.
# Econometric demand estimation utilizes logarithmic transformations $\log(P)$ to stabilize variance
# and directly interpret regression coefficients as constant price elasticity $\epsilon$.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw price distribution
sns.histplot(df_listings["price_usd"], bins=40, kde=True, ax=axes[0], color="#2b5c8f")
axes[0].set_title("Raw Price Distribution ($ USD)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Nightly Price ($)")
axes[0].set_ylabel("Count")

# Log-transformed price distribution
log_p = np.log(df_listings["price_usd"])
sns.histplot(log_p, bins=40, kde=True, ax=axes[1], color="#e26d5c")
axes[1].set_title("Log-Transformed Price Distribution log(P)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("log(Price)")
axes[1].set_ylabel("Density")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Submarket Price Gradients Across Neighborhoods
# We examine how median prices vary across Austin's primary municipal submarkets.

# %%
nb_summary = df_listings.groupby("neighbourhood_assigned").agg(
    median_price=("price_usd", "median"),
    mean_price=("price_usd", "mean"),
    listing_count=("listing_id", "count"),
    mean_accommodates=("accommodates", "mean"),
    mean_rating=("review_scores_rating", "mean")
).sort_values("median_price", ascending=False).reset_index()

print("Submarket Summary:")
print(nb_summary)

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df_listings,
    x="neighbourhood_assigned",
    y="price_usd",
    order=nb_summary["neighbourhood_assigned"],
    palette="Blues_r"
)
plt.title("Nightly Price Variation by Austin Neighborhood", fontsize=14, fontweight="bold")
plt.xlabel("Neighborhood")
plt.ylabel("Nightly Price ($ USD)")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 800)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Temporal Demand Cycles & Event Shocks
# We analyze the daily price and booking rate timeline to identify weekend premiums
# and major event demand surges (SXSW in March, ACL/F1 in October).

# %%
daily_agg = df_cal.groupby("date").agg(
    avg_price=("daily_price_usd", "mean"),
    booking_rate=("is_booked", "mean"),
    event_shock=("event_intensity_score", "mean")
).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 6))

color = "#1f77b4"
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Average Nightly Price ($)", color=color, fontsize=12)
ax1.plot(daily_agg["date"], daily_agg["avg_price"], color=color, linewidth=1.8, label="Avg Price ($)")
ax1.tick_params(axis="y", labelcolor=color)

# Secondary axis for booking rate
ax2 = ax1.twinx()
color = "#2ca02c"
ax2.set_ylabel("Market Booking Rate (%)", color=color, fontsize=12)
ax2.plot(daily_agg["date"], daily_agg["booking_rate"] * 100.0, color=color, linewidth=1.5, alpha=0.75, linestyle="--", label="Booking Rate (%)")
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Market Equilibrium Timeline: Price vs. Occupancy Rate with Demand Shocks", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Weekend vs. Weekday Premium
# Austin's entertainment and leisure-driven economy causes significant weekend premiums.

# %%
df_cal["day_of_week"] = pd.to_datetime(df_cal["date"]).dt.day_name()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_agg = df_cal.groupby("day_of_week")["daily_price_usd"].mean().reindex(dow_order)

plt.figure(figsize=(10, 4))
sns.barplot(x=dow_agg.index, y=dow_agg.values, palette="viridis")
plt.title("Day of Week Nightly Price Premium", fontsize=13, fontweight="bold")
plt.ylabel("Average Price ($ USD)")
plt.ylim(dow_agg.min() * 0.9, dow_agg.max() * 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Correlation Heatmap & The Naive Simultaneity Puzzle
# A naive regression of bookings on price often yields a positive or near-zero coefficient
# because high demand drives hosts to raise prices (simultaneity).
# This motivates Instrumental Variable (2SLS) estimation.

# %%
numeric_cols = ["price_usd", "accommodates", "bedrooms", "bathrooms", "review_scores_rating", "number_of_reviews"]
corr = df_listings[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
plt.title("Cross-Sectional Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print("EDA completed successfully.")
