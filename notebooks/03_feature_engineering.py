# %% [markdown]
# # Notebook 03: Feature Engineering & Instrumental Variable Construction
#
# ### Dynamic Pricing Elasticity Engine for Short-Term Rentals
#
# This notebook details the transformation of raw spatial, temporal, and textual data into
# econometric regressors and exogenous instruments:
# 1. **Demand Proxies**: Rolling booking velocities (7d, 14d) and forward 30-day demand quantity.
# 2. **Geospatial Competitor Density**: KDTree / BallTree queries for competitor counts and local price benchmarks.
# 3. **NLP Amenity Embeddings**: TF-IDF tokenization and TruncatedSVD decomposition into latent luxury scores.
# 4. **Instrumental Variables Construction**:
#    - Cost Shifter: Cleaning fee per guest capacity.
#    - Hausman Spatial Lag: Competitor pricing in distinct adjacent submarkets.
#    - Meteorological Utility Shock: Extreme cooling degree days interacting with property size.

# %%
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(".").resolve().parent if Path(".").resolve().name == "notebooks" else Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils import setup_logger, load_config
from feature_engineering import FeatureEngineeringPipeline, AmenityNLPExtractor, SpatialCompetitorEngine

config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
proc_dir = PROJECT_ROOT / config["paths"]["processed_dir"]

panel_path = proc_dir / "modeling_panel_features.parquet"
if not panel_path.exists():
    print("Running feature engineering pipeline to generate feature panel...")
    fe_pipe = FeatureEngineeringPipeline(config)
    df_panel = fe_pipe.process()
else:
    df_panel = pd.read_parquet(panel_path)

print(f"Feature panel loaded successfully: {df_panel.shape[0]:,} rows x {df_panel.shape[1]} columns")

# %% [markdown]
# ## 1. NLP Amenity Extraction & Latent Luxury Index
# Textual amenities (e.g., 'hot tub', 'pool', 'designer kitchen', 'EV charger') convey substantial
# unobserved quality that affects both pricing and demand.
# We transform these into a continuous Luxury Index in $[0, 1]$.

# %%
listings_path = proc_dir / "listings_cleaned.parquet"
df_listings = pd.read_parquet(listings_path)

nlp_extractor = AmenityNLPExtractor(max_features=50, n_components=4)
svd_feats, luxury_idx, vocab = nlp_extractor.fit_transform(df_listings["amenities"])

df_listings["amenity_luxury_index"] = luxury_idx

plt.figure(figsize=(10, 4))
sns.histplot(luxury_idx, bins=30, kde=True, color="#4c72b0")
plt.title("Distribution of Derived Amenity Luxury Index", fontsize=13, fontweight="bold")
plt.xlabel("Luxury Score (Normalized [0, 1])")
plt.ylabel("Listing Count")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Geospatial Spatial Competition & Downtown Proximity
# Competitor density strongly conditions listing market power.
# We visualize how competitor density within 1km relates to distance from the downtown core.

# %%
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df_listings,
    x="distance_to_downtown_km",
    y="competitor_density_1km",
    hue="neighbourhood_assigned",
    alpha=0.75,
    s=50
)
plt.title("Spatial Competitor Density vs. Distance to Downtown Austin", fontsize=13, fontweight="bold")
plt.xlabel("Distance to Downtown (km)")
plt.ylabel("Competitor Listings within 1km Radius")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Instrumental Variables: Relevance & Exclusion Intuition
# To obtain unbiased price elasticity $\beta$, an instrument $Z$ must satisfy:
# 1. **Instrument Relevance**: $\text{Cov}(Z, \log(P)) \neq 0$ (strongly correlated with price).
# 2. **Instrument Exogeneity (Exclusion Restriction)**: $\text{Cov}(Z, u) = 0$ (affects demand ONLY through price).

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Instrument 1: Cleaning fee per guest
sns.regplot(
    data=df_listings.sample(min(800, len(df_listings))),
    x="cost_shifter_cleaning_fee",
    y="price_usd",
    ax=axes[0],
    scatter_kws={"alpha": 0.4, "color": "#1f77b4"},
    line_kws={"color": "red"}
)
axes[0].set_title("Instrument 1: Cost Shifter (Cleaning Fee)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Cleaning Fee per Guest ($)")
axes[0].set_ylabel("Listing Price ($)")

# Instrument 2: Hausman Spatial Price Lag
sns.regplot(
    data=df_listings.sample(min(800, len(df_listings))),
    x="submarket_lag_competitor_price",
    y="price_usd",
    ax=axes[1],
    scatter_kws={"alpha": 0.4, "color": "#2ca02c"},
    line_kws={"color": "red"}
)
axes[1].set_title("Instrument 2: Hausman Spatial Price Lag", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Adjacent Submarket Average Price ($)")
axes[1].set_ylabel("Listing Price ($)")

# Instrument 3: Weather Cooling Degree Shock
sample_panel = df_panel.sample(min(1000, len(df_panel)))
sns.regplot(
    data=sample_panel,
    x="weather_energy_cost_shock",
    y="daily_price_usd",
    ax=axes[2],
    scatter_kws={"alpha": 0.4, "color": "#ff7f0e"},
    line_kws={"color": "red"}
)
axes[2].set_title("Instrument 3: Weather Energy Cost Shock", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Extreme Temperature x Size Interaction")
axes[2].set_ylabel("Daily Listing Price ($)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Dependent Variable: Forward 30-Day Demand Quantity
# In econometric rental studies, demand is measured as the number of days booked in the forward booking window.

# %%
plt.figure(figsize=(10, 4))
sns.histplot(df_panel["demand_quantity_next_30d"], bins=31, color="#2b5c8f", kde=False)
plt.title("Forward 30-Day Demand Quantity Distribution (Days Booked)", fontsize=13, fontweight="bold")
plt.xlabel("Days Booked in Forward 30-Day Horizon")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

print("Feature engineering walkthrough completed.")
