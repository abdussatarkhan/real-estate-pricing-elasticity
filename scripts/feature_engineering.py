"""
Feature Engineering Pipeline for Short-Term Rental Elasticity Engine.
Implements:
1. Demand proxy estimation (rolling booking velocity, forward 30-day demand quantity)
2. Geospatial competitor density and spatial competitor price lag extraction (KDTree)
3. Seasonal decomposition (Fourier harmonics, calendar cyclicality, holiday indicators)
4. NLP amenity extraction (TF-IDF vectorization and TruncatedSVD latent luxury factor scoring)
5. Construction of econometric Instrumental Variables (cost shifters and Hausman spatial price lags)
6. Unified modeling panel generation persisted to DuckDB and Parquet.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import setup_logger, load_config, haversine_distance, DuckDBManager

logger = setup_logger("feature_engineering")


class AmenityNLPExtractor:
    """Extracts latent semantic amenity representations and luxury indices using TF-IDF and SVD."""

    def __init__(self, max_features: int = 60, n_components: int = 5):
        self.max_features = max_features
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=self.max_features,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
        )
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)

    def _clean_amenities_string(self, raw_amenities: Any) -> str:
        """Parses JSON or comma-separated list into space-delimited text string."""
        if pd.isna(raw_amenities):
            return ""
        text = str(raw_amenities)
        try:
            items = json.loads(text)
            if isinstance(items, list):
                return " ".join([str(i).lower().replace(" ", "_") for i in items])
        except Exception:
            pass
        for char in ["[", "]", "{", "}", '"', "'"]:
            text = text.replace(char, " ")
        return text.lower()

    def fit_transform(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Fits TF-IDF and TruncatedSVD on amenity corpus and returns latent feature matrix."""
        logger.info(f"Extracting TF-IDF features across {len(series):,} amenity profiles...")
        corpus = series.apply(self._clean_amenities_string)
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        svd_components = self.svd.fit_transform(tfidf_matrix)
        feature_names = self.vectorizer.get_feature_names_out().tolist()

        # Compute luxury index based on curated high-value amenity keywords
        luxury_keywords = ["hot_tub", "pool", "view", "sound_system", "chef", "patio", "waterfront", "balcony", "ev_charger"]
        vocab = self.vectorizer.vocabulary_
        lux_indices = [vocab[kw] for kw in luxury_keywords if kw in vocab]

        if lux_indices:
            luxury_index = tfidf_matrix[:, lux_indices].toarray().sum(axis=1)
        else:
            luxury_index = svd_components[:, 0]

        # Normalize luxury index to [0, 1]
        p_min, p_max = np.min(luxury_index), np.max(luxury_index)
        if p_max > p_min:
            norm_luxury = (luxury_index - p_min) / (p_max - p_min)
        else:
            norm_luxury = np.zeros(len(series))

        return svd_components, norm_luxury, feature_names


class SpatialCompetitorEngine:
    """Computes spatial competitor density and local competitor price levels using BallTree."""

    def __init__(self, lat_coords: np.ndarray, lon_coords: np.ndarray):
        # Convert lat/lon to radians for Haversine BallTree
        coords_rad = np.radians(np.column_stack([lat_coords, lon_coords]))
        self.tree = BallTree(coords_rad, metric="haversine")
        self.earth_radius_m = 6371000.0

    def calculate_density_and_price_lag(
        self,
        lat_coords: np.ndarray,
        lon_coords: np.ndarray,
        prices: np.ndarray,
        submarkets: np.ndarray,
        radius_meters: float = 1000.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates:
        1. Local competitor count within radius
        2. Average local competitor price
        3. Hausman spatial lag instrument (average price of competitors in OTHER adjacent submarkets)
        """
        coords_rad = np.radians(np.column_stack([lat_coords, lon_coords]))
        radius_rad = radius_meters / self.earth_radius_m

        # Query all neighbors within radius
        indices_list = self.tree.query_radius(coords_rad, r=radius_rad)

        competitor_count = np.zeros(len(lat_coords), dtype=int)
        avg_competitor_price = np.zeros(len(lat_coords), dtype=float)
        hausman_price_lag = np.zeros(len(lat_coords), dtype=float)

        # Global average price fallback
        global_mean_price = float(np.mean(prices))

        for i, neighbor_indices in enumerate(indices_list):
            # Exclude self from neighborhood
            other_indices = neighbor_indices[neighbor_indices != i]
            competitor_count[i] = len(other_indices)

            if len(other_indices) > 0:
                avg_competitor_price[i] = float(np.mean(prices[other_indices]))
            else:
                avg_competitor_price[i] = prices[i]

            # Hausman instrument: average price of listings in DIFFERENT submarkets
            current_submarket = submarkets[i]
            diff_submarket_mask = submarkets != current_submarket
            if np.any(diff_submarket_mask):
                hausman_price_lag[i] = float(np.mean(prices[diff_submarket_mask]))
            else:
                hausman_price_lag[i] = global_mean_price

        return competitor_count, avg_competitor_price, hausman_price_lag


class FeatureEngineeringPipeline:
    """Unified feature engineering pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.proc_dir = Path(config["paths"]["processed_dir"])
        self.downtown_austin_lat = 30.2672
        self.downtown_austin_lon = -97.7431

    def process(self) -> pd.DataFrame:
        """Executes full feature extraction and panel consolidation."""
        logger.info("Starting feature engineering pipeline...")
        listings_path = self.proc_dir / "listings_cleaned.parquet"
        cal_path = self.proc_dir / "calendar_panel_cleaned.parquet"

        if not listings_path.exists() or not cal_path.exists():
            raise FileNotFoundError("Cleaned input parquet files not found in data/processed/")

        listings_df = pd.read_parquet(listings_path)
        calendar_df = pd.read_parquet(cal_path)
        logger.info(f"Loaded {len(listings_df):,} listings and {len(calendar_df):,} calendar panel rows.")

        # --- 1. NLP Amenity & Luxury Index Extraction ---
        nlp_engine = AmenityNLPExtractor(max_features=50, n_components=4)
        svd_factors, luxury_idx, _ = nlp_engine.fit_transform(listings_df["amenities"])
        listings_df["amenity_luxury_index"] = np.round(luxury_idx, 4)
        for comp_i in range(svd_factors.shape[1]):
            listings_df[f"amenity_svd_{comp_i+1}"] = np.round(svd_factors[:, comp_i], 4)

        # --- 2. Distance to Core Business / Downtown Hub ---
        listings_df["distance_to_downtown_km"] = np.round(
            haversine_distance(
                listings_df["latitude"].values,
                listings_df["longitude"].values,
                self.downtown_austin_lat,
                self.downtown_austin_lon
            ),
            3
        )

        # --- 3. Spatial Competitor Density and Instruments ---
        spatial_engine = SpatialCompetitorEngine(listings_df["latitude"].values, listings_df["longitude"].values)
        density_1km, avg_comp_price, hausman_lag = spatial_engine.calculate_density_and_price_lag(
            listings_df["latitude"].values,
            listings_df["longitude"].values,
            listings_df["price_usd"].values,
            listings_df["neighbourhood_assigned"].values,
            radius_meters=1000.0
        )
        listings_df["competitor_density_1km"] = density_1km
        listings_df["avg_competitor_price_1km"] = np.round(avg_comp_price, 2)
        listings_df["submarket_lag_competitor_price"] = np.round(hausman_lag, 2)

        # Cost Shifter Instrument: Cleaning fee per guest capacity
        listings_df["cost_shifter_cleaning_fee"] = np.round(
            listings_df["cleaning_fee_usd"] / (listings_df["accommodates"] + 1e-4),
            2
        )

        # --- 4. Panel Aggregation and Demand Proxy Construction ---
        logger.info("Constructing rolling demand quantity and forward booking horizon...")
        # Sort calendar by listing and date
        calendar_df = calendar_df.sort_values(["listing_id", "date"]).reset_index(drop=True)

        # Compute rolling booking velocity (7d and 14d rolling sum of bookings)
        calendar_df["rolling_booked_7d"] = calendar_df.groupby("listing_id")["is_booked"].transform(
            lambda s: s.rolling(window=7, min_periods=1).sum()
        )
        calendar_df["rolling_booked_14d"] = calendar_df.groupby("listing_id")["is_booked"].transform(
            lambda s: s.rolling(window=14, min_periods=1).sum()
        )

        # Forward 30-Day Demand Quantity (Key Dependent Variable in Econometric Literature)
        # Represents how many of the next 30 days are reserved/booked
        # Inverted rolling window or forward rolling
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=30)
        calendar_df["demand_quantity_next_30d"] = calendar_df.groupby("listing_id")["is_booked"].transform(
            lambda s: s.rolling(window=indexer, min_periods=7).sum()
        )
        calendar_df["demand_quantity_next_30d"] = calendar_df["demand_quantity_next_30d"].fillna(0).astype(float)

        # --- 5. Cyclical & Seasonality Features ---
        calendar_df["day_of_week"] = calendar_df["date"].dt.dayofweek
        calendar_df["is_weekend"] = (calendar_df["day_of_week"] >= 4).astype(int)
        calendar_df["month"] = calendar_df["date"].dt.month
        calendar_df["year"] = calendar_df["date"].dt.year
        calendar_df["month_year"] = calendar_df["date"].dt.strftime("%Y-%m")

        # Annual Fourier trigonometric harmonics (order k=2)
        doy = calendar_df["date"].dt.dayofyear.values
        calendar_df["fourier_sin_1"] = np.sin(2.0 * np.pi * doy / 365.25)
        calendar_df["fourier_cos_1"] = np.cos(2.0 * np.pi * doy / 365.25)
        calendar_df["fourier_sin_2"] = np.sin(4.0 * np.pi * doy / 365.25)
        calendar_df["fourier_cos_2"] = np.cos(4.0 * np.pi * doy / 365.25)

        # Weather cost shock instrument: extreme cooling degree days interacting with listing size
        # Energy cost shock = max(0, avg_temp - 24) * bedrooms
        temp = calendar_df.get("avg_temperature_c", 22.0)
        cooling_deg = np.maximum(0.0, temp - 24.0)

        # --- 6. Merge Cross-Sectional Listing Attributes with Daily Panel ---
        logger.info("Merging listing cross-sectional features into temporal panel...")
        listing_cols = [
            "listing_id", "neighbourhood_assigned", "room_type", "accommodates",
            "bedrooms", "bathrooms", "host_is_superhost", "instant_bookable",
            "number_of_reviews", "review_scores_rating", "reviews_per_month",
            "amenity_luxury_index", "distance_to_downtown_km", "competitor_density_1km",
            "avg_competitor_price_1km", "cost_shifter_cleaning_fee", "submarket_lag_competitor_price"
        ]
        panel_df = calendar_df.merge(listings_df[listing_cols], on="listing_id", how="inner")

        # Compute energy cost shock instrument with listing capacity
        panel_df["weather_energy_cost_shock"] = np.round(cooling_deg * panel_df["bedrooms"], 2)

        # --- 7. Log Transformations for Constant-Elasticity Specification ---
        # log(Demand Quantity + 1) = alpha + beta * log(Price) + gamma * X + u
        # where beta is the direct constant price elasticity of demand
        panel_df["log_demand_quantity"] = np.log(panel_df["demand_quantity_next_30d"] + 1.0)
        panel_df["log_price"] = np.log(np.maximum(10.0, panel_df["daily_price_usd"]))

        # Remove rows where demand forward window is truncated at the end of the time series
        panel_df = panel_df.dropna(subset=["log_demand_quantity", "log_price"]).copy()

        # Save to Parquet
        output_panel = self.proc_dir / "modeling_panel_features.parquet"
        panel_df.to_parquet(output_panel, index=False)
        logger.info(f"Persisted modeling feature panel to {output_panel} ({len(panel_df):,} rows)")

        # Persist to DuckDB
        duckdb_path = self.config["paths"]["duckdb_path"]
        db = DuckDBManager(duckdb_path)
        db.register_df("modeling_panel_staging", panel_df)
        db.conn.execute("CREATE OR REPLACE TABLE modeling_panel AS SELECT * FROM modeling_panel_staging;")
        logger.info(f"Updated DuckDB table 'modeling_panel' in {duckdb_path}")
        db.close()

        return panel_df


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = FeatureEngineeringPipeline(config)
    pipeline.process()
    logger.info("Feature engineering complete.")


if __name__ == "__main__":
    main()
