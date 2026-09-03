"""
Listing Market Segmentation Engine.
Applies unsupervised K-Means clustering across listing amenity profiles, luxury indices,
capacity, and spatial metrics to identify distinct customer willingness-to-pay tiers.
Produces segment definitions utilized for stratified elasticity estimation.
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import setup_logger, load_config, DuckDBManager

logger = setup_logger("segmentation")


class MarketSegmentationEngine:
    """Clustering engine for short-term rental property tier segmentation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_dir = Path(config["paths"]["models_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.proc_dir = Path(config["paths"]["processed_dir"])
        self.n_clusters = config["segmentation"].get("n_clusters", 4)
        self.feature_cols = config["segmentation"].get("features", [
            "accommodates", "bedrooms", "bathrooms", "amenity_luxury_index",
            "distance_to_downtown_km", "price_usd"
        ])
        self.cluster_names_map = config["segmentation"].get("cluster_names", {
            0: "Budget Urban Studio",
            1: "Midscale Standard Condo",
            2: "Premium Family Home",
            3: "Luxury Experiential Villa"
        })
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=15)

    def fit_predict(self, listings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fits K-Means model on normalized property attributes and assigns segment labels."""
        logger.info(f"Extracting features for {len(listings_df):,} listings: {self.feature_cols}")

        clean_df = listings_df.dropna(subset=self.feature_cols).copy()
        X = clean_df[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.kmeans.fit(X_scaled)
        labels = self.kmeans.labels_
        clean_df["cluster_id"] = labels

        # Compute diagnostic metrics
        sil_score = float(silhouette_score(X_scaled, labels))
        inertia = float(self.kmeans.inertia_)
        logger.info(f"K-Means (k={self.n_clusters}) - Silhouette Score: {sil_score:.3f}, Inertia: {inertia:,.1f}")

        # Order clusters monotonically by mean price for consistent human-interpretable labeling
        cluster_price_rank = clean_df.groupby("cluster_id")["price_usd"].mean().sort_values().index.tolist()
        label_remapping = {old_id: new_rank for new_rank, old_id in enumerate(cluster_price_rank)}

        clean_df["cluster_rank"] = clean_df["cluster_id"].map(label_remapping)
        clean_df["cluster_segment"] = clean_df["cluster_rank"].map(lambda r: self.cluster_names_map.get(r, f"Segment_{r}"))

        # Compute descriptive segment profiles
        profiles = {}
        for rank in range(self.n_clusters):
            seg_name = self.cluster_names_map.get(rank, f"Segment_{rank}")
            subset = clean_df[clean_df["cluster_rank"] == rank]
            profiles[seg_name] = {
                "count": int(len(subset)),
                "market_share_pct": float(round(len(subset) / len(clean_df) * 100.0, 2)),
                "mean_price_usd": float(round(subset["price_usd"].mean(), 2)),
                "median_price_usd": float(round(subset["price_usd"].median(), 2)),
                "mean_accommodates": float(round(subset["accommodates"].mean(), 2)),
                "mean_bedrooms": float(round(subset["bedrooms"].mean(), 2)),
                "mean_luxury_index": float(round(subset["amenity_luxury_index"].mean(), 3)),
                "mean_dist_downtown_km": float(round(subset["distance_to_downtown_km"].mean(), 2))
            }

        summary_metrics = {
            "n_clusters": self.n_clusters,
            "silhouette_score": sil_score,
            "inertia": inertia,
            "profiles": profiles
        }

        # Persist trained model artifacts
        scaler_path = self.models_dir / "segmentation_scaler.joblib"
        model_path = self.models_dir / "kmeans_segmentation.joblib"
        meta_path = self.models_dir / "segmentation_profiles.json"

        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.kmeans, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)

        logger.info(f"Saved clustering artifacts to {self.models_dir}")
        return clean_df, summary_metrics

    def annotate_panel(self, panel_df: pd.DataFrame, listings_clustered_df: pd.DataFrame) -> pd.DataFrame:
        """Propagates cluster segmentation assignments onto the temporal modeling panel."""
        logger.info("Propagating cluster segment labels into daily panel records...")
        seg_lookup = listings_clustered_df[["listing_id", "cluster_rank", "cluster_segment"]].drop_duplicates()
        annotated_panel = panel_df.merge(seg_lookup, on="listing_id", how="left")

        # Save annotated parquet
        out_path = self.proc_dir / "modeling_panel_segmented.parquet"
        annotated_panel.to_parquet(out_path, index=False)
        logger.info(f"Annotated segmented panel saved to {out_path} ({len(annotated_panel):,} rows)")

        # Save to DuckDB
        duckdb_path = self.config["paths"]["duckdb_path"]
        db = DuckDBManager(duckdb_path)
        db.register_df("panel_segmented_staging", annotated_panel)
        db.conn.execute("CREATE OR REPLACE TABLE modeling_panel_segmented AS SELECT * FROM panel_segmented_staging;")
        logger.info("DuckDB table 'modeling_panel_segmented' updated.")
        db.close()

        return annotated_panel


def main():
    parser = argparse.ArgumentParser(description="Market Segmentation Clustering")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    proc_dir = Path(config["paths"]["processed_dir"])
    listings_path = proc_dir / "listings_cleaned.parquet"
    panel_path = proc_dir / "modeling_panel_features.parquet"

    if not listings_path.exists() or not panel_path.exists():
        logger.error("Required preprocessed files missing. Run preprocessing and feature engineering first.")
        sys.exit(1)

    listings_df = pd.read_parquet(listings_path)
    panel_df = pd.read_parquet(panel_path)

    engine = MarketSegmentationEngine(config)
    clustered_listings, metrics = engine.fit_predict(listings_df)
    engine.annotate_panel(panel_df, clustered_listings)

    logger.info("Market segmentation executed successfully.")
    for seg, p in metrics["profiles"].items():
        logger.info(f"[{seg}] Share: {p['market_share_pct']}% | Mean Price: ${p['mean_price_usd']} | Accommodates: {p['mean_accommodates']}")


if __name__ == "__main__":
    main()
