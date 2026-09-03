"""
Utility functions and shared infrastructure for the Dynamic Pricing Elasticity Engine.
Includes logging, configuration management, DuckDB integration, spatial distance
calculations, and econometric evaluation metrics.
"""

import os
import sys
import logging
import yaml
import numpy as np
import pandas as pd
import duckdb
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path


def setup_logger(
    name: str = "pricing_elasticity",
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configures and returns a structured logger with standardized formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


logger = setup_logger()


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Safely loads and validates configuration dictionary from YAML file.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found at {config_path}. Falling back to default project root relative path.")
        # Attempt to find relative to parent directory
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / "config" / "config.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Configuration file cannot be located at {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across NumPy and standard libraries for empirical reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)


def haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float
) -> np.ndarray:
    """
    Vectorized calculation of Haversine great-circle distance between an array of
    coordinates (lat1, lon1) and a single target reference point (lat2, lon2) in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


class DuckDBManager:
    """
    High-performance wrapper for DuckDB analytical querying, data persistence,
    and Parquet/OLAP acceleration.
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            self.db_path = ":memory:"
        else:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.db_path = db_path
        self.conn = duckdb.connect(database=self.db_path)
        logger.info(f"Connected to DuckDB instance: {self.db_path}")

    def register_df(self, view_name: str, df: pd.DataFrame) -> None:
        """Registers a pandas DataFrame as a queryable virtual table in DuckDB."""
        self.conn.register(view_name, df)
        logger.debug(f"Registered DataFrame as view '{view_name}' ({len(df):,} rows)")

    def execute_query(self, query: str) -> pd.DataFrame:
        """Executes an OLAP SQL query and returns the result as a pandas DataFrame."""
        try:
            return self.conn.execute(query).df()
        except Exception as e:
            logger.error(f"Failed executing SQL query: {query}\nError: {e}")
            raise

    def export_table_to_parquet(self, table_name: str, output_path: str) -> None:
        """Exports a database table or view directly to an optimized Snappy-compressed Parquet file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        query = f"COPY {table_name} TO '{output_path}' (FORMAT PARQUET, COMPRESSION SNAPPY);"
        self.conn.execute(query)
        logger.info(f"Table '{table_name}' exported successfully to {output_path}")

    def close(self) -> None:
        """Closes the active database connection."""
        self.conn.close()
        logger.info("DuckDB connection closed.")


def calculate_price_elasticity(
    delta_quantity_pct: np.ndarray,
    delta_price_pct: np.ndarray
) -> np.ndarray:
    """
    Computes arc price elasticity of demand:
    epsilon = (% change in Quantity) / (% change in Price)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        elasticity = np.where(
            np.abs(delta_price_pct) > 1e-6,
            delta_quantity_pct / delta_price_pct,
            np.nan
        )
    return elasticity


def optimal_monopoly_price(
    marginal_cost: float,
    elasticity: float,
    min_price_floor: float = 20.0,
    max_price_cap: float = 2000.0
) -> float:
    """
    Calculates theoretically optimal price under standard microeconomic profit maximization:
    P* = MC * (elasticity / (1 + elasticity)) for elasticity < -1.
    Includes safeguard clipping to prevent pathological unbounded pricing.
    """
    if elasticity >= -1.0:
        # Inelastic demand regime: standard Lerner formula requires non-infinite cap
        # Price at upper bound or markup ceiling
        return max_price_cap

    markup_factor = elasticity / (1.0 + elasticity)
    p_star = marginal_cost * markup_factor
    return float(np.clip(p_star, min_price_floor, max_price_cap))


def calculate_econometric_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Computes standard empirical goodness-of-fit and forecast error metrics.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    # Avoid zero division in MAPE
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2)
    }


def validate_schema(df: pd.DataFrame, required_columns: List[str], df_name: str = "DataFrame") -> None:
    """
    Verifies that all required features exist in the target DataFrame.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing mandatory columns: {missing}")
    logger.debug(f"Schema validation passed for {df_name}. All {len(required_columns)} columns present.")
