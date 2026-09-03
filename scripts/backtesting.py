"""
Counterfactual Backtesting and Revenue Uplift Simulation Engine.
Implements:
1. Microeconomic dynamic pricing policy optimization using structural elasticity estimates
2. Simulation of counterfactual demand under elasticity-adjusted prices
3. Holdout period (2024) performance evaluation
4. Bootstrap confidence intervals for aggregate revenue uplift
5. Submarket and listing-tier uplift decomposition.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import setup_logger, load_config, DuckDBManager, set_seed

logger = setup_logger("backtesting")


class DynamicPricingPolicy:
    """Computes elasticity-calibrated optimal price recommendations."""

    def __init__(
        self,
        segment_elasticities: Dict[str, float],
        marginal_cost_ratio: float = 0.18,
        lower_bound_ratio: float = 0.75,
        upper_bound_ratio: float = 1.35
    ):
        self.segment_elasticities = segment_elasticities
        self.marginal_cost_ratio = marginal_cost_ratio
        self.lower_bound_ratio = lower_bound_ratio
        self.upper_bound_ratio = upper_bound_ratio
        self.default_elasticity = -1.45

    def recommend_price(
        self,
        base_price: float,
        segment: str,
        event_intensity: float,
        is_weekend: int,
        temp_shock: float
    ) -> float:
        """
        Calculates optimal dynamic price recommendation:
        Adjusts price relative to demand surge factors weighted by price sensitivity.
        """
        eps = self.segment_elasticities.get(segment, self.default_elasticity)

        # Baseline demand shift factor from exogenous shocks
        # High event intensity & weekend shift demand outward
        demand_shift = 0.12 * event_intensity + 0.15 * is_weekend + 0.05 * temp_shock

        # Inverse elasticity rule: in more inelastic segments (|eps| smaller),
        # host can exploit positive demand shifts with higher markups
        markup_multiplier = 1.0 + (demand_shift / abs(eps))

        # Soft demand periods: discount price to stimulate booking volume
        if demand_shift <= 0:
            discount_incentive = -0.08 / abs(eps)
            markup_multiplier = 1.0 + discount_incentive

        # Bound recommended price within policy guardrails [lower_bound, upper_bound]
        p_opt = base_price * markup_multiplier
        p_min = base_price * self.lower_bound_ratio
        p_max = base_price * self.upper_bound_ratio

        return float(np.clip(p_opt, p_min, p_max))


class BacktestingEngine:
    """Simulates counterfactual market outcomes and computes statistical revenue uplift."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.proc_dir = Path(config["paths"]["processed_dir"])
        self.models_dir = Path(config["paths"]["models_dir"])
        self.holdout_start = config["backtesting"].get("holdout_start", "2024-01-01")
        self.holdout_end = config["backtesting"].get("holdout_end", "2024-12-31")
        self.mc_ratio = config["backtesting"].get("marginal_cost_ratio", 0.18)
        self.lower_ratio = config["backtesting"]["price_adjustment_bounds"].get("lower_bound", 0.75)
        self.upper_ratio = config["backtesting"]["price_adjustment_bounds"].get("upper_bound", 1.35)
        self.n_bootstraps = config["backtesting"].get("bootstrap_iterations", 500)

    def load_holdout_panel(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Loads segmented panel and extracts 2024 out-of-sample holdout test partition."""
        panel_path = self.proc_dir / "modeling_panel_segmented.parquet"
        if not panel_path.exists():
            panel_path = self.proc_dir / "modeling_panel_features.parquet"
        if not panel_path.exists():
            raise FileNotFoundError(f"Segmented feature panel not found in {self.proc_dir}")

        df = pd.read_parquet(panel_path)
        df["date"] = pd.to_datetime(df["date"])

        # Filter holdout period
        holdout_mask = (df["date"] >= pd.to_datetime(self.holdout_start)) & (df["date"] <= pd.to_datetime(self.holdout_end))
        holdout_df = df[holdout_mask].copy()
        logger.info(f"Filtered 2024 holdout dataset: {len(holdout_df):,} daily listing observations.")

        # Load segment elasticities from econometric results if available
        elasticity_meta = self.models_dir / "elasticity_estimation_results.json"
        segment_elasticities = {}
        if elasticity_meta.exists():
            with open(elasticity_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            seg_data = meta.get("segment_elasticities", {})
            for k, v in seg_data.items():
                segment_elasticities[k] = v.get("elasticity", -1.45)
            # Global fallback
            default_eps = meta.get("iv_2sls", {}).get("beta_price", -1.45)
        else:
            default_eps = -1.45
            segment_elasticities = {
                "Budget Urban Studio": -1.82,
                "Midscale Standard Condo": -1.48,
                "Premium Family Home": -1.25,
                "Luxury Experiential Villa": -0.92
            }

        return holdout_df, segment_elasticities

    def run_simulation(self, holdout_df: pd.DataFrame, segment_elasticities: Dict[str, float]) -> pd.DataFrame:
        """Simulates counterfactual booking outcomes and revenues."""
        logger.info("Executing counterfactual dynamic pricing simulation...")
        policy = DynamicPricingPolicy(
            segment_elasticities=segment_elasticities,
            marginal_cost_ratio=self.mc_ratio,
            lower_bound_ratio=self.lower_ratio,
            upper_bound_ratio=self.upper_ratio
        )

        sim_df = holdout_df.copy()

        # Extract parameters for vectorized computation
        base_prices = sim_df["daily_price_usd"].values
        actual_booked = sim_df["is_booked"].values
        events = sim_df.get("event_intensity_score", pd.Series(0, index=sim_df.index)).values
        weekends = sim_df.get("is_weekend", pd.Series(0, index=sim_df.index)).values
        temps = sim_df.get("avg_temperature_c", pd.Series(22.0, index=sim_df.index)).values
        temp_shocks = np.clip((temps - 24.0) / 10.0, 0, 1.5)

        segments = sim_df.get("cluster_segment", pd.Series("Standard", index=sim_df.index)).values

        # Generate recommended prices
        rec_prices = np.zeros(len(sim_df))
        elasticities = np.zeros(len(sim_df))

        for i in range(len(sim_df)):
            seg = str(segments[i])
            rec_p = policy.recommend_price(
                base_price=base_prices[i],
                segment=seg,
                event_intensity=events[i],
                is_weekend=weekends[i],
                temp_shock=temp_shocks[i]
            )
            rec_prices[i] = rec_p
            elasticities[i] = segment_elasticities.get(seg, -1.45)

        sim_df["recommended_price_usd"] = np.round(rec_prices, 2)
        sim_df["price_ratio"] = rec_prices / (base_prices + 1e-4)

        # Constant elasticity demand response: Q_cf = Q_actual * (P_opt / P_actual) ^ epsilon
        # For booked units: booking probability is 1.0; price change alters probability
        # For unbooked units: base probability ~0.35; price change alters booking propensity
        base_prob = np.where(actual_booked == 1, 0.85, 0.25)
        cf_prob = base_prob * (sim_df["price_ratio"].values ** elasticities)
        cf_prob = np.clip(cf_prob, 0.02, 0.98)

        # Expected revenues
        sim_df["actual_revenue_usd"] = np.where(actual_booked == 1, base_prices, 0.0)
        sim_df["counterfactual_expected_revenue_usd"] = np.round(rec_prices * cf_prob, 2)

        return sim_df

    def evaluate_uplift(self, sim_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates total revenue uplift and bootstrap confidence intervals."""
        total_actual_rev = float(np.sum(sim_df["actual_revenue_usd"]))
        total_cf_rev = float(np.sum(sim_df["counterfactual_expected_revenue_usd"]))
        dollar_gain = total_cf_rev - total_actual_rev
        pct_gain = (dollar_gain / (total_actual_rev + 1e-5)) * 100.0

        logger.info(f"Actual Holdout Revenue:         ${total_actual_rev:,.2f}")
        logger.info(f"Counterfactual Engine Revenue:  ${total_cf_rev:,.2f}")
        logger.info(f"Net Revenue Uplift:             +${dollar_gain:,.2f} (+{pct_gain:.2f}%)")

        # Bootstrap Confidence Intervals
        set_seed(42)
        logger.info(f"Bootstrapping confidence intervals ({self.n_bootstraps} iterations)...")
        n = len(sim_df)
        boot_pcts = []
        act_rev_arr = sim_df["actual_revenue_usd"].values
        cf_rev_arr = sim_df["counterfactual_expected_revenue_usd"].values

        for _ in range(self.n_bootstraps):
            sample_idx = np.random.randint(0, n, size=n)
            s_act = np.sum(act_rev_arr[sample_idx])
            s_cf = np.sum(cf_rev_arr[sample_idx])
            if s_act > 0:
                boot_pcts.append((s_cf - s_act) / s_act * 100.0)

        ci_low = float(np.percentile(boot_pcts, 2.5))
        ci_high = float(np.percentile(boot_pcts, 97.5))
        logger.info(f"95% Bootstrap CI for Uplift: [{ci_low:.2f}%, {ci_high:.2f}%]")

        # Segment-level decomposition
        segment_breakdown = {}
        if "cluster_segment" in sim_df.columns:
            for seg, group in sim_df.groupby("cluster_segment"):
                s_act = float(np.sum(group["actual_revenue_usd"]))
                s_cf = float(np.sum(group["counterfactual_expected_revenue_usd"]))
                s_gain = s_cf - s_act
                s_pct = (s_gain / (s_act + 1e-5)) * 100.0
                segment_breakdown[str(seg)] = {
                    "actual_revenue_usd": round(s_act, 2),
                    "counterfactual_revenue_usd": round(s_cf, 2),
                    "uplift_usd": round(s_gain, 2),
                    "uplift_pct": round(s_pct, 2)
                }

        results = {
            "holdout_period": f"{self.holdout_start} to {self.holdout_end}",
            "total_observations": int(len(sim_df)),
            "actual_total_revenue_usd": round(total_actual_rev, 2),
            "counterfactual_total_revenue_usd": round(total_cf_rev, 2),
            "net_uplift_dollars": round(dollar_gain, 2),
            "net_uplift_percent": round(pct_gain, 2),
            "bootstrap_ci_95": [round(ci_low, 2), round(ci_high, 2)],
            "segment_breakdown": segment_breakdown
        }

        # Persist summary
        summary_path = self.models_dir / "backtesting_revenue_uplift.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Backtesting summary written to {summary_path}")

        # Persist simulation panel
        sim_out_parquet = self.proc_dir / "backtest_simulation_results.parquet"
        sim_df.to_parquet(sim_out_parquet, index=False)
        logger.info(f"Simulation panel written to {sim_out_parquet}")

        # Update DuckDB
        duckdb_path = self.config["paths"]["duckdb_path"]
        db = DuckDBManager(duckdb_path)
        db.register_df("sim_results_staging", sim_df)
        db.conn.execute("CREATE OR REPLACE TABLE backtest_results AS SELECT * FROM sim_results_staging;")
        logger.info("DuckDB table 'backtest_results' created.")
        db.close()

        return results


def main():
    parser = argparse.ArgumentParser(description="Counterfactual Pricing Backtesting Engine")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = BacktestingEngine(config)

    holdout_df, segment_elasticities = engine.load_holdout_panel()
    sim_df = engine.run_simulation(holdout_df, segment_elasticities)
    results = engine.evaluate_uplift(sim_df)

    logger.info("Backtesting and uplift analysis completed successfully.")


if __name__ == "__main__":
    main()
