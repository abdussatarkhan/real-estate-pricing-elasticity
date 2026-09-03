# %% [markdown]
# # Notebook 04: Instrumental Variable Regression (2SLS) & Econometric Diagnostics
#
# ### Dynamic Pricing Elasticity Engine for Short-Term Rentals
#
# This notebook delivers the core econometric identification analysis:
# 1. **The Endogeneity Problem**: Price is endogenous due to unobserved demand shocks ($Cov(\log(P), u) \neq 0$).
# 2. **Naive OLS vs. 2SLS**: Demonstrating severe attenuation / simultaneity bias in OLS.
# 3. **Instrument Diagnostics**:
#    - First-Stage Relevance ($F > 10$ rule of thumb).
#    - Durbin-Wu-Hausman Endogeneity Test ($p < 0.05$ confirms OLS inconsistency).
#    - Sargan-Hansen Overidentifying Restrictions J-Test ($p > 0.05$ confirms instrument validity).
# 4. **Stratified Segment Elasticities**: Quantifying price sensitivity across budget, midscale, family, and luxury tiers.

# %%
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(".").resolve().parent if Path(".").resolve().name == "notebooks" else Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils import setup_logger, load_config
from elasticity_modeling import ElasticityModelTrainer, EconometricDiagnostics
from segmentation import MarketSegmentationEngine

config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
proc_dir = PROJECT_ROOT / config["paths"]["processed_dir"]

panel_path = proc_dir / "modeling_panel_features.parquet"
if not panel_path.exists():
    from feature_engineering import FeatureEngineeringPipeline
    fe_pipe = FeatureEngineeringPipeline(config)
    fe_pipe.process()

df_panel = pd.read_parquet(panel_path)
print(f"Loaded Modeling Panel: {df_panel.shape[0]:,} observations")

# %% [markdown]
# ## 1. Model Setup & Specification
#
# We estimate the constant-elasticity log-log demand equation:
# $$\log(Q_{it} + 1) = \alpha + \beta \log(P_{it}) + \mathbf{X}_{it}'\boldsymbol{\gamma} + \delta_t + u_{it}$$
#
# Where:
# - $Q_{it}$: Forward 30-day booking quantity
# - $P_{it}$: Daily listing price
# - $\beta$: Price elasticity of demand (percentage change in bookings per 1% change in price)
# - $\mathbf{X}_{it}$: Exogenous controls (capacity, bedrooms, ratings, weather, events, amenities)
# - Instruments $\mathbf{Z}_{it}$: Cleaning fee per guest, adjacent submarket lag price, weather energy cost shock

# %%
trainer = ElasticityModelTrainer(config)
y, X_ctrl, endog, instruments = trainer.prepare_data(df_panel)

print(f"Dependent variable: {trainer.dep_var}")
print(f"Endogenous regressor: {trainer.endog_var}")
print(f"Exogenous instruments ({len(trainer.instrument_names)}): {trainer.instrument_names}")
print(f"Exogenous controls ({len(trainer.control_names)}): {trainer.control_names}")

# %% [markdown]
# ## 2. Stage 1: OLS Benchmark (Simultaneity Bias)
# Naive OLS estimates understate true consumer price sensitivity because hosts raise prices
# when unobserved local demand surges.

# %%
ols_model = trainer.fit_ols_benchmark(y, X_ctrl, endog)
beta_ols = ols_model.params[trainer.endog_var]
se_ols = ols_model.bse[trainer.endog_var]
print(f"Naive OLS Price Elasticity: beta = {beta_ols:.4f} (SE: {se_ols:.4f})")

# %% [markdown]
# ## 3. Stage 2: First-Stage Regression & Instrument Relevance
# We evaluate the first stage: $\log(P_{it}) = \pi_0 + \mathbf{Z}_{it}'\boldsymbol{\pi}_1 + \mathbf{X}_{it}'\boldsymbol{\pi}_2 + v_{it}$

# %%
Z_stage1 = sm.add_constant(pd.concat([instruments, X_ctrl], axis=1))
first_stage = sm.OLS(endog, Z_stage1).fit(cov_type="HC1")
f_stat, f_pval = EconometricDiagnostics.first_stage_f_statistic(first_stage, trainer.instrument_names)

print("=" * 60)
print("FIRST-STAGE INSTRUMENT RELEVANCE REGRESSION")
print("=" * 60)
for inst in trainer.instrument_names:
    coef = first_stage.params[inst]
    se = first_stage.bse[inst]
    t = coef / se
    print(f"Instrument '{inst}': coef={coef:.4f}, SE={se:.4f}, t-stat={t:.2f}")

print("-" * 60)
print(f"Excluded Instruments Joint F-Statistic: {f_stat:.2f}")
print(f"p-value: {f_pval:.4e}")
if f_stat > 10.0:
    print("PASS: First-stage F-statistic exceeds Stock-Yogo rule of thumb (F > 10).")
else:
    print("WARNING: Weak instrument warning (F < 10).")
print("=" * 60)

# %% [markdown]
# ## 4. Two-Stage Least Squares (2SLS) Structural Estimation & Diagnostics
# Using instrumented variation in price, we recover the causal elasticity parameter $\beta_{2SLS}$.

# %%
iv_model, _, diagnostics = trainer.fit_2sls_instrumental_variables(y, X_ctrl, endog, instruments)

beta_2sls = iv_model.params[trainer.endog_var]
se_2sls = iv_model.bse[trainer.endog_var]

print("\n" + "=" * 60)
print("STRUCTURAL 2SLS DEMAND ESTIMATE")
print("=" * 60)
print(f"Causal Price Elasticity (beta_2SLS): {beta_2sls:.4f}")
print(f"Standard Error:                    {se_2sls:.4f}")
print(f"95% Confidence Interval:          [{beta_2sls - 1.96*se_2sls:.4f}, {beta_2sls + 1.96*se_2sls:.4f}]")
print("-" * 60)
print(f"Hausman Endogeneity Test F-Stat:   {diagnostics['hausman_endogeneity_test']['hausman_f_stat']:.3f} (p = {diagnostics['hausman_endogeneity_test']['hausman_p_val']:.4e})")
print(f"Sargan Over-ID J-Stat:             {diagnostics['sargan_overidentification_test'].get('sargan_stat', np.nan):.3f} (p = {diagnostics['sargan_overidentification_test'].get('p_value', np.nan):.4f})")
print("=" * 60)

# %% [markdown]
# ## 5. Comparison: OLS vs. 2SLS Point Estimates
# Notice how the 2SLS estimate reveals substantially greater price sensitivity ($|\beta_{2SLS}| > |\beta_{OLS}|$).
# Naive OLS was contaminated by positive demand-side correlation.

# %%
comp_df = pd.DataFrame({
    "Estimator": ["Naive OLS", "Structural 2SLS (IV)"],
    "Elasticity": [beta_ols, beta_2sls],
    "Std_Error": [se_ols, se_2sls],
    "CI_Lower": [beta_ols - 1.96 * se_ols, beta_2sls - 1.96 * se_2sls],
    "CI_Upper": [beta_ols + 1.96 * se_ols, beta_2sls + 1.96 * se_2sls]
})
print("Estimation Comparison Table:")
print(comp_df)

plt.figure(figsize=(7, 4))
plt.errorbar(
    x=comp_df["Elasticity"],
    y=comp_df["Estimator"],
    xerr=1.96 * comp_df["Std_Error"],
    fmt="o",
    color="#1f77b4",
    ecolor="#d62728",
    elinewidth=2.5,
    capsize=6,
    markersize=8
)
plt.axvline(0, color="gray", linestyle="--", alpha=0.7)
plt.title("Omitted Variable Bias: Naive OLS vs. Instrumental Variables 2SLS", fontsize=12, fontweight="bold")
plt.xlabel("Estimated Price Elasticity Coefficient (beta)")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Stratified Elasticity by Market Segment
# Price elasticity varies substantially across property types:
# Luxury listings cater to price-insensitive travelers, whereas budget urban units face highly elastic substitution.

# %%
listings_path = proc_dir / "listings_cleaned.parquet"
df_listings = pd.read_parquet(listings_path)
seg_engine = MarketSegmentationEngine(config)
clustered_listings, _ = seg_engine.fit_predict(df_listings)
annotated_panel = seg_engine.annotate_panel(df_panel, clustered_listings)

segment_results = trainer.fit_segment_elasticities(annotated_panel)
seg_df = pd.DataFrame([
    {
        "Segment": k,
        "Elasticity": v["elasticity"],
        "SE": v["std_err"],
        "CI_Low": v["ci_95"][0],
        "CI_High": v["ci_95"][1],
        "N": v["n_obs"]
    }
    for k, v in segment_results.items()
]).sort_values("Elasticity")

print("\nStratified Segment Elasticity Table:")
print(seg_df)

plt.figure(figsize=(9, 4))
sns.barplot(data=seg_df, x="Segment", y="Elasticity", palette="mako")
plt.title("Price Elasticity Across Property Market Segments", fontsize=13, fontweight="bold")
plt.ylabel("Elasticity (epsilon)")
plt.xticks(rotation=15, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

print("Econometric modeling walkthrough completed.")
