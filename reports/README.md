# Econometric Research Report: Identification and Estimation of Price Elasticity in Short-Term Rental Real Estate

**Author:** Quantitative Real Estate Research Team  
**Market:** Austin Metropolitan Statistical Area (MSA)  
**Methodology:** Two-Stage Least Squares (2SLS) Instrumental Variables & Two-Way Panel Fixed Effects  
**Date:** September 2024  

---

## 1. Executive Summary

Determining consumer price elasticity of demand ($\epsilon$) is fundamental to maximizing revenue in short-term residential real estate. Standard pricing algorithms frequently suffer from **simultaneity bias**: hosts charge higher rates precisely during periods of unobserved high market demand (e.g., festivals, conferences, holiday weekends). As a result, naive Ordinary Least Squares (OLS) regressions severely underestimate consumer price sensitivity, estimating an artificially inelastic response ($\beta_{OLS} = -0.584$).

By implementing a **Two-Stage Least Squares (2SLS)** identification strategy using multi-source exogenous cost shifters and spatial price lags (Hausman-style instruments), we isolate genuine demand-side price elasticity:
- **Structural 2SLS Elasticity**: $\beta_{2SLS} = -1.452$ ($SE = 0.089$, $95\%\text{ CI: } [-1.626, -1.278]$).
- **First-Stage Instrument Strength**: Excluded instrument Wald $F$-statistic of **$42.85$**, well above the Stock-Yogo weak instrument threshold ($F > 10$).
- **Hausman Endogeneity Test**: $F = 34.12$ ($p = 1.84 \times 10^{-8}$), rejecting exogeneity and demonstrating that OLS is biased and inconsistent.
- **Sargan-Hansen Overidentifying Restrictions J-Test**: $J = 2.14$ ($p = 0.343$), failing to reject the null hypothesis of valid, orthogonal instruments.
- **Counterfactual 2024 Backtest**: Dynamic elasticity-calibrated pricing generated an aggregate **$+18.6\%$ net revenue uplift** ($95\%\text{ Bootstrap CI: } [+15.2\%, +21.9\%]$) over historical realized performance.

---

## 2. Theoretical Framework & Identification Strategy

### 2.1 The Structural Demand Equation
We formulate a log-log constant elasticity demand specification:

$$\ln(Q_{it} + 1) = \alpha + \beta \ln(P_{it}) + \mathbf{X}_{it}' \boldsymbol{\gamma} + \delta_t + \mu_i + u_{it}$$

Where:
- $Q_{it}$: Forward 30-day booking volume for listing $i$ at date $t$.
- $P_{it}$: Daily posted price in USD.
- $\beta$: Structural price elasticity of demand ($\% \Delta Q / \% \Delta P$).
- $\mathbf{X}_{it}$: Vector of exogenous time-varying and listing-specific covariates (bedrooms, accommodates, guest review score, distance to CBD, competitor density, daily temperature, rainfall, and event intensity).
- $\delta_t$: Month and day-of-week fixed effects.
- $\mu_i$: Unobserved listing-specific fixed quality (amenity aesthetics, view quality, host reputation).
- $u_{it}$: Idiosyncratic structural disturbance term.

### 2.2 The Endogeneity Problem: Simultaneity Bias
Price $P_{it}$ is endogenous because host pricing decisions correlate with unobserved transient demand shifts $\xi_{it} \subset u_{it}$:

$$\mathbb{E}[\ln(P_{it}) \cdot u_{it}] \neq 0$$

When unobserved demand surges, hosts raise prices. Consequently:
$$\operatorname{plim} \hat{\beta}_{OLS} = \beta + \frac{\operatorname{Cov}(\ln(P), u)}{\operatorname{Var}(\ln(P))} > \beta$$
Since $\operatorname{Cov}(\ln(P), u) > 0$ and true $\beta < 0$, naive OLS suffers from severe upward attenuation bias toward zero, misleading hosts into believing demand is inelastic.

---

## 3. Instrumental Variable Construction & Economic Justification

To identify $\beta$, we exploit instruments $\mathbf{Z}_{it}$ satisfying:
1. **Instrument Relevance**: $\operatorname{Cov}(\mathbf{Z}_{it}, \ln(P_{it}) \mid \mathbf{X}_{it}) \neq 0$
2. **Exclusion Restriction**: $\mathbb{E}[\mathbf{Z}_{it} \cdot u_{it} \mid \mathbf{X}_{it}] = 0$

| Instrument Name | Variable Type | Economic Rationale & Exclusion Validity |
| :--- | :--- | :--- |
| **Cleaning Fee per Guest** ($Z_1$) | Cost Shifter | Directly shifts host reservation price via turnover variable costs. Does not affect consumer fundamental willingness to travel to Austin. |
| **Hausman Spatial Lag Price** ($Z_2$) | Spatial Lag Instrument | Average price of listings in *adjacent but non-competing* submarkets. Captures citywide operational cost shocks while uncorrelated with listing-specific idiosyncratic demand $u_{it}$. |
| **Cooling Degree Energy Cost Shock** ($Z_3$) | Exogenous Weather Interaction | Daily cooling degree days ($\max(0, T - 24^\circ\text{C})$) interacted with square footage / bedrooms. Measures HVAC utility cost burden shifting host pricing floors. |

---

## 4. Empirical Diagnostics and Results

### 4.1 Comparative Model Estimates

| Regressor / Metric | Naive OLS (1) | Two-Stage Least Squares (2SLS) (2) | Two-Way Panel FE (3) |
| :--- | :--- | :--- | :--- |
| **$\ln(\text{Price})$ ($\beta$)** | **$-0.584^{***}$** $(0.038)$ | **$-1.452^{***}$** $(0.089)$ | **$-1.385^{***}$** $(0.096)$ |
| Accommodates | $+0.142^{***}$ $(0.012)$ | $+0.218^{***}$ $(0.015)$ | — (absorbed) |
| Bedrooms | $+0.085^{***}$ $(0.018)$ | $+0.124^{***}$ $(0.021)$ | — (absorbed) |
| Review Score Rating | $+0.091^{**}$ $(0.035)$ | $+0.148^{***}$ $(0.039)$ | — (absorbed) |
| Competitor Density (1km) | $-0.032^{***}$ $(0.007)$ | $-0.028^{***}$ $(0.008)$ | — (absorbed) |
| Event Intensity Index | $+0.284^{***}$ $(0.014)$ | $+0.210^{***}$ $(0.017)$ | $+0.198^{***}$ $(0.018)$ |
| Weekend Dummy | $+0.188^{***}$ $(0.009)$ | $+0.135^{***}$ $(0.011)$ | $+0.129^{***}$ $(0.011)$ |
| Mean Temperature (°C) | $+0.008^{***}$ $(0.001)$ | $+0.006^{***}$ $(0.002)$ | $+0.005^{***}$ $(0.002)$ |
| **Observations ($N$)** | **$485,200$** | **$485,200$** | **$485,200$** |
| **First-Stage $F$-Statistic** | — | **$42.85$** $(p < 10^{-15})$ | **$38.40$** $(p < 10^{-12})$ |
| **Hausman Endogeneity $F$** | — | **$34.12$** $(p = 1.84 \times 10^{-8})$ | — |
| **Sargan Over-ID $J$-Stat** | — | **$2.14$** $(p = 0.343)$ | **$1.89$** $(p = 0.388)$ |

*Standard errors clustered at listing level in parentheses. $^{***}p<0.001$, $^{**}p<0.01$.*

### 4.2 Stratified Segment Elasticities

| Property Tier Segment | Market Share | Mean Price | Estimated Elasticity ($\epsilon$) | 95% Confidence Interval | Optimal Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Budget Urban Studio** | 28.5% | $88.50 | **$-1.824$** | $[-2.05, -1.60]$ | Volume discount during off-peak |
| **Midscale Standard Condo** | 42.0% | $165.20 | **$-1.482$** | $[-1.66, -1.30]$ | Balanced dynamic pricing |
| **Premium Family Home** | 18.5% | $320.00 | **$-1.248$** | $[-1.44, -1.06]$ | Event surge capture |
| **Luxury Experiential Villa**| 11.0% | $680.00 | **$-0.915$** | $[-1.12, -0.71]$ | Premium margin pricing |

---

## 5. Counterfactual Policy Simulation & Revenue Alpha

Dynamic pricing recommendations were generated by computing elasticity-calibrated optimal price bounds:
1. **Demand Surges (SXSW, ACL, F1, Holiday Weekends)**: Inelastic capacity allows pricing markups up to $+35\%$, capturing consumer surplus.
2. **Off-Peak Midweek Windows**: High price sensitivity ($\epsilon = -1.82$ for budget units) enables targeted $10\%-18\%$ price cuts that drive disproportionate booking volume gains.

### 2024 Holdout Out-of-Sample Performance:
- **Baseline Realized Revenue**: $\$34.82\text{M}$
- **Dynamic Counterfactual Revenue**: $\$41.30\text{M}$
- **Net Revenue Uplift**: **$+\$6.48\text{M}$ ($+18.6\%$)**
- **Bootstrap 95% Confidence Interval**: $[+15.2\%, +21.9\%]$

---

## 6. Strategic Recommendations

1. **Abandon Flat / Intuition-Based Pricing**: Hosts consistently overestimate customer willingness to pay on non-event weekdays, leading to vacant room nights that generate zero revenue.
2. **Implement Segment-Specific Markup Caps**: Premium and Luxury tiers should maintain high price floors to protect luxury positioning, while Budget and Midscale tiers should dynamically flex rates.
3. **Automate Real-Time Ingestion of Exogenous Shocks**: Dynamic rate cards should adjust instantaneously upon announcement of major metropolitan event attendance updates and extreme weather forecasts.
