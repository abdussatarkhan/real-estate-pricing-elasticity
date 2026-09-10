# Short-Term Rental Pricing Elasticity & Revenue Maximizer

[![Econometrics](https://img.shields.io/badge/Econometrics-2SLS_IV-0056B3?style=for-the-badge)](https://en.wikipedia.org/wiki/Instrumental_variables_estimation) [![Python](https://img.shields.io/badge/Python-Statsmodels-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.statsmodels.org/)
[![Author](https://img.shields.io/badge/Author-Abdussatar-E50914?style=for-the-badge&logo=github&logoColor=white)](https://github.com/satarabdus692-bot)

> **An econometric pricing elasticity and dynamic revenue optimization engine utilizing Two-Stage Least Squares (2SLS) instrumental variables and neighborhood panel fixed effects to estimate true causal price elasticity.**

---

## 🏛️ System Architecture

```mermaid
graph TD
    Listings[Property Listings & Daily Booking Data] --> IV[Instrumental Variables: Competitor Price Shocks]
    IV --> 2SLS[Two-Stage Least Squares 2SLS Regression]
    2SLS --> Elasticity[Neighborhood Demand Elasticity Curves]
    Elasticity --> RevenueOpt[Optimal Dynamic Price Multiplier]
```

---

## 🌟 Key Features & Capabilities

- **Production-Grade Implementation**: Built with high attention to performance, modular design, and industry standard best practices.
- **Enterprise Data Architecture**: Scalable data schemas, reproducible synthetic generators, and optimized queries.
- **Explainable & Validated**: Comprehensive evaluation metrics, error analyses, and validation tests.
- **Comprehensive Tech Stack**: `Python` `Statsmodels` `Pandas` `Econometrics` `2SLS` `Optimization`.

---

## 📊 Visual Preview & Analysis

<div align="center">

![real-estate-pricing-elasticity preview](images/elasticity_curve_by_neighborhood.png)

</div>

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/satarabdus692-bot/real-estate-pricing-elasticity.git
cd real-estate-pricing-elasticity
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

---

## 👨‍💻 Author & Profile

Built and maintained by **Abdussatar** ([@satarabdus692-bot](https://github.com/satarabdus692-bot)).  
For technical discussions, collaboration, or queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abdus-satar-5150813b5/) or [GitHub](https://github.com/satarabdus692-bot).

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.
