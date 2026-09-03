# Raw Data Directory & Acquisition Protocol

This directory houses the raw data feeds required by the Dynamic Pricing Elasticity Engine. Due to size limitations and terms of service, large raw datasets are excluded from version control via `.gitignore`.

---

## 1. Inside Airbnb Data Feeds
Inside Airbnb provides detailed snapshots of city-level short-term rental listings, calendar availability, historical reviews, and neighborhood boundary definitions.

- **Source**: [Inside Airbnb Data Portal](http://insideairbnb.com/get-the-data.html)
- **Target Market**: Austin, TX (or target metropolitan area)
- **Target Snapshot**: March 2024 (or most recent quarterly release)
- **Required Files**:
  1. `listings.csv.gz` (Detailed listing attributes: amenities, pricing, location coordinates, host attributes, review scores)
  2. `calendar.csv.gz` (~7M-10M records: daily listing availability, adjusted price, minimum/maximum nights for 365 days forward)
  3. `reviews.csv.gz` (Timestamped guest review records used to infer booking events via the San Francisco Model)
  4. `neighbourhoods.geojson` (Official municipal planning district polygons for spatial aggregation)

### Automated Download via Python
Run the ingestion pipeline to automatically stream and extract data into this directory:
```bash
python scripts/data_collection.py --source inside_airbnb --city austin
```

---

## 2. NOAA Climate Data Online (CDO) API
Daily meteorological records are leveraged as exogenous cost and demand shifters (e.g., HVAC electricity cost shifts, weather-induced tourist influx).

- **Source**: [NOAA NCEI Climate Data Online](https://www.ncei.noaa.gov/cdo-web/)
- **API Token Registration**: Obtain a free token at [NOAA CDO Token Request](https://www.ncei.noaa.gov/cdo-web/token)
- **Station ID**: `GHCND:USW00013904` (Austin Bergstrom International Airport Station)
- **Key Metrics Collected**:
  - `TMAX`: Maximum daily temperature (tenths of °C)
  - `TMIN`: Minimum daily temperature (tenths of °C)
  - `PRCP`: Daily precipitation amount (tenths of mm)
  - `AWND`: Average daily wind speed (meters/sec)

### Automated Fetch:
```bash
python scripts/data_collection.py --source noaa --api-token <YOUR_NOAA_TOKEN>
```

---

## 3. Eventbrite Events API
Local event volume, major conferences (e.g., SXSW, Austin City Limits, Formula 1), concerts, and festivals represent transient demand shocks.

- **Source**: [Eventbrite Developer Portal](https://www.eventbrite.com/platform/api)
- **API Key**: Configure OAuth token in `.env` or `config/config.yaml`
- **Search Parameters**:
  - Center: Austin Downtown (30.2672° N, 97.7431° W)
  - Radius: 25 km
  - Categories: Music, Business, Festivals, Sports

### Automated Fetch:
```bash
python scripts/data_collection.py --source eventbrite --api-key <YOUR_EVENTBRITE_KEY>
```

---

## Synthetic / Simulation Mode
For offline reproducibility without requiring external live API credentials, `scripts/data_collection.py` features a `--sample` flag that generates a statistically representative synthetic dataset matching the schema of Inside Airbnb 7M-row calendar, NOAA weather, and Eventbrite feeds.
```bash
python scripts/data_collection.py --sample
```
