-- ============================================================================
-- DUCKDB OLAP ANALYTICAL QUERY SUITE
-- Dynamic Pricing Elasticity Engine for Short-Term Rentals
-- ============================================================================

-- Query 1: Submarket Key Performance Indicators (ADR, RevPAR, Occupancy Rate)
-- Computes core hospitality metrics aggregated by municipal planning district.
SELECT 
    l.neighbourhood_assigned AS neighbourhood,
    COUNT(DISTINCT l.listing_id) AS active_listings,
    ROUND(AVG(c.daily_price_usd), 2) AS adr_average_daily_rate,
    ROUND(AVG(c.is_booked) * 100.0, 2) AS occupancy_rate_pct,
    ROUND(AVG(c.daily_price_usd * c.is_booked), 2) AS revpar_revenue_per_available_room,
    ROUND(SUM(c.daily_price_usd * c.is_booked), 2) AS total_realized_revenue,
    ROUND(AVG(l.review_scores_rating), 2) AS avg_guest_rating
FROM listings l
JOIN calendar c ON l.listing_id = c.listing_id
GROUP BY l.neighbourhood_assigned
ORDER BY total_realized_revenue DESC;


-- Query 2: Rolling Occupancy Velocity via Window Functions
-- Calculates 7-day, 14-day, and 30-day trailing booking velocity per listing.
SELECT 
    c.listing_id,
    c.date,
    c.daily_price_usd,
    c.is_booked,
    SUM(c.is_booked) OVER (
        PARTITION BY c.listing_id 
        ORDER BY c.date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_booked_7d,
    SUM(c.is_booked) OVER (
        PARTITION BY c.listing_id 
        ORDER BY c.date 
        ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
    ) AS rolling_booked_14d,
    ROUND(AVG(c.is_booked) OVER (
        PARTITION BY c.listing_id 
        ORDER BY c.date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) * 100.0, 2) AS trailing_30d_occupancy_pct
FROM calendar c
ORDER BY c.listing_id, c.date;


-- Query 3: Exogenous Event Demand Shock Impact Analysis
-- Measures price premiums and booking elasticity during major convention/festival windows (SXSW, F1)
-- compared against non-event baseline periods.
WITH event_impact AS (
    SELECT 
        l.neighbourhood_assigned,
        CASE 
            WHEN c.event_intensity_score > 0 THEN 'Event Surge Period'
            ELSE 'Standard Period'
        END AS market_regime,
        c.daily_price_usd,
        c.is_booked
    FROM calendar c
    JOIN listings l ON c.listing_id = l.listing_id
)
SELECT 
    neighbourhood_assigned,
    market_regime,
    COUNT(*) AS observed_room_nights,
    ROUND(AVG(daily_price_usd), 2) AS avg_price,
    ROUND(AVG(is_booked) * 100.0, 2) AS occupancy_pct,
    ROUND(AVG(daily_price_usd * is_booked), 2) AS revpar
FROM event_impact
GROUP BY neighbourhood_assigned, market_regime
ORDER BY neighbourhood_assigned, market_regime DESC;


-- Query 4: Day-of-Week Weekend Premium & Arc Elasticity Cohorts
-- Computes the weekend price elasticity differential across accommodation tiers.
SELECT 
    l.room_type,
    CASE 
        WHEN EXTRACT(DOW FROM c.date) IN (5, 6, 0) THEN 'Weekend (Fri-Sun)'
        ELSE 'Weekday (Mon-Thu)'
    END AS day_classification,
    COUNT(*) AS total_nights,
    ROUND(AVG(c.daily_price_usd), 2) AS mean_price,
    ROUND(MEDIAN(c.daily_price_usd), 2) AS median_price,
    ROUND(AVG(c.is_booked) * 100.0, 2) AS booking_rate_pct
FROM calendar c
JOIN listings l ON c.listing_id = l.listing_id
GROUP BY l.room_type, day_classification
ORDER BY l.room_type, day_classification;


-- Query 5: Counterfactual Pricing Backtest Revenue Comparison by Market Tier
-- Compares actual historical revenues versus dynamic elasticity model revenues across clusters.
SELECT 
    b.cluster_segment,
    COUNT(*) AS total_simulated_nights,
    ROUND(AVG(b.daily_price_usd), 2) AS avg_historical_price,
    ROUND(AVG(b.recommended_price_usd), 2) AS avg_recommended_price,
    ROUND(SUM(b.actual_revenue_usd), 2) AS total_actual_revenue,
    ROUND(SUM(b.counterfactual_expected_revenue_usd), 2) AS total_counterfactual_revenue,
    ROUND(SUM(b.counterfactual_expected_revenue_usd) - SUM(b.actual_revenue_usd), 2) AS net_revenue_gain,
    ROUND((SUM(b.counterfactual_expected_revenue_usd) - SUM(b.actual_revenue_usd)) / SUM(b.actual_revenue_usd) * 100.0, 2) AS net_uplift_pct
FROM backtest_results b
GROUP BY b.cluster_segment
ORDER BY net_revenue_gain DESC;


-- Query 6: Spatial Competitor Price Dispersion & Hausman Lag Verification
-- Computes mean listing price in each neighborhood vs all other external neighborhoods.
WITH submarket_means AS (
    SELECT 
        neighbourhood_assigned,
        AVG(price_usd) AS internal_avg_price,
        COUNT(*) AS listing_count
    FROM listings
    GROUP BY neighbourhood_assigned
),
overall_total AS (
    SELECT SUM(price_usd) AS total_p, COUNT(*) AS total_n FROM listings
)
SELECT 
    s.neighbourhood_assigned,
    ROUND(s.internal_avg_price, 2) AS neighborhood_price,
    s.listing_count,
    ROUND((o.total_p - (s.internal_avg_price * s.listing_count)) / (o.total_n - s.listing_count), 2) AS external_hausman_instrument_price
FROM submarket_means s
CROSS JOIN overall_total o
ORDER BY s.internal_avg_price DESC;
