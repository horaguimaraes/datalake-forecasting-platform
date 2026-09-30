-- Weekly price statistics per state and product, plus a national roll-up (state = 'BR').
-- Weeks start on Monday. `days_covered` (distinct collection days) lets downstream steps
-- recognise partial weeks at the edges of the data.
WITH prices AS (
    SELECT * FROM read_parquet('{silver_glob}', hive_partitioning = true)
),
by_state AS (
    SELECT CAST(date_trunc('week', collected_on) AS DATE) AS week_start,
           state,
           product,
           avg(price)                     AS avg_price,
           median(price)                  AS median_price,
           min(price)                     AS min_price,
           max(price)                     AS max_price,
           coalesce(stddev_pop(price), 0) AS std_price,
           count(*)                       AS observations,
           count(DISTINCT cnpj)           AS stations,
           count(DISTINCT collected_on)   AS days_covered
    FROM prices
    GROUP BY 1, 2, 3
),
national AS (
    SELECT CAST(date_trunc('week', collected_on) AS DATE) AS week_start,
           'BR'                           AS state,
           product,
           avg(price)                     AS avg_price,
           median(price)                  AS median_price,
           min(price)                     AS min_price,
           max(price)                     AS max_price,
           coalesce(stddev_pop(price), 0) AS std_price,
           count(*)                       AS observations,
           count(DISTINCT cnpj)           AS stations,
           count(DISTINCT collected_on)   AS days_covered
    FROM prices
    GROUP BY 1, 3
)
SELECT * FROM by_state
UNION ALL
SELECT * FROM national
ORDER BY product, state, week_start
