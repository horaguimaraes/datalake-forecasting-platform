-- One row per station with its most recent attributes and how much data we hold for it.
WITH prices AS (
    SELECT * FROM read_parquet('{silver_glob}', hive_partitioning = true)
),
latest AS (
    SELECT *, row_number() OVER (PARTITION BY cnpj ORDER BY collected_on DESC) AS rn
    FROM prices
)
SELECT l.cnpj, l.station_name, l.brand, l.region, l.state, l.city, l.neighborhood,
       s.first_seen, s.last_seen, s.observations
FROM latest l
JOIN (
    SELECT cnpj, min(collected_on) AS first_seen, max(collected_on) AS last_seen, count(*) AS observations
    FROM prices GROUP BY cnpj
) s USING (cnpj)
WHERE l.rn = 1
ORDER BY l.state, l.city, l.station_name
