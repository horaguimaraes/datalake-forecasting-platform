# Power BI

The platform exposes the same data two ways, so Power BI can sit on either:

| Option | Source | When to use |
|:--|:--|:--|
| **A. Parquet** | `gold/weekly_prices.parquet`, `forecasts/latest.parquet`, `gold/stations.parquet` | Local report; no server needed |
| **B. REST API** | `/v1/forecast`, `/v1/models`, `/v1/history` | The platform is deployed |

[`queries.pq`](queries.pq) has the Power Query (M) for both. Create a `LakePath` (and `ApiBase`) parameter, paste each query into a blank query, and load.

> **Status:** the M queries are written against the real column names and response shapes of this platform, which its tests cover. They have **not been exercised in Power BI Desktop**, and no `.pbix` is committed, so treat this as a tested-on-the-data-side starting point.

## Model

```
WeeklyPrices (week_start, state, product)    -> history and the national (BR) series
Forecasts    (target_week, state, product)   -> forecast, lo, hi, naive, horizon
Stations     (cnpj)                          -> optional dimension
```

Relate `WeeklyPrices[week_start]` and `Forecasts[target_week]` to a shared date table, and `state` / `product` to small dimension tables, so one slicer drives both history and forecast.

## Suggested measures (DAX)

```dax
Avg Price = AVERAGE ( WeeklyPrices[avg_price] )

Forecast = AVERAGE ( Forecasts[forecast] )

Forecast Low  = AVERAGE ( Forecasts[lo] )
Forecast High = AVERAGE ( Forecasts[hi] )

Expected Change % =
VAR Now = CALCULATE ( AVERAGE ( Forecasts[price_origin] ) )
RETURN DIVIDE ( [Forecast] - Now, Now )

Forecast vs No Change % =
DIVIDE ( [Forecast] - AVERAGE ( Forecasts[naive] ), AVERAGE ( Forecasts[naive] ) )
```

## Suggested report

1. **Forecast**: line chart of `Avg Price` (history) and `Forecast` with `Forecast Low`/`High` as the band, sliced by product and state.
2. **Where prices are heading**: bar chart of `Expected Change %` by state for a chosen horizon.
3. **Trust**: cards with the champion model, its skill vs naive and the 95% interval from `ModelEvidence`. Show them next to the forecast so viewers see how much to believe it.
