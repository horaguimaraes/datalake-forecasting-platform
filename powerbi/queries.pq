// =====================================================================================
// Power Query (M) for the fuel price forecast platform.
//
// Two ways to feed Power BI; pick one:
//   A) Parquet files straight from the lake (no server needed, best for a local report)
//   B) The REST API (best when the platform is deployed)
//
// Create a text parameter first (Home > Transform data > Manage parameters):
//   LakePath   e.g.  C:\path\to\datalake-forecasting-platform\data\lake
//   ApiBase    e.g.  http://localhost:8000        (only for option B)
// =====================================================================================


// ---- A) Parquet from the lake ------------------------------------------------------

// Query: WeeklyPrices   (gold layer: one row per week x state x product, plus BR = national)
let
    Source = Parquet.Document(File.Contents(LakePath & "\gold\weekly_prices.parquet")),
    Typed  = Table.TransformColumnTypes(
                Source,
                {{"week_start", type date}, {"avg_price", type number}, {"median_price", type number},
                 {"min_price", type number}, {"max_price", type number}, {"observations", Int64.Type},
                 {"stations", Int64.Type}, {"days_covered", Int64.Type}},
                "en-US")
in
    Typed


// Query: Forecasts   (one row per series x horizon: forecast, 80% interval, naive baseline)
let
    Source = Parquet.Document(File.Contents(LakePath & "\forecasts\latest.parquet")),
    Typed  = Table.TransformColumnTypes(
                Source,
                {{"origin_week", type date}, {"target_week", type date}, {"horizon", Int64.Type},
                 {"price_origin", type number}, {"forecast", type number}, {"lo", type number},
                 {"hi", type number}, {"naive", type number}},
                "en-US")
in
    Typed


// Query: Stations   (dimension)
let
    Source = Parquet.Document(File.Contents(LakePath & "\gold\stations.parquet")),
    Typed  = Table.TransformColumnTypes(Source, {{"cnpj", type text}, {"first_seen", type date}, {"last_seen", type date}}, "en-US")
in
    Typed


// ---- B) REST API -------------------------------------------------------------------

// Query: fnApiForecast   (function: product, state -> table of forecast points)
(product as text, optional state as nullable text) as table =>
let
    Params = [product = product, state = if state = null then "BR" else state],
    Json   = Json.Document(Web.Contents(ApiBase, [RelativePath = "v1/forecast", Query = Params])),
    Points = Table.FromRecords(Json[points]),
    Typed  = Table.TransformColumnTypes(Points, {{"target_week", type date}, {"forecast", type number}, {"lo", type number},
                                                  {"hi", type number}, {"naive", type number}}, "en-US")
in
    Typed


// Query: ModelEvidence   (champion, skill vs naive and its confidence interval, interval coverage)
let
    Json = Json.Document(Web.Contents(ApiBase, [RelativePath = "v1/models"])),
    Row  = [champion = Json[champion], skill = Json[champion_skill_ci][skill], skill_lo = Json[champion_skill_ci][lo],
            skill_hi = Json[champion_skill_ci][hi], coverage = Json[interval_coverage][empirical], data_through = Json[data_through]]
in
    Table.FromRecords({Row})
