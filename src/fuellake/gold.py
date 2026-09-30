"""Gold layer: business-ready tables built with SQL over silver.

The SQL lives in `sql/*.sql` (readable and diff-able on its own) and runs in DuckDB, which reads the
Parquet files in place. Nothing is loaded into a server.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from importlib.resources import files
from typing import Any

import duckdb
import pandas as pd

from .config import Settings

GOLD_TABLES = {"weekly_prices": "weekly_prices.sql", "stations": "stations.sql"}


def load_sql(name: str, settings: Settings) -> str:
    text = files("fuellake").joinpath("sql", name).read_text(encoding="utf-8")
    return text.replace("{silver_glob}", settings.silver_glob)


def build_gold(settings: Settings, log: Callable[[str], None] = lambda message: None) -> dict[str, Any]:
    settings.ensure()
    if not any(settings.silver_prices.glob("*/*/*.parquet")):
        raise RuntimeError("Silver is empty: run the silver step first.")

    tmp = settings.gold.with_name("gold.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    con = duckdb.connect(":memory:")
    metrics: dict[str, Any] = {}
    for table, sql_file in GOLD_TABLES.items():
        target = (tmp / f"{table}.parquet").as_posix()
        con.execute(f"COPY ({load_sql(sql_file, settings)}) TO '{target}' (FORMAT parquet, COMPRESSION zstd)")
        rows = con.execute(f"SELECT count(*) FROM read_parquet('{target}')").fetchone()[0]
        metrics[table] = {"rows": rows}
        log(f"  gold.{table}: {rows:,} rows")

    weekly = con.execute(
        f"SELECT min(week_start), max(week_start), count(DISTINCT product), count(DISTINCT state) "
        f"FROM read_parquet('{(tmp / 'weekly_prices.parquet').as_posix()}')"
    ).fetchone()
    metrics["weekly_prices"] |= {
        "week_from": weekly[0].isoformat(),
        "week_to": weekly[1].isoformat(),
        "products": weekly[2],
        "states": weekly[3],
    }
    con.close()
    (tmp / "_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if settings.gold.exists():
        shutil.rmtree(settings.gold)
    tmp.replace(settings.gold)
    return metrics


def read_weekly_prices(settings: Settings) -> pd.DataFrame:
    df = pd.read_parquet(settings.gold / "weekly_prices.parquet")
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def open_lake(settings: Settings) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB with a view per lake table, for ad-hoc SQL (`fuellake sql "..."`)."""
    con = duckdb.connect(":memory:")
    if any(settings.silver_prices.glob("*/*/*.parquet")):
        con.execute(
            f"CREATE VIEW silver_prices AS SELECT * FROM read_parquet('{settings.silver_glob}', hive_partitioning = true)"
        )
    for table in GOLD_TABLES:
        path = settings.gold / f"{table}.parquet"
        if path.exists():
            con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path.as_posix()}')")
    forecast = settings.forecasts / "latest.parquet"
    if forecast.exists():
        con.execute(f"CREATE VIEW forecasts AS SELECT * FROM read_parquet('{forecast.as_posix()}')")
    return con
