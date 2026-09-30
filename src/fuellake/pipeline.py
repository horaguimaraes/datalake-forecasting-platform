"""The pipeline as a DAG: bronze -> silver -> gold -> model -> forecast -> report,
with a data-quality gate between every layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .bronze import fetch_bronze, import_local
from .config import Settings
from .gold import build_gold, read_weekly_prices
from .ml.backtest import HORIZONS
from .ml.forecast import make_forecast
from .ml.train import DEFAULT_MODELS, TrainConfig, train
from .orchestrator import Pipeline, Task
from .quality import run_quality
from .report import write_report
from .silver import build_silver


@dataclass(frozen=True)
class RunParams:
    start: str = "2023-01"
    end: str | None = None  # default: the last complete month
    local_dir: Path | None = None  # offline mode: read raw files from here instead of downloading
    models: tuple[str, ...] = tuple(DEFAULT_MODELS)
    horizons: tuple[int, ...] = HORIZONS
    n_folds: int = 12
    force: bool = False
    as_of: date | None = None  # "today" for freshness checks (reproducible runs / demos)


def default_end(today: date | None = None) -> str:
    today = today or date.today()
    year, month = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
    return f"{year}-{month:02d}"


def build_pipeline(settings: Settings, params: RunParams, log: Callable[[str], None] = print) -> Pipeline:
    def bronze() -> dict[str, Any]:
        if params.local_dir:
            return import_local(settings, params.local_dir, log=log).to_dict()
        end = params.end or default_end(params.as_of)
        return fetch_bronze(settings, params.start, end, force=params.force, today=params.as_of, log=log).to_dict()

    def gate(layer: str) -> Callable[[], dict[str, int]]:
        def run() -> dict[str, int]:
            report = run_quality(settings, layers=(layer,), today=params.as_of)
            for check in report.warnings:
                log(f"  WARN  {check.name}: {check.detail[:200]}")
            report.enforce()
            return report.summary()

        return run

    def silver() -> dict[str, Any]:
        metrics = build_silver(settings, log=log)
        return {k: v for k, v in metrics.items() if k != "sources"}

    def gold() -> dict[str, Any]:
        return build_gold(settings, log=log)

    def train_task() -> dict[str, Any]:
        weekly = read_weekly_prices(settings)
        config = TrainConfig(models=params.models, horizons=params.horizons, n_folds=params.n_folds)
        metrics = train(settings, weekly, config, log=log)
        return {"run_id": metrics["run_id"], "champion": metrics["champion"], "skill_ci": metrics["champion_skill_ci"]}

    def forecast() -> dict[str, Any]:
        forecast_df = make_forecast(settings, read_weekly_prices(settings))
        return {"rows": len(forecast_df), "origin_week": str(forecast_df["origin_week"].iloc[0].date())}

    def report() -> str:
        weekly = read_weekly_prices(settings)
        import pandas as pd

        return write_report(settings, weekly, pd.read_parquet(settings.forecasts / "latest.parquet"))

    return Pipeline(
        [
            Task("bronze", bronze, retries=2, retry_delay=3, description="Fetch raw ANP files (or import local ones)"),
            Task("quality_bronze", gate("bronze"), deps=("bronze",), description="Checksums and source coverage"),
            Task("silver", silver, deps=("quality_bronze",), description="Clean, type, deduplicate -> Parquet"),
            Task("quality_silver", gate("silver"), deps=("silver",), description="Schema, keys, reject rate"),
            Task("gold", gold, deps=("quality_silver",), description="Weekly price series with SQL"),
            Task("quality_gold", gate("gold"), deps=("gold",), description="Coverage, freshness, anomalies"),
            Task("train", train_task, deps=("quality_gold",), description="Backtest, pick champion, register"),
            Task("forecast", forecast, deps=("train",), description="Forecast every series, 1-8 weeks"),
            Task("report", report, deps=("forecast",), description="HTML report"),
        ]
    )
