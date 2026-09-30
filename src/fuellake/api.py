"""REST API that serves what the pipeline produced: history, forecasts, model metrics, data quality, lineage.

It only reads the lake (Parquet/JSON), so it can be scaled out or restarted freely, and it always
reports which model run and data produced an answer.
"""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from . import __version__
from .bronze import Manifest
from .config import Settings
from .ml.train import load_metrics


class HistoryPoint(BaseModel):
    week_start: str
    avg_price: float
    median_price: float
    stations: int
    observations: int


class ForecastPoint(BaseModel):
    horizon: int
    target_week: str
    forecast: float
    lo: float | None
    hi: float | None
    naive: float


class ForecastResponse(BaseModel):
    product: str
    state: str
    origin_week: str
    price_origin: float
    model: str
    model_run: str
    interval: str
    points: list[ForecastPoint]


class Mover(BaseModel):
    product: str
    state: str
    price_origin: float
    forecast: float
    change_pct: float


def _mtime_cache(loader: Callable[[Path], Any]) -> Callable[[Path], Any]:
    """Cache a file-reading function until the file changes, so a new pipeline run is picked up live."""
    cache: dict[Path, tuple[float, Any]] = {}

    @wraps(loader)
    def wrapper(path: Path) -> Any:
        mtime = path.stat().st_mtime
        hit = cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        value = loader(path)
        cache[path] = (mtime, value)
        return value

    return wrapper


@_mtime_cache
def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def create_app(settings: Settings | None = None, api_key: str | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    api_key = api_key if api_key is not None else (os.environ.get("API_KEY") or None)

    app = FastAPI(
        title="Fuel Price Forecast API",
        version=__version__,
        description="Weekly fuel price history and 1-8 week forecasts for Brazil, built from ANP open data.",
    )

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if api_key and not (x_api_key and hmac.compare_digest(x_api_key, api_key)):
            raise HTTPException(status_code=401, detail="Missing or invalid x-api-key header")

    auth = [Depends(require_key)]

    def forecasts() -> pd.DataFrame:
        path = settings.forecasts / "latest.parquet"
        if not path.exists():
            raise HTTPException(status_code=503, detail="No forecast available yet. Run the pipeline first.")
        return _read_parquet(path)

    def weekly() -> pd.DataFrame:
        path = settings.gold / "weekly_prices.parquet"
        if not path.exists():
            raise HTTPException(status_code=503, detail="Gold layer is empty. Run the pipeline first.")
        return _read_parquet(path)

    def json_file(path: Path, name: str) -> Any:
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"{name} not available yet")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "layers": {
                "bronze": settings.bronze_manifest.exists(),
                "silver": settings.silver_metrics.exists(),
                "gold": (settings.gold / "weekly_prices.parquet").exists(),
                "model": (settings.models / "latest.json").exists(),
                "forecast": (settings.forecasts / "latest.parquet").exists(),
            },
        }

    @app.get("/v1/series", dependencies=auth)
    def series() -> dict[str, Any]:
        df = forecasts()
        first = df[df["horizon"] == df["horizon"].min()]
        return {
            "origin_week": str(first["origin_week"].iloc[0].date()),
            "data": [
                {"product": r.product, "state": r.state, "price_origin": round(r.price_origin, 3)}
                for r in first.sort_values(["product", "state"]).itertuples()
            ],
        }

    @app.get("/v1/history", response_model=list[HistoryPoint], dependencies=auth)
    def history(
        product: str = Query(..., examples=["GASOLINA"]),
        state: str = Query("BR", min_length=2, max_length=2, description="UF, or BR for national"),
        weeks: int = Query(104, ge=1, le=520),
    ) -> list[HistoryPoint]:
        df = weekly()
        rows = df[(df["product"] == product.upper()) & (df["state"] == state.upper())].sort_values("week_start").tail(weeks)
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"No history for {product.upper()} / {state.upper()}")
        return [
            HistoryPoint(
                week_start=str(pd.Timestamp(r.week_start).date()),
                avg_price=round(r.avg_price, 4),
                median_price=round(r.median_price, 4),
                stations=int(r.stations),
                observations=int(r.observations),
            )
            for r in rows.itertuples()
        ]

    @app.get("/v1/forecast", response_model=ForecastResponse, dependencies=auth)
    def forecast(
        product: str = Query(..., examples=["GASOLINA"]),
        state: str = Query("BR", min_length=2, max_length=2),
    ) -> ForecastResponse:
        df = forecasts()
        rows = df[(df["product"] == product.upper()) & (df["state"] == state.upper())].sort_values("horizon")
        if rows.empty:
            known = sorted(df["product"].unique())
            raise HTTPException(status_code=404, detail=f"No forecast for {product.upper()} / {state.upper()}. Products: {known}")
        first = rows.iloc[0]

        def num(value: float) -> float | None:
            return None if pd.isna(value) else round(float(value), 4)

        return ForecastResponse(
            product=first["product"],
            state=first["state"],
            origin_week=str(pd.Timestamp(first["origin_week"]).date()),
            price_origin=round(float(first["price_origin"]), 4),
            model=first["model"],
            model_run=first["model_run"],
            interval="80% (empirical residual quantiles from the backtest)",
            points=[
                ForecastPoint(
                    horizon=int(r.horizon),
                    target_week=str(pd.Timestamp(r.target_week).date()),
                    forecast=round(float(r.forecast), 4),
                    lo=num(r.lo),
                    hi=num(r.hi),
                    naive=round(float(r.naive), 4),
                )
                for r in rows.itertuples()
            ],
        )

    @app.get("/v1/forecast/movers", response_model=list[Mover], dependencies=auth)
    def movers(
        horizon: int = Query(4, ge=1, le=8),
        direction: str = Query("up", pattern="^(up|down)$"),
        limit: int = Query(10, ge=1, le=100),
    ) -> list[Mover]:
        df = forecasts()
        rows = df[(df["horizon"] == horizon) & (df["state"] != "BR")].copy()
        rows["change"] = rows["forecast"] / rows["price_origin"] - 1
        rows = rows.sort_values("change", ascending=direction == "down").head(limit)
        return [
            Mover(
                product=r.product,
                state=r.state,
                price_origin=round(r.price_origin, 3),
                forecast=round(r.forecast, 3),
                change_pct=round(r.change * 100, 2),
            )
            for r in rows.itertuples()
        ]

    @app.get("/v1/models", dependencies=auth)
    def models() -> dict[str, Any]:
        metrics = load_metrics(settings)
        if metrics is None:
            raise HTTPException(status_code=404, detail="No trained model yet")
        keep = [
            "run_id",
            "trained_at",
            "champion",
            "data_through",
            "n_series",
            "backtest",
            "ranking",
            "champion_skill_ci",
            "interval_coverage",
        ]
        return {k: metrics[k] for k in keep}

    @app.get("/v1/quality", dependencies=auth)
    def quality() -> Any:
        return json_file(settings.reports / "quality_report.json", "Quality report")

    @app.get("/v1/lineage", dependencies=auth)
    def lineage() -> dict[str, Any]:
        manifest = Manifest.load(settings.bronze_manifest)
        silver = json_file(settings.silver_metrics, "Silver metrics")
        return {
            "bronze": {
                "files": len(manifest.ok_entries()),
                "bytes": sum(e["bytes"] for e in manifest.ok_entries()),
                "known_gaps": sorted({m for e in manifest.missing_entries() for m in e.get("covers", [])}),
                "sources": [{"id": e["id"], "sha256": e["sha256"], "fetched_at": e["fetched_at"]} for e in manifest.ok_entries()],
            },
            "silver": {k: v for k, v in silver.items() if k != "sources"},
        }

    return app
