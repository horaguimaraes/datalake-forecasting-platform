"""Where things live. The whole lake is one directory tree, so it can be moved, zipped or deleted."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Lake layout. Every path is derived from `root`.

    bronze/     raw files exactly as downloaded, plus a manifest with checksums
    silver/     cleaned, typed, deduplicated Parquet (Hive-partitioned by year/month)
    gold/       business-level aggregates ready for analytics and modelling
    models/     trained model registry
    forecasts/  forecast outputs
    reports/    quality report and the HTML forecast report
    runs/       one JSON log per pipeline run
    """

    root: Path

    @classmethod
    def from_env(cls, root: str | os.PathLike[str] | None = None) -> Settings:
        return cls(Path(root or os.environ.get("FUELLAKE_ROOT", "data/lake")))

    @property
    def bronze(self) -> Path:
        return self.root / "bronze"

    @property
    def bronze_manifest(self) -> Path:
        return self.bronze / "_manifest.json"

    @property
    def silver(self) -> Path:
        return self.root / "silver"

    @property
    def silver_prices(self) -> Path:
        return self.silver / "prices"

    @property
    def silver_glob(self) -> str:
        return (self.silver_prices / "*" / "*" / "*.parquet").as_posix()

    @property
    def silver_metrics(self) -> Path:
        return self.silver / "_metrics.json"

    @property
    def silver_rejects(self) -> Path:
        return self.silver / "_rejects.parquet"

    @property
    def gold(self) -> Path:
        return self.root / "gold"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def forecasts(self) -> Path:
        return self.root / "forecasts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    def ensure(self) -> None:
        for path in (self.bronze, self.silver, self.gold, self.models, self.forecasts, self.reports, self.runs):
            path.mkdir(parents=True, exist_ok=True)
