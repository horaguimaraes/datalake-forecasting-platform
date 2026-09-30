"""Data quality checks that run between layers.

Severity decides what happens: an `error` stops the pipeline (bad data must not reach the model),
a `warn` is recorded in the report and the run continues. Everything lands in
`reports/quality_report.json`, so a reviewer can see exactly what was checked and what was found.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import duckdb
import pandas as pd

from .bronze import Manifest, verify_bronze
from .config import Settings

EXPECTED_SILVER_COLUMNS = {
    "cnpj",
    "station_name",
    "brand",
    "region",
    "state",
    "city",
    "neighborhood",
    "product",
    "unit",
    "collected_on",
    "price",
    "purchase_price",
    "source_id",
}


class QualityError(RuntimeError):
    """At least one error-severity check failed."""


@dataclass(frozen=True)
class QualityConfig:
    reject_warn_rate: float = 0.01  # more than 1% rejected rows -> warn
    reject_error_rate: float = 0.05  # more than 5% -> stop
    max_weekly_jump: float = 0.15  # a national weekly move above 15% is suspicious
    max_monthly_jump: float = 0.30  # month-over-month median move above 30% is suspicious
    stale_warn_days: int = 21  # newest week older than this -> warn
    stale_error_days: int = 180
    min_stations_per_week: int = 5  # national series need at least this many stations


@dataclass
class Check:
    name: str
    layer: str
    severity: str  # "error" | "warn"
    passed: bool
    detail: str
    value: Any = None


@dataclass
class QualityReport:
    generated_at: str
    checks: list[Check] = field(default_factory=list)

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "warn"]

    def summary(self) -> dict[str, int]:
        return {
            "checks": len(self.checks),
            "passed": sum(c.passed for c in self.checks),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"generated_at": self.generated_at, "summary": self.summary(), "checks": [asdict(c) for c in self.checks]}

    def enforce(self) -> None:
        if self.errors:
            raise QualityError("; ".join(f"[{c.layer}] {c.name}: {c.detail}" for c in self.errors))


def _check(report: QualityReport, name: str, layer: str, severity: str, passed: bool, detail: str, value: Any = None) -> None:
    report.checks.append(Check(name, layer, severity, bool(passed), detail, value))


# --------------------------------------------------------------------------- bronze


def check_bronze(settings: Settings, report: QualityReport) -> None:
    manifest = Manifest.load(settings.bronze_manifest)
    _check(
        report,
        "bronze_not_empty",
        "bronze",
        "error",
        bool(manifest.ok_entries()),
        f"{len(manifest.ok_entries())} files in bronze",
    )
    problems = verify_bronze(settings)
    _check(
        report,
        "bronze_checksums",
        "bronze",
        "error",
        not problems,
        "; ".join(problems) or "every file matches its recorded SHA-256",
    )
    gaps = sorted({m for e in manifest.missing_entries() for m in e.get("covers", [])})
    covered = {m for e in manifest.ok_entries() for m in e.get("covers", [])}
    gaps = [m for m in gaps if m not in covered]
    _check(
        report,
        "bronze_source_gaps",
        "bronze",
        "warn",
        not gaps,
        f"ANP has not published: {', '.join(gaps)}" if gaps else "every requested month is available",
        gaps,
    )


# --------------------------------------------------------------------------- silver


def check_silver(settings: Settings, report: QualityReport, config: QualityConfig | None = None) -> None:
    config = config or QualityConfig()
    metrics = json.loads(settings.silver_metrics.read_text(encoding="utf-8"))
    con = duckdb.connect(":memory:")
    source = f"read_parquet('{settings.silver_glob}', hive_partitioning = true)"

    columns = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()}
    missing = sorted(EXPECTED_SILVER_COLUMNS - columns)
    _check(
        report,
        "silver_schema",
        "silver",
        "error",
        not missing,
        f"missing columns: {', '.join(missing)}" if missing else "all expected columns present",
    )

    rows = con.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
    _check(report, "silver_not_empty", "silver", "error", rows > 0, f"{rows:,} rows", rows)

    read = max(metrics["rows_read"], 1)
    rate = (metrics["rows_rejected"] + metrics["rows_malformed"]) / read
    _check(
        report,
        "silver_reject_rate_error",
        "silver",
        "error",
        rate <= config.reject_error_rate,
        f"{rate:.2%} of rows rejected (limit {config.reject_error_rate:.0%})",
        rate,
    )
    _check(
        report,
        "silver_reject_rate_warn",
        "silver",
        "warn",
        rate <= config.reject_warn_rate,
        f"{rate:.2%} of rows rejected: {metrics['rejects_by_reason']}",
        rate,
    )

    dupes = con.execute(
        f"SELECT count(*) FROM (SELECT 1 FROM {source} GROUP BY cnpj, product, collected_on HAVING count(*) > 1)"
    ).fetchone()[0]
    _check(report, "silver_unique_key", "silver", "error", dupes == 0, f"{dupes} duplicated (station, product, date) keys", dupes)

    nulls = con.execute(f"SELECT count(*) FROM {source} WHERE price IS NULL OR collected_on IS NULL OR cnpj IS NULL").fetchone()[
        0
    ]
    _check(report, "silver_no_null_keys", "silver", "error", nulls == 0, f"{nulls} rows with null price/date/cnpj", nulls)

    # month-over-month median per product: catches unit changes and corrupted months
    jumps = con.execute(
        f"""WITH m AS (SELECT product, date_trunc('month', collected_on) AS month, median(price) AS med
                    FROM {source} GROUP BY 1, 2),
                 c AS (SELECT product, month, med, lag(med) OVER (PARTITION BY product ORDER BY month) AS prev FROM m)
            SELECT product, strftime(month, '%Y-%m') AS month, med / prev - 1 AS change FROM c
            WHERE prev IS NOT NULL AND abs(med / prev - 1) > {config.max_monthly_jump} ORDER BY 1, 2"""
    ).fetchall()
    _check(
        report,
        "silver_monthly_price_jumps",
        "silver",
        "warn",
        not jumps,
        "; ".join(f"{p} {m} ({c:+.0%})" for p, m, c in jumps)
        or f"no monthly median moved more than {config.max_monthly_jump:.0%}",
        [{"product": p, "month": m, "change": round(c, 4)} for p, m, c in jumps],
    )
    con.close()


# --------------------------------------------------------------------------- gold


def _gap_ranges(weeks: pd.Series) -> list[tuple[str, str, int]]:
    """Runs of missing Mondays between the first and last week. Returns (first_missing, last_missing, n_weeks)."""
    full = pd.date_range(weeks.min(), weeks.max(), freq="W-MON")
    missing = full.difference(pd.DatetimeIndex(weeks))
    out: list[tuple[str, str, int]] = []
    if not len(missing):
        return out
    start = prev = missing[0]
    for week in missing[1:]:
        if (week - prev).days > 7:
            out.append((start.date().isoformat(), prev.date().isoformat(), (prev - start).days // 7 + 1))
            start = week
        prev = week
    out.append((start.date().isoformat(), prev.date().isoformat(), (prev - start).days // 7 + 1))
    return out


def check_gold(settings: Settings, report: QualityReport, today: date | None = None, config: QualityConfig | None = None) -> None:
    config = config or QualityConfig()
    today = today or datetime.now(UTC).date()
    weekly = pd.read_parquet(settings.gold / "weekly_prices.parquet")
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    _check(report, "gold_not_empty", "gold", "error", len(weekly) > 0, f"{len(weekly):,} weekly rows", len(weekly))

    national = weekly[weekly["state"] == "BR"]
    latest = national["week_start"].max()
    age = (pd.Timestamp(today) - latest).days
    _check(
        report,
        "gold_freshness_error",
        "gold",
        "error",
        age <= config.stale_error_days,
        f"newest week is {latest.date()} ({age} days old, limit {config.stale_error_days})",
        age,
    )
    _check(
        report,
        "gold_freshness_warn",
        "gold",
        "warn",
        age <= config.stale_warn_days,
        f"newest week is {latest.date()} ({age} days old, expected at most {config.stale_warn_days})",
        age,
    )

    gap_report = {}
    for product, group in national.groupby("product"):
        gaps = _gap_ranges(group["week_start"])
        if gaps:
            gap_report[str(product)] = [{"from": a, "to": b, "weeks": n} for a, b, n in gaps]
    total_gap_weeks = sum(g["weeks"] for gaps in gap_report.values() for g in gaps)
    described = []
    for product, gaps in gap_report.items():
        spans = ", ".join("{}..{} ({}w)".format(g["from"], g["to"], g["weeks"]) for g in gaps)
        described.append(f"{product}: {spans}")
    _check(
        report,
        "gold_weekly_coverage",
        "gold",
        "warn",
        not gap_report,
        "; ".join(described) or "no missing weeks in the national series",
        {"missing_weeks": total_gap_weeks, "gaps": gap_report},
    )

    sparse = national[national["stations"] < config.min_stations_per_week]
    _check(
        report,
        "gold_national_sample_size",
        "gold",
        "warn",
        sparse.empty,
        f"{len(sparse)} national weeks have fewer than {config.min_stations_per_week} stations"
        if len(sparse)
        else "every national week has enough stations",
        len(sparse),
    )

    ordered = national.sort_values(["product", "week_start"]).copy()
    ordered["change"] = ordered.groupby("product")["avg_price"].pct_change()
    # a jump across a gap is a data hole, not a price move: only judge consecutive weeks
    ordered["consecutive"] = ordered.groupby("product")["week_start"].diff().dt.days == 7
    jumps = ordered[(ordered["change"].abs() > config.max_weekly_jump) & ordered["consecutive"]]
    _check(
        report,
        "gold_weekly_price_jumps",
        "gold",
        "warn",
        jumps.empty,
        "; ".join(f"{r.product} {r.week_start.date()} ({r.change:+.0%})" for r in jumps.itertuples())
        or f"no national weekly move above {config.max_weekly_jump:.0%}",
        [{"product": r.product, "week": str(r.week_start.date()), "change": round(r.change, 4)} for r in jumps.itertuples()],
    )


# --------------------------------------------------------------------------- entry point


def run_quality(
    settings: Settings,
    layers: tuple[str, ...] = ("bronze", "silver", "gold"),
    today: date | None = None,
    config: QualityConfig | None = None,
    write: bool = True,
) -> QualityReport:
    """Run the checks for the requested layers, persist the report, and return it."""
    report = QualityReport(generated_at=datetime.now(UTC).isoformat())
    if "bronze" in layers:
        check_bronze(settings, report)
    if "silver" in layers:
        check_silver(settings, report, config)
    if "gold" in layers:
        check_gold(settings, report, today, config)
    if write:
        settings.reports.mkdir(parents=True, exist_ok=True)
        path = settings.reports / "quality_report.json"
        merged = report
        if path.exists():  # keep the checks of layers that were not re-run
            previous = json.loads(path.read_text(encoding="utf-8"))
            kept = [Check(**c) for c in previous["checks"] if c["layer"] not in layers]
            merged = QualityReport(report.generated_at, kept + report.checks)
        path.write_text(json.dumps(merged.to_dict(), indent=2, default=str), encoding="utf-8")
    return report
