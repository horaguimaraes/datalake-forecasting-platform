"""Command line interface: `fuellake run | demo | status | sql | graph | serve`."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .config import Settings
from .pipeline import RunParams, build_pipeline

DEMO_AS_OF = date(2026, 9, 30)  # the bundled sample ends here; freshness is judged relative to it


def parse_horizons(text: str) -> tuple[int, ...]:
    """'1-8' -> 1..8, '1,2,4' -> (1, 2, 4)."""
    if "-" in text:
        low, high = text.split("-", 1)
        values = range(int(low), int(high) + 1)
    else:
        values = [int(part) for part in text.split(",")]
    result = tuple(values)
    if not result or min(result) < 1:
        raise ValueError(f"Invalid horizons '{text}': use e.g. 1-8 or 1,2,4")
    return result


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.from_env(args.root)


def cmd_run(args: argparse.Namespace) -> int:
    settings = _settings(args)
    params = RunParams(
        start=args.start,
        end=args.end,
        local_dir=Path(args.local) if args.local else None,
        models=tuple(args.models.split(",")) if args.models else RunParams().models,
        horizons=parse_horizons(args.horizons) if args.horizons else RunParams().horizons,
        n_folds=args.folds,
        force=args.force,
        as_of=date.fromisoformat(args.as_of) if args.as_of else None,
    )
    pipeline = build_pipeline(settings, params, log=print)
    only = args.only.split(",") if args.only else None
    print(f"Lake: {settings.root.resolve()}")
    run = pipeline.run(only=only, start_from=args.from_task, run_dir=settings.runs, log=print)
    print()
    for name, result in run.results.items():
        print(f"  {result.status:8} {name:15} {result.duration_s:6.1f}s" + (f"  {result.error}" if result.error else ""))
    if run.ok:
        print(f"\nDone. Report: {settings.reports / 'forecast_report.html'}")
    return 0 if run.ok else 1


def cmd_demo(args: argparse.Namespace) -> int:
    """Offline end-to-end run on the bundled sample."""
    sample = Path(args.sample)
    if not sample.exists():
        print(f"Sample data not found at {sample}. Run from the repository root.", file=sys.stderr)
        return 1
    args.local, args.as_of = str(sample), DEMO_AS_OF.isoformat()
    args.start, args.end, args.models, args.folds, args.force, args.horizons = "2023-01", None, None, 8, False, None
    args.only = args.from_task = None
    return cmd_run(args)


def cmd_status(args: argparse.Namespace) -> int:
    from .bronze import Manifest
    from .ml.train import load_metrics

    s = _settings(args)
    manifest = Manifest.load(s.bronze_manifest)
    print(f"Lake: {s.root.resolve()}")
    covered = {m for e in manifest.ok_entries() for m in e.get("covers", [])}
    gaps = sorted({m for e in manifest.missing_entries() for m in e.get("covers", [])} - covered)
    print(f"bronze: {len(manifest.ok_entries())} files" + (f", not published by ANP: {', '.join(gaps)}" if gaps else ""))
    if s.silver_metrics.exists():
        m = json.loads(s.silver_metrics.read_text(encoding="utf-8"))
        print(f"silver: {m['rows_silver']:,} rows, {m['date_from']} to {m['date_to']}, {m['stations']:,} stations")
    metrics = load_metrics(s)
    if metrics:
        print(f"model:  {metrics['champion']} (run {metrics['run_id']}, data through {metrics['data_through']})")
    runs = sorted(s.runs.glob("*.json")) if s.runs.exists() else []
    if runs:
        last = json.loads(runs[-1].read_text(encoding="utf-8"))
        print(f"last run: {last['run_id']} ok={last['ok']}")
    return 0


def cmd_sql(args: argparse.Namespace) -> int:
    from .gold import open_lake

    con = open_lake(_settings(args))
    print(con.execute(args.query).fetchdf().to_string(index=False))
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    print(build_pipeline(_settings(args), RunParams()).to_mermaid())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(_settings(args)), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fuellake", description="Fuel price data lake and forecasting platform.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_root(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", help="lake directory (default: $FUELLAKE_ROOT or data/lake)")

    run = sub.add_parser("run", help="run the pipeline (or part of it)")
    add_root(run)
    run.add_argument("--start", default="2023-01", help="first month to load, YYYY-MM (default 2023-01)")
    run.add_argument("--end", help="last month to load, YYYY-MM (default: last complete month)")
    run.add_argument("--local", help="offline: read raw ANP files from this folder instead of downloading")
    run.add_argument("--only", help="comma-separated task names to run")
    run.add_argument("--from", dest="from_task", help="run this task and everything downstream")
    run.add_argument("--models", help="comma-separated models to backtest (must include naive)")
    run.add_argument("--horizons", help="forecast horizons in weeks, e.g. 1-8 or 1,2,4 (default 1-8)")
    run.add_argument("--folds", type=int, default=12, help="backtest folds (default 12)")
    run.add_argument("--force", action="store_true", help="re-download files even if cached")
    run.add_argument("--as-of", help="treat this date (YYYY-MM-DD) as today for freshness checks")
    run.set_defaults(func=cmd_run)

    demo = sub.add_parser("demo", help="offline end-to-end run on the bundled sample data")
    add_root(demo)
    demo.add_argument("--sample", default="data/sample", help="folder with the sample raw files")
    demo.set_defaults(func=cmd_demo, root="data/demo_lake")

    status = sub.add_parser("status", help="what is in the lake")
    add_root(status)
    status.set_defaults(func=cmd_status)

    sql = sub.add_parser("sql", help="run SQL over the lake (views: silver_prices, weekly_prices, stations, forecasts)")
    add_root(sql)
    sql.add_argument("query")
    sql.set_defaults(func=cmd_sql)

    graph = sub.add_parser("graph", help="print the pipeline DAG as a Mermaid diagram")
    add_root(graph)
    graph.set_defaults(func=cmd_graph)

    serve = sub.add_parser("serve", help="start the REST API")
    add_root(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import duckdb

    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError, duckdb.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
