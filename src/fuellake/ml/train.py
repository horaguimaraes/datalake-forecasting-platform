"""Train, evaluate and register models.

The registry is a folder per run plus `models/latest.json` pointing at the champion, so a forecast
can always say which model and which data produced it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import joblib
import pandas as pd

from ..config import Settings
from .backtest import (
    HORIZONS,
    bootstrap_skill,
    interval_coverage,
    interval_quantiles,
    run_backtest,
    score,
    summarize,
)
from .dataset import PanelConfig, build_panel
from .features import add_targets, make_features
from .models import new_model

DEFAULT_MODELS = ["naive", "ma4", "ma8", "drift", "ridge", "gbm", "blend"]


@dataclass(frozen=True)
class TrainConfig:
    models: tuple[str, ...] = tuple(DEFAULT_MODELS)
    horizons: tuple[int, ...] = HORIZONS
    n_folds: int = 12
    step_weeks: int = 2
    coverage: float = 0.8
    panel: PanelConfig = field(default_factory=PanelConfig)


def train(
    settings: Settings,
    weekly: pd.DataFrame,
    config: TrainConfig | None = None,
    log: Callable[[str], None] = lambda message: None,
) -> dict[str, Any]:
    config = config or TrainConfig()
    if "naive" not in config.models:
        raise ValueError("The 'naive' baseline is required: every score is reported relative to it.")

    panel = build_panel(weekly, config.panel)
    features = make_features(panel)
    states = sorted(features["state"].unique())
    log(f"  panel: {panel.groupby(['product', 'state']).ngroups} series, {len(panel):,} weekly rows")

    predictions = run_backtest(features, list(config.models), config.horizons, config.n_folds, config.step_weeks, log=log)
    scores = score(predictions)
    ranking = summarize(scores)
    champion = str(ranking.iloc[0]["model"])
    log(f"  champion: {champion} (MAE R$ {ranking.iloc[0]['mae']:.4f}, skill vs naive {ranking.iloc[0]['skill_vs_naive']:+.1%})")

    champion_preds = predictions[predictions["model"] == champion]
    skill_ci = bootstrap_skill(predictions, champion) if champion != "naive" else None
    quantiles = interval_quantiles(champion_preds, config.coverage)
    coverage = interval_coverage(champion_preds, config.coverage)

    fitted = {}
    for h in config.horizons:
        data = add_targets(features, h)
        data = data[data["y"].notna()]
        fitted[h] = new_model(champion).fit(data, data["y"].to_numpy(), h)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.models / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "champion": champion,
            "models": fitted,
            "states": states,
            "panel_config": asdict(config.panel),
            "intervals": quantiles,
            "horizons": list(config.horizons),
            "coverage": config.coverage,
        },
        run_dir / "artifact.joblib",
    )
    predictions.to_parquet(run_dir / "backtest_predictions.parquet")
    metrics = {
        "run_id": run_id,
        "trained_at": datetime.now(UTC).isoformat(),
        "champion": champion,
        "data_through": str(panel["week_start"].max().date()),
        "n_series": int(panel.groupby(["product", "state"]).ngroups),
        "n_weekly_rows": int(len(panel)),
        "backtest": {
            "n_folds": config.n_folds,
            "step_weeks": config.step_weeks,
            "horizons": list(config.horizons),
            "test_origins_from": str(pd.Timestamp(predictions["origin"].min()).date()),
            "test_origins_to": str(pd.Timestamp(predictions["origin"].max()).date()),
        },
        "ranking": ranking.round(6).to_dict(orient="records"),
        "by_horizon": scores.round(6).to_dict(orient="records"),
        "champion_skill_ci": skill_ci,
        "interval_coverage": coverage,
        "config": {**asdict(config), "models": list(config.models)},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (settings.models / "latest.json").write_text(
        json.dumps({"run_id": run_id, "path": run_id, "champion": champion, "data_through": metrics["data_through"]}, indent=2),
        encoding="utf-8",
    )
    return metrics


def latest_run(settings: Settings) -> dict[str, Any] | None:
    pointer = settings.models / "latest.json"
    return json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else None


def load_metrics(settings: Settings) -> dict[str, Any] | None:
    latest = latest_run(settings)
    if not latest:
        return None
    return json.loads((settings.models / latest["path"] / "metrics.json").read_text(encoding="utf-8"))


def load_artifact(settings: Settings) -> dict[str, Any]:
    latest = latest_run(settings)
    if not latest:
        raise FileNotFoundError("No trained model found. Run the pipeline (or its 'train' step) first.")
    return joblib.load(settings.models / latest["path"] / "artifact.joblib")
