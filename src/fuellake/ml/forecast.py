"""Produce forecasts with the registered champion model."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ..config import Settings
from .dataset import PanelConfig, build_panel
from .features import make_features
from .train import latest_run, load_artifact


def make_forecast(settings: Settings, weekly: pd.DataFrame) -> pd.DataFrame:
    """Forecast every active series from its most recent week, for each trained horizon.

    Output columns: product, state, origin_week, target_week, horizon, price_origin, forecast,
    lo, hi (the interval), naive, model, model_run.
    """
    artifact = load_artifact(settings)
    panel = build_panel(
        weekly, PanelConfig(**{**artifact["panel_config"], "products": tuple(artifact["panel_config"]["products"])})
    )
    features = make_features(panel, states=artifact["states"])
    origin_week = features["week_start"].max()
    origins = features[(features["week_start"] == origin_week) & features["price"].notna()]
    if origins.empty:
        raise ValueError("No series has a price in the most recent week; cannot forecast.")

    run_id = latest_run(settings)["run_id"]
    generated = datetime.now(UTC).isoformat()
    frames = []
    for h, model in artifact["models"].items():
        pred = model.predict(origins)
        price = origins["price"].to_numpy()
        lo_q = np.array([artifact["intervals"].get((p, h), (np.nan, np.nan))[0] for p in origins["product"]])
        hi_q = np.array([artifact["intervals"].get((p, h), (np.nan, np.nan))[1] for p in origins["product"]])
        frames.append(
            pd.DataFrame(
                {
                    "product": origins["product"].to_numpy(),
                    "state": origins["state"].to_numpy(),
                    "origin_week": origin_week,
                    "target_week": origin_week + pd.Timedelta(weeks=int(h)),
                    "horizon": int(h),
                    "price_origin": price,
                    "forecast": price * np.exp(pred),
                    "lo": price * np.exp(pred + lo_q),
                    "hi": price * np.exp(pred + hi_q),
                    "naive": price,
                    "model": artifact["champion"],
                    "model_run": run_id,
                    "generated_at": generated,
                }
            )
        )
    out = pd.concat(frames, ignore_index=True).sort_values(["product", "state", "horizon"]).reset_index(drop=True)
    settings.forecasts.mkdir(parents=True, exist_ok=True)
    out.to_parquet(settings.forecasts / "latest.parquet", index=False)
    out.to_parquet(settings.forecasts / f"forecast_{origin_week.date()}.parquet", index=False)
    return out
