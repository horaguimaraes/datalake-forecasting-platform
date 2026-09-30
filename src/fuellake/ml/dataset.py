"""From gold weekly statistics to a clean, regular panel of price series."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MODEL_PRODUCTS = ("GASOLINA", "ETANOL", "DIESEL S10")
NATIONAL = "BR"


@dataclass(frozen=True)
class PanelConfig:
    products: tuple[str, ...] = MODEL_PRODUCTS
    min_days_covered: int = 3  # a week seen on fewer collection days is treated as missing
    max_impute_weeks: int = 6  # linearly bridge gaps up to this long; longer gaps stay missing
    min_stations: int = 10  # state series with a median below this are too noisy to model
    min_history_weeks: int = 40


def build_panel(weekly: pd.DataFrame, config: PanelConfig | None = None) -> pd.DataFrame:
    """One row per (product, state, week) on a complete weekly grid.

    Columns: week_start, product, state, price, observed (a real measurement), imputed (bridged gap),
    stations, dispersion (std / mean within the week).
    Weeks with too little coverage are dropped *before* gap-filling, so a half-observed week is
    never mistaken for a real price.
    """
    config = config or PanelConfig()
    df = weekly[weekly["product"].isin(config.products)].copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df[df["days_covered"] >= config.min_days_covered]

    typical_stations = df.groupby(["product", "state"])["stations"].median()
    eligible = typical_stations[typical_stations >= config.min_stations].index
    df = df.set_index(["product", "state"]).loc[eligible].reset_index()

    last_week = df["week_start"].max()
    frames = []
    for (product, state), group in df.groupby(["product", "state"], sort=True):
        group = group.sort_values("week_start").set_index("week_start")
        grid = pd.date_range(group.index.min(), last_week, freq="W-MON")
        series = group["avg_price"].reindex(grid)
        observed = series.notna()
        filled = series.interpolate(method="linear", limit=config.max_impute_weeks, limit_area="inside")
        frame = pd.DataFrame(
            {
                "week_start": grid,
                "product": product,
                "state": state,
                "price": filled.to_numpy(),
                "observed": observed.to_numpy(),
                "imputed": (filled.notna() & ~observed).to_numpy(),
                "stations": group["stations"].reindex(grid).ffill().to_numpy(),
                "dispersion": (group["std_price"] / group["avg_price"]).reindex(grid).ffill().to_numpy(),
            }
        )
        if observed.sum() >= config.min_history_weeks:
            frames.append(frame)
    if not frames:
        raise ValueError(
            f"No series has at least {config.min_history_weeks} weeks of history. Load more months, or lower min_history_weeks."
        )
    panel = pd.concat(frames, ignore_index=True)
    panel["price"] = panel["price"].astype("float64")
    return panel


def series_keys(panel: pd.DataFrame) -> list[tuple[str, str]]:
    return sorted({(p, s) for p, s in zip(panel["product"], panel["state"], strict=True)})


def active_series(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows of the most recent week per series that actually has a price there (the forecast origins)."""
    last = panel["week_start"].max()
    recent = panel[(panel["week_start"] == last) & panel["price"].notna()]
    return recent
