"""Rolling-origin (forward-chaining) evaluation.

For each cutoff week `c` a model is trained only on rows whose *target* was already known by `c`
(`origin + h <= c`) and tested on origins in the following weeks. Training on any later
information would inflate the scores, so the split is written down in one place and tested.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .features import add_targets
from .models import new_model

HORIZONS = tuple(range(1, 9))


def cutoffs(features: pd.DataFrame, horizon: int, n_folds: int, step_weeks: int) -> list[pd.Timestamp]:
    """Cutoff weeks, oldest first. The last fold's test window ends at the last origin with a known target."""
    last_origin = features["week_start"].max() - pd.Timedelta(weeks=horizon)
    return [last_origin - pd.Timedelta(weeks=step_weeks * (n_folds - i)) for i in range(n_folds)]


def split_fold(data: pd.DataFrame, cutoff: pd.Timestamp, step_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on rows whose target was already known at `cutoff`; test on origins in the next `step_weeks`.

    This is the single place that decides what a model may see, so leakage is impossible by construction
    and easy to test: max(train.target_week) <= cutoff < min(test.week_start).
    """
    train = data[data["target_week"] <= cutoff]
    test = data[(data["week_start"] > cutoff) & (data["week_start"] <= cutoff + pd.Timedelta(weeks=step_weeks))]
    return train, test


def run_backtest(
    features: pd.DataFrame,
    model_names: list[str],
    horizons: tuple[int, ...] = HORIZONS,
    n_folds: int = 12,
    step_weeks: int = 2,
    min_train_rows: int = 300,
    log: Callable[[str], None] = lambda message: None,
) -> pd.DataFrame:
    """Out-of-sample predictions for every model, horizon and fold (long format)."""
    rows = []
    for h in horizons:
        data = add_targets(features, h)
        data = data[data["y"].notna()]
        for cutoff in cutoffs(features, h, n_folds, step_weeks):
            train, test = split_fold(data, cutoff, step_weeks)
            if len(train) < min_train_rows or test.empty:
                continue
            for name in model_names:
                model = new_model(name).fit(train, train["y"].to_numpy(), h)
                pred = model.predict(test)
                rows.append(
                    pd.DataFrame(
                        {
                            "model": name,
                            "horizon": h,
                            "cutoff": cutoff,
                            "product": test["product"].to_numpy(),
                            "state": test["state"].to_numpy(),
                            "origin": test["week_start"].to_numpy(),
                            "target_week": test["target_week"].to_numpy(),
                            "price_origin": test["price"].to_numpy(),
                            "y_true": test["y"].to_numpy(),
                            "y_pred": pred,
                        }
                    )
                )
        log(f"  backtest horizon {h}: done")
    if not rows:
        raise ValueError("Backtest produced no folds: not enough history for the requested folds and horizons.")
    return pd.concat(rows, ignore_index=True)


def score(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE (R$), MAPE and skill vs the naive baseline, per model and horizon."""
    p = predictions.copy()
    truth = p["price_origin"] * np.exp(p["y_true"])
    guess = p["price_origin"] * np.exp(p["y_pred"])
    p["abs_err"] = (guess - truth).abs()
    p["ape"] = p["abs_err"] / truth
    table = (
        p.groupby(["model", "horizon"]).agg(mae=("abs_err", "mean"), mape=("ape", "mean"), n=("abs_err", "size")).reset_index()
    )
    naive = table[table["model"] == "naive"].set_index("horizon")["mae"]
    table["skill_vs_naive"] = 1 - table["mae"] / table["horizon"].map(naive)
    return table


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per model: mean MAE/MAPE over horizons and mean skill vs naive."""
    return (
        scores.groupby("model")
        .agg(mae=("mae", "mean"), mape=("mape", "mean"), skill_vs_naive=("skill_vs_naive", "mean"))
        .sort_values("mae")
        .reset_index()
    )


def interval_quantiles(predictions: pd.DataFrame, coverage: float = 0.8) -> dict[tuple[str, int], tuple[float, float]]:
    """Empirical residual quantiles in log-return space, per (product, horizon)."""
    lo_q, hi_q = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    resid = predictions["y_true"] - predictions["y_pred"]
    out = {}
    for (product, h), idx in predictions.groupby(["product", "horizon"]).groups.items():
        r = resid.loc[idx].dropna()
        if len(r) >= 20:
            out[(str(product), int(h))] = (float(r.quantile(lo_q)), float(r.quantile(hi_q)))
    return out


def interval_coverage(predictions: pd.DataFrame, coverage: float = 0.8, holdout_share: float = 1 / 3) -> dict[str, float]:
    """Honest coverage: quantiles are learned on the *earlier* folds and tested on the *later* ones."""
    cutoffs_sorted = sorted(predictions["cutoff"].unique())
    split = cutoffs_sorted[int(len(cutoffs_sorted) * (1 - holdout_share))]
    early, late = predictions[predictions["cutoff"] < split], predictions[predictions["cutoff"] >= split]
    quantiles = interval_quantiles(early, coverage)
    inside, total = 0, 0
    for row in late.itertuples():
        q = quantiles.get((row.product, int(row.horizon)))
        if q is None:
            continue
        resid = row.y_true - row.y_pred
        inside += q[0] <= resid <= q[1]
        total += 1
    return {"nominal": coverage, "empirical": inside / total if total else float("nan"), "n": total}


def bootstrap_skill(
    predictions: pd.DataFrame, model: str, n_boot: int = 500, seed: int = 0, ci: float = 0.95
) -> dict[str, float]:
    """Skill vs naive with a confidence interval.

    Test rows from the same origin week are strongly correlated (one shock hits every state), so the
    bootstrap resamples *weeks*, not rows. Otherwise the interval would be far too optimistic.
    """
    p = predictions[predictions["model"].isin([model, "naive"])].copy()
    truth = p["price_origin"] * np.exp(p["y_true"])
    p["abs_err"] = (p["price_origin"] * np.exp(p["y_pred"]) - truth).abs()
    # one row per (origin week, horizon, model): sum of errors and count
    agg = p.groupby(["origin", "horizon", "model"])["abs_err"].agg(["sum", "count"]).unstack("model")
    agg = agg.dropna()
    origins = agg.index.get_level_values("origin").unique()
    by_origin = {o: agg.xs(o, level="origin") for o in origins}

    def skill(chosen: list) -> float:
        frame = pd.concat([by_origin[o] for o in chosen])
        mae_model = frame[("sum", model)].sum() / frame[("count", model)].sum()
        mae_naive = frame[("sum", "naive")].sum() / frame[("count", "naive")].sum()
        return 1 - mae_model / mae_naive

    rng = np.random.default_rng(seed)
    point = skill(list(origins))
    draws = [skill(list(rng.choice(origins, size=len(origins), replace=True))) for _ in range(n_boot)]
    tail = (1 - ci) / 2
    return {
        "skill": float(point),
        "lo": float(np.quantile(draws, tail)),
        "hi": float(np.quantile(draws, 1 - tail)),
        "n_weeks": int(len(origins)),
    }
