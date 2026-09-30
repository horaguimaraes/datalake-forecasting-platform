"""Forecasting models. All predict the *log return* from the origin week to `origin + h`.

Predicting a return (instead of a price level) makes one model valid for every product and state,
and turns "forecast the price" into "forecast the change", which is what carries the signal.

Baselines matter as much as the learned models: a forecast is only useful if it beats "next week
looks like this week" (naive) and the simple smoothers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


class ForecastModel(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: np.ndarray, horizon: int) -> ForecastModel: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class Naive:
    """Next weeks look like this week."""

    name: str = "naive"

    def fit(self, X, y, horizon):
        return self

    def predict(self, X):
        return np.zeros(len(X))


@dataclass
class MeanReversion:
    """Predict a return back to the trailing k-week mean. Weekly state averages are noisy (a different
    sample of stations each week), so the smoothed level is often a better guess than the last point."""

    k: int = 4
    name: str = ""

    def __post_init__(self):
        self.name = self.name or f"ma{self.k}"

    def fit(self, X, y, horizon):
        return self

    def predict(self, X):
        return -X[f"dev_{self.k}"].fillna(0.0).to_numpy()


@dataclass
class Drift:
    """Average historical return for the horizon (captures inflation-like upward drift)."""

    name: str = "drift"
    mean_: float = 0.0

    def fit(self, X, y, horizon):
        self.mean_ = float(np.nanmean(y)) if len(y) else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


@dataclass
class RidgeModel:
    name: str = "ridge"
    alpha: float = 10.0
    pipeline: object = field(default=None, repr=False)

    def fit(self, X, y, horizon):
        cols = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
        self.cols_ = cols
        self.product_dummies_ = sorted(X["product_code"].dropna().unique())
        self.pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=self.alpha))
        self.pipeline.fit(self._design(X), y)
        return self

    def _design(self, X):
        base = X[self.cols_].to_numpy(dtype=float)
        dummies = np.column_stack([(X["product_code"] == p).to_numpy(dtype=float) for p in self.product_dummies_])
        return np.hstack([base, dummies])

    def predict(self, X):
        return self.pipeline.predict(self._design(X))


@dataclass
class GradientBoosting:
    name: str = "gbm"
    max_iter: int = 150
    learning_rate: float = 0.05
    max_depth: int = 4
    min_samples_leaf: int = 60
    random_state: int = 0
    model: object = field(default=None, repr=False)

    def fit(self, X, y, horizon):
        categorical = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_COLUMNS]
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error",  # robust: price series have occasional shocks
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=1.0,
            categorical_features=categorical,
            early_stopping=False,  # a random validation split would leak the future in time series
            random_state=self.random_state,
        )
        self.model.fit(self._design(X), y)
        return self

    @staticmethod
    def _design(X):
        return X[FEATURE_COLUMNS].to_numpy(dtype=float)

    def predict(self, X):
        return self.model.predict(self._design(X))


@dataclass
class Blend:
    """Equal-weight average of the ridge and gradient-boosting predictions."""

    name: str = "blend"
    members: list = field(default_factory=lambda: [RidgeModel(), GradientBoosting()])

    def fit(self, X, y, horizon):
        for member in self.members:
            member.fit(X, y, horizon)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)


def default_models() -> dict[str, ForecastModel]:
    """Fresh (unfitted) instances, keyed by name."""
    models: list[ForecastModel] = [
        Naive(),
        MeanReversion(4),
        MeanReversion(8),
        Drift(),
        RidgeModel(),
        GradientBoosting(),
        Blend(),
    ]
    return {m.name: m for m in models}


def new_model(name: str) -> ForecastModel:
    factory = {
        "naive": Naive,
        "ma4": lambda: MeanReversion(4),
        "ma8": lambda: MeanReversion(8),
        "drift": Drift,
        "ridge": RidgeModel,
        "gbm": GradientBoosting,
        "blend": Blend,
    }
    if name not in factory:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(factory)}")
    return factory[name]()
