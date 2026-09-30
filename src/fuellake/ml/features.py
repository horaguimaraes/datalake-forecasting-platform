"""Leak-free feature engineering.

Every feature at week `t` is computed only from weeks <= t (trailing windows and positive lags),
which the test-suite enforces by perturbing the future and checking nothing changes. The target
for horizon h is the log return between t and t+h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import NATIONAL

PRODUCT_CODES = {"GASOLINA": 0, "ETANOL": 1, "DIESEL S10": 2}
KEYS = ["product", "state"]


def _rolling_mean(s: pd.Series, window: int, min_frac: float = 0.75) -> pd.Series:
    return s.rolling(window, min_periods=max(2, int(window * min_frac))).mean()


def make_features(panel: pd.DataFrame, states: list[str] | None = None) -> pd.DataFrame:
    """Feature table, one row per (product, state, week_start) that has a price.

    `states` fixes the categorical encoding so training and inference always agree.
    """
    df = panel.sort_values([*KEYS, "week_start"]).reset_index(drop=True).copy()
    df["log_price"] = np.log(df["price"])
    grouped = df.groupby(KEYS, sort=False)["log_price"]

    df["ret_1"] = grouped.diff()
    df["ret_2"] = df.groupby(KEYS, sort=False)["ret_1"].shift(1)
    df["ret_3"] = df.groupby(KEYS, sort=False)["ret_1"].shift(2)
    df["ret_4"] = df.groupby(KEYS, sort=False)["ret_1"].shift(3)
    for k in (4, 13, 26):
        df[f"mom_{k}"] = df["log_price"] - grouped.shift(k)

    for k in (4, 8, 26, 52):
        mean_price = df.groupby(KEYS, sort=False)["price"].transform(lambda s, k=k: _rolling_mean(s, k))
        df[f"dev_{k}"] = df["log_price"] - np.log(mean_price)

    df["vol_8"] = df.groupby(KEYS, sort=False)["ret_1"].transform(lambda s: s.rolling(8, min_periods=5).std())
    df["log_stations"] = np.log1p(df["stations"])

    # cross-section: how the national price of the same product is moving, and where this state sits
    national = df[df["state"] == NATIONAL][["product", "week_start", "log_price", "ret_1", "mom_4"]].rename(
        columns={"log_price": "nat_log_price", "ret_1": "nat_ret_1", "mom_4": "nat_mom_4"}
    )
    df = df.merge(national, on=["product", "week_start"], how="left")
    df["rel_nat"] = df["log_price"] - df["nat_log_price"]
    df["rel_nat_dev"] = df["rel_nat"] - df.groupby(KEYS, sort=False)["rel_nat"].transform(lambda s: _rolling_mean(s, 26))

    # ethanol / gasoline price ratio in the same state (the "70% rule" signal)
    pivot = df.pivot_table(index=["state", "week_start"], columns="product", values="log_price")
    if {"ETANOL", "GASOLINA"} <= set(pivot.columns):
        ratio = (pivot["ETANOL"] - pivot["GASOLINA"]).rename("eth_gas_ratio").reset_index()
        df = df.merge(ratio, on=["state", "week_start"], how="left")
        df["eth_gas_mom_4"] = df["eth_gas_ratio"] - df.groupby(KEYS, sort=False)["eth_gas_ratio"].shift(4)
    else:
        df["eth_gas_ratio"] = np.nan
        df["eth_gas_mom_4"] = np.nan

    week = df["week_start"].dt.isocalendar().week.astype(float)
    df["woy_sin"] = np.sin(2 * np.pi * week / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * week / 52.0)
    df["month"] = df["week_start"].dt.month.astype(float)

    df["product_code"] = df["product"].map(PRODUCT_CODES).astype(float)
    state_list = states if states is not None else sorted(df["state"].unique())
    df["state_code"] = df["state"].map({s: i for i, s in enumerate(state_list)}).astype(float)
    return df


FEATURE_COLUMNS = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_4",
    "mom_4",
    "mom_13",
    "mom_26",
    "dev_4",
    "dev_8",
    "dev_26",
    "dev_52",
    "vol_8",
    "log_stations",
    "dispersion",
    "nat_ret_1",
    "nat_mom_4",
    "rel_nat",
    "rel_nat_dev",
    "eth_gas_ratio",
    "eth_gas_mom_4",
    "woy_sin",
    "woy_cos",
    "month",
    "product_code",
    "state_code",
]
CATEGORICAL_COLUMNS = ["product_code", "state_code"]


def add_targets(features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add `y` (log return t -> t+h) and `target_week`. Only *observed* target weeks count as truth."""
    df = features.copy()
    g = df.groupby(KEYS, sort=False)
    future_price = g["price"].shift(-horizon)
    future_observed = g["observed"].shift(-horizon)
    df["y"] = np.log(future_price) - df["log_price"]
    df.loc[future_observed != True, "y"] = np.nan  # noqa: E712 - imputed weeks are not ground truth
    df["target_week"] = df["week_start"] + pd.Timedelta(weeks=horizon)
    return df
