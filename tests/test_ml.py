from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fuellake.config import Settings
from fuellake.gold import read_weekly_prices
from fuellake.ml.backtest import (
    bootstrap_skill,
    cutoffs,
    interval_coverage,
    interval_quantiles,
    run_backtest,
    score,
    split_fold,
    summarize,
)
from fuellake.ml.dataset import PanelConfig, build_panel
from fuellake.ml.features import FEATURE_COLUMNS, add_targets, make_features
from fuellake.ml.models import Drift, GradientBoosting, MeanReversion, Naive, default_models, new_model
from fuellake.ml.train import TrainConfig, load_metrics, train
from fuellake.pipeline import RunParams, build_pipeline
from tests.synth import SynthConfig, write_monthly_files


def weekly_frame(weeks, values, product="GASOLINA", state="SP", stations=50, days=5):
    return pd.DataFrame(
        {
            "week_start": pd.to_datetime(weeks),
            "state": state,
            "product": product,
            "avg_price": values,
            "median_price": values,
            "min_price": values,
            "max_price": values,
            "std_price": 0.1,
            "observations": stations,
            "stations": stations,
            "days_covered": days,
        }
    )


MONDAYS = pd.date_range("2024-01-01", periods=60, freq="W-MON")


class TestPanel:
    cfg = PanelConfig(min_history_weeks=10)

    def test_regular_weekly_grid_with_no_gaps_is_untouched(self):
        panel = build_panel(weekly_frame(MONDAYS, np.linspace(5, 6, 60)), self.cfg)
        assert len(panel) == 60 and panel["observed"].all() and not panel["imputed"].any()

    def test_short_gaps_are_bridged_and_flagged_but_never_counted_as_observed(self):
        weeks = MONDAYS.delete([20, 21, 22])
        values = np.linspace(5, 6, 60)[[i for i in range(60) if i not in (20, 21, 22)]]
        panel = build_panel(weekly_frame(weeks, values), self.cfg)
        bridged = panel[panel["imputed"]]
        assert len(bridged) == 3 and not bridged["observed"].any() and bridged["price"].notna().all()
        assert bridged["price"].is_monotonic_increasing  # linear bridge between the neighbours

    def test_long_gaps_stay_missing(self):
        weeks = MONDAYS.delete(range(20, 30))  # 10 weeks > max_impute_weeks
        panel = build_panel(weekly_frame(weeks, np.linspace(5, 6, 50)), self.cfg)
        assert panel["price"].isna().sum() == 10 - 6  # only the first 6 weeks of the hole get bridged

    def test_weeks_seen_on_too_few_days_are_treated_as_missing(self):
        df = weekly_frame(MONDAYS, np.linspace(5, 6, 60))
        df.loc[30, "days_covered"] = 1
        panel = build_panel(df, self.cfg)
        row = panel[panel["week_start"] == MONDAYS[30]].iloc[0]
        assert not row["observed"] and row["imputed"]

    def test_noisy_small_series_and_short_histories_are_excluded(self):
        small = weekly_frame(MONDAYS, np.linspace(5, 6, 60), state="AC", stations=3)
        big = weekly_frame(MONDAYS, np.linspace(5, 6, 60), state="SP")
        panel = build_panel(pd.concat([small, big]), self.cfg)
        assert set(panel["state"]) == {"SP"}
        with pytest.raises(ValueError, match="at least"):
            build_panel(weekly_frame(MONDAYS[:5], np.ones(5)), self.cfg)

    def test_only_modelled_products_are_kept(self):
        df = pd.concat([weekly_frame(MONDAYS, np.ones(60) * 5), weekly_frame(MONDAYS, np.ones(60) * 5, product="GNV")])
        assert set(build_panel(df, self.cfg)["product"]) == {"GASOLINA"}


@pytest.fixture(scope="module")
def panel(lake):
    return build_panel(read_weekly_prices(lake))


@pytest.fixture(scope="module")
def features(panel):
    return make_features(panel)


class TestFeatures:
    def test_features_never_use_the_future(self, panel):
        """Change every price after week T; features at or before T must be identical."""
        cut = panel["week_start"].sort_values().unique()[80]
        rng = np.random.default_rng(1)
        perturbed = panel.copy()
        later = perturbed["week_start"] > cut
        perturbed.loc[later, "price"] = perturbed.loc[later, "price"] * rng.uniform(0.5, 2.0, later.sum())
        base, changed = make_features(panel), make_features(perturbed)
        key = ["product", "state", "week_start"]
        a = base[base["week_start"] <= cut].sort_values(key)[FEATURE_COLUMNS].reset_index(drop=True)
        b = changed[changed["week_start"] <= cut].sort_values(key)[FEATURE_COLUMNS].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_target_is_the_log_return_to_h_weeks_ahead(self, features):
        data = add_targets(features, 3)
        one = data[(data["product"] == "GASOLINA") & (data["state"] == "SP")].reset_index(drop=True)
        i = 50
        assert one.loc[i, "y"] == pytest.approx(np.log(one.loc[i + 3, "price"] / one.loc[i, "price"]))
        assert one.loc[i, "target_week"] == one.loc[i + 3, "week_start"]

    def test_the_last_h_weeks_have_no_target(self, features):
        data = add_targets(features, 4)
        last = data.groupby(["product", "state"])["week_start"].transform("max")
        assert data.loc[data["week_start"] > last - pd.Timedelta(weeks=4), "y"].isna().all()

    def test_a_bridged_week_is_never_used_as_ground_truth(self):
        weeks = MONDAYS.delete([30])
        values = np.linspace(5, 6, 59)
        f = make_features(build_panel(weekly_frame(weeks, values), PanelConfig(min_history_weeks=10)))
        data = add_targets(f, 1)
        origin = data[data["week_start"] == MONDAYS[29]].iloc[0]  # its target week (30) was bridged
        assert np.isnan(origin["y"])

    def test_categorical_encoding_is_fixed_by_the_state_list(self, panel):
        states = sorted(panel["state"].unique())
        assert set(make_features(panel, states)["state_code"].dropna()) <= set(range(len(states)))


class TestBacktest:
    def test_split_never_lets_training_see_past_the_cutoff(self, features):
        data = add_targets(features, 4)
        data = data[data["y"].notna()]
        for cutoff in cutoffs(features, 4, n_folds=8, step_weeks=2):
            train_rows, test_rows = split_fold(data, cutoff, 2)
            assert train_rows["target_week"].max() <= cutoff
            assert test_rows["week_start"].min() > cutoff
            assert test_rows["week_start"].max() <= cutoff + pd.Timedelta(weeks=2)

    def test_the_last_fold_ends_at_the_last_origin_with_a_known_target(self, features):
        h, step = 3, 2
        last_cutoff = cutoffs(features, h, n_folds=5, step_weeks=step)[-1]
        last_origin_with_target = features["week_start"].max() - pd.Timedelta(weeks=h)
        assert last_cutoff + pd.Timedelta(weeks=step) == last_origin_with_target

    def test_predictions_are_out_of_sample_and_complete(self, features):
        preds = run_backtest(features, ["naive", "ma4"], horizons=(1, 2), n_folds=4, step_weeks=2)
        assert set(preds["model"]) == {"naive", "ma4"} and set(preds["horizon"]) == {1, 2}
        assert (preds["origin"] > preds["cutoff"]).all()
        assert preds.groupby("model").size().nunique() == 1  # every model is scored on identical rows

    def test_not_enough_history_is_an_error_not_an_empty_result(self, features):
        with pytest.raises(ValueError, match="no folds"):
            run_backtest(features.head(30), ["naive"], horizons=(1,), n_folds=3, step_weeks=2)

    def test_score_computes_mae_mape_and_skill_by_hand(self):
        rows = []
        for model, pred in (("naive", 0.0), ("good", np.log(1.02)), ("bad", np.log(1.20))):
            rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "horizon": 1,
                        "cutoff": pd.Timestamp("2024-01-01"),
                        "product": "GASOLINA",
                        "state": "SP",
                        "origin": pd.Timestamp("2024-01-08"),
                        "target_week": pd.Timestamp("2024-01-15"),
                        "price_origin": [10.0, 10.0],
                        "y_true": [np.log(1.03), np.log(1.03)],
                        "y_pred": pred,
                    }
                )
            )
        table = score(pd.concat(rows)).set_index("model")
        assert table.loc["naive", "mae"] == pytest.approx(0.3) and table.loc["naive", "skill_vs_naive"] == 0
        assert table.loc["good", "mae"] == pytest.approx(0.1) and table.loc["good", "skill_vs_naive"] == pytest.approx(
            1 - 0.1 / 0.3
        )
        assert table.loc["bad", "skill_vs_naive"] < 0
        assert table.loc["naive", "mape"] == pytest.approx(0.3 / 10.3)
        assert summarize(score(pd.concat(rows))).iloc[0]["model"] == "good"

    def test_bootstrap_ci_is_positive_for_a_real_edge_and_zero_for_a_copy_of_naive(self):
        rng = np.random.default_rng(0)
        origins = pd.date_range("2024-01-01", periods=30, freq="W-MON")
        truth = rng.normal(0, 0.03, len(origins))
        frames = []
        for model, pred in (("naive", np.zeros(len(origins))), ("edge", truth * 0.6), ("copy", np.zeros(len(origins)))):
            frames.append(
                pd.DataFrame(
                    {"model": model, "horizon": 1, "origin": origins, "price_origin": 6.0, "y_true": truth, "y_pred": pred}
                )
            )
        preds = pd.concat(frames)
        edge, same = bootstrap_skill(preds, "edge"), bootstrap_skill(preds, "copy")
        assert edge["lo"] > 0 and edge["lo"] < edge["skill"] < edge["hi"] and edge["n_weeks"] == 30
        assert same["skill"] == pytest.approx(0) and same["lo"] == pytest.approx(0) and same["hi"] == pytest.approx(0)

    def test_interval_quantiles_and_honest_coverage_on_well_behaved_residuals(self):
        rng = np.random.default_rng(3)
        n_cut, per = 12, 400
        df = pd.DataFrame(
            {
                "cutoff": np.repeat(pd.date_range("2024-01-01", periods=n_cut, freq="2W-MON"), per),
                "product": "GASOLINA",
                "horizon": 1,
                "y_pred": 0.0,
                "y_true": rng.normal(0, 0.02, n_cut * per),
            }
        )
        q = interval_quantiles(df, 0.8)[("GASOLINA", 1)]
        assert q[0] == pytest.approx(-1.2816 * 0.02, abs=0.004) and q[1] == pytest.approx(1.2816 * 0.02, abs=0.004)
        cov = interval_coverage(df, 0.8)
        assert cov["empirical"] == pytest.approx(0.8, abs=0.04) and cov["n"] > 1000

    def test_quantiles_are_not_reported_from_tiny_samples(self):
        df = pd.DataFrame({"product": "GASOLINA", "horizon": 1, "y_true": np.arange(5.0), "y_pred": 0.0})
        assert interval_quantiles(df) == {}


class TestModels:
    def make_x(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        x = pd.DataFrame(rng.normal(0, 1, (n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        x["product_code"] = rng.integers(0, 3, n).astype(float)
        x["state_code"] = rng.integers(0, 6, n).astype(float)
        x.iloc[::7, 2] = np.nan  # models must tolerate missing features
        return x

    def test_baselines(self):
        x = self.make_x()
        assert (Naive().fit(x, np.zeros(len(x)), 1).predict(x) == 0).all()
        assert np.allclose(MeanReversion(4).predict(x), -x["dev_4"].fillna(0))
        assert Drift().fit(x, np.array([0.1, 0.3]), 1).predict(x[:3]).tolist() == [pytest.approx(0.2)] * 3

    @pytest.mark.parametrize("name", ["ridge", "gbm", "blend"])
    def test_learned_models_fit_predict_and_handle_missing_values(self, name):
        x = self.make_x()
        y = 0.5 * x["ret_1"].fillna(0).to_numpy() + np.random.default_rng(1).normal(0, 0.1, len(x))
        pred = new_model(name).fit(x, y, 1).predict(x)
        assert pred.shape == (len(x),) and np.isfinite(pred).all()
        assert np.corrcoef(pred, y)[0, 1] > 0.5  # learned the planted signal

    def test_gradient_boosting_is_deterministic(self):
        x = self.make_x()
        y = x["ret_2"].fillna(0).to_numpy()
        a = GradientBoosting().fit(x, y, 1).predict(x)
        b = GradientBoosting().fit(x, y, 1).predict(x)
        assert np.allclose(a, b)

    def test_registry_of_models(self):
        assert {"naive", "ma4", "ma8", "drift", "ridge", "gbm", "blend"} == set(default_models())
        with pytest.raises(ValueError, match="Unknown model"):
            new_model("prophet")


class TestTrainingOnSyntheticData:
    def test_learned_model_beats_naive_when_there_is_structure_to_learn(self, lake):
        m = load_metrics(lake)
        ranking = {r["model"]: r for r in m["ranking"]}
        assert m["champion"] in {"blend", "gbm", "ridge", "ma4"}, m["champion"]
        assert ranking["naive"]["skill_vs_naive"] == 0
        assert ranking[m["champion"]]["skill_vs_naive"] > 0.05
        assert m["champion_skill_ci"]["lo"] > 0  # not just a lucky point estimate

    def test_registry_layout_and_pointer(self, lake):
        latest = json.loads((lake.models / "latest.json").read_text())
        run_dir = lake.models / latest["path"]
        assert {p.name for p in run_dir.iterdir()} == {"artifact.joblib", "metrics.json", "backtest_predictions.parquet"}
        assert latest["champion"] == load_metrics(lake)["champion"]

    def test_interval_coverage_is_in_a_sane_band(self, lake):
        cov = load_metrics(lake)["interval_coverage"]
        assert cov["n"] > 100 and 0.5 < cov["empirical"] < 0.99

    def test_the_naive_baseline_is_mandatory(self, lake):
        with pytest.raises(ValueError, match="naive"):
            train(lake, read_weekly_prices(lake), TrainConfig(models=("ridge",)))

    def test_without_structure_the_pipeline_refuses_to_promote_a_learned_model(self, tmp_path):
        """Prices with no sampling noise are a pure random walk with drift: nothing beats 'no change'."""
        raw = tmp_path / "raw"
        write_monthly_files(
            raw, SynthConfig(station_noise=0.0, weeks=110, stations_per_state=20, states=("SP", "RJ", "MG", "PR"))
        )
        settings = Settings(tmp_path / "lake")
        params = RunParams(local_dir=raw, models=("naive", "ridge", "gbm"), horizons=(1, 2), n_folds=10, as_of=date(2024, 12, 1))
        run = build_pipeline(settings, params, log=lambda m: None).run()
        assert run.ok, {n: r.error for n, r in run.results.items() if r.error}
        m = load_metrics(settings)
        assert m["champion"] == "naive" and m["champion_skill_ci"] is None
