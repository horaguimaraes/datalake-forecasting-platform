from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fuellake.api import create_app
from fuellake.cli import main, parse_horizons
from fuellake.gold import read_weekly_prices
from fuellake.ml.forecast import make_forecast
from fuellake.ml.train import load_metrics
from fuellake.pipeline import default_end
from tests.conftest import HORIZONS


@pytest.fixture(scope="module")
def forecast(lake) -> pd.DataFrame:
    return pd.read_parquet(lake.forecasts / "latest.parquet")


class TestForecast:
    def test_one_row_per_series_and_horizon(self, lake, forecast):
        n_series = forecast.groupby(["product", "state"]).ngroups
        assert n_series == 3 * 7  # 3 products x (6 states + national)
        assert len(forecast) == n_series * len(HORIZONS)
        assert set(forecast["horizon"]) == set(HORIZONS)

    def test_dates_line_up_with_the_horizon(self, forecast):
        origin = forecast["origin_week"].max()
        assert forecast["origin_week"].nunique() == 1
        assert (forecast["target_week"] == origin + pd.to_timedelta(forecast["horizon"] * 7, unit="D")).all()

    def test_values_are_sane_and_intervals_bracket_the_forecast(self, forecast):
        assert (forecast[["forecast", "lo", "hi", "naive"]] > 0).all().all()
        assert (forecast["lo"] <= forecast["forecast"]).all() and (forecast["forecast"] <= forecast["hi"]).all()
        widths = forecast.groupby("horizon").apply(lambda g: (g["hi"] - g["lo"]).mean(), include_groups=False)
        assert (widths > 0).all()  # (not necessarily growing: with noise-dominated data error does not grow with horizon)
        assert (forecast["forecast"] / forecast["price_origin"] - 1).abs().max() < 0.25  # no absurd jumps

    def test_records_which_model_and_run_produced_it(self, lake, forecast):
        assert set(forecast["model"]) == {load_metrics(lake)["champion"]}
        assert set(forecast["model_run"]) == {load_metrics(lake)["run_id"]}

    def test_forecasting_again_is_reproducible(self, lake, forecast):
        again = make_forecast(lake, read_weekly_prices(lake))
        np.testing.assert_allclose(again["forecast"].to_numpy(), forecast["forecast"].to_numpy(), rtol=1e-9)

    def test_forecast_without_a_trained_model_is_a_clear_error(self, empty_lake):
        with pytest.raises(FileNotFoundError, match="No trained model"):
            make_forecast(empty_lake, pd.DataFrame())


@pytest.fixture(scope="module")
def client(lake) -> TestClient:
    return TestClient(create_app(lake, api_key=""))


class TestApi:
    def test_health_reports_every_layer(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok" and all(body["layers"].values())

    def test_series_lists_every_forecastable_series(self, client):
        body = client.get("/v1/series").json()
        assert len(body["data"]) == 21 and {"product", "state", "price_origin"} <= set(body["data"][0])

    def test_history_is_ordered_and_case_insensitive(self, client):
        r = client.get("/v1/history", params={"product": "gasolina", "state": "sp", "weeks": 20})
        rows = r.json()
        assert r.status_code == 200 and len(rows) == 20
        assert [x["week_start"] for x in rows] == sorted(x["week_start"] for x in rows)
        assert rows[0]["avg_price"] > 0 and rows[0]["stations"] > 0

    def test_national_is_the_default_state(self, client):
        assert client.get("/v1/forecast", params={"product": "GASOLINA"}).json()["state"] == "BR"

    def test_forecast_payload(self, client):
        body = client.get("/v1/forecast", params={"product": "ETANOL", "state": "RJ"}).json()
        assert body["product"] == "ETANOL" and len(body["points"]) == len(HORIZONS)
        first = body["points"][0]
        assert first["horizon"] == 1 and first["lo"] <= first["forecast"] <= first["hi"]
        assert "80%" in body["interval"] and body["model_run"]

    def test_unknown_series_are_404_with_hints(self, client):
        r = client.get("/v1/forecast", params={"product": "BANANA"})
        assert r.status_code == 404 and "Products" in r.json()["detail"]
        assert client.get("/v1/history", params={"product": "GASOLINA", "state": "ZZ"}).status_code == 404

    @pytest.mark.parametrize(
        "path,params",
        [
            ("/v1/history", {}),
            ("/v1/history", {"product": "GASOLINA", "weeks": 0}),
            ("/v1/history", {"product": "GASOLINA", "state": "SPX"}),
            ("/v1/forecast/movers", {"horizon": 99}),
            ("/v1/forecast/movers", {"direction": "sideways"}),
        ],
    )
    def test_invalid_parameters_are_422(self, client, path, params):
        assert client.get(path, params=params).status_code == 422

    def test_movers_are_sorted_in_the_requested_direction(self, client):
        up = client.get("/v1/forecast/movers", params={"horizon": 2, "direction": "up", "limit": 5}).json()
        down = client.get("/v1/forecast/movers", params={"horizon": 2, "direction": "down", "limit": 5}).json()
        assert [m["change_pct"] for m in up] == sorted((m["change_pct"] for m in up), reverse=True)
        assert [m["change_pct"] for m in down] == sorted(m["change_pct"] for m in down)
        assert all(m["state"] != "BR" for m in up + down)

    def test_models_endpoint_exposes_the_evidence_not_just_the_answer(self, client, lake):
        body = client.get("/v1/models").json()
        assert body["champion"] == load_metrics(lake)["champion"] and body["ranking"][0]["model"] == body["champion"]
        assert {"lo", "hi"} <= set(body["champion_skill_ci"]) and "empirical" in body["interval_coverage"]

    def test_quality_and_lineage(self, client):
        assert client.get("/v1/quality").json()["summary"]["errors"] == 0
        lineage = client.get("/v1/lineage").json()
        assert lineage["bronze"]["files"] > 0 and lineage["silver"]["rows_silver"] > 10_000
        assert all(len(s["sha256"]) == 64 for s in lineage["bronze"]["sources"])

    def test_an_empty_lake_answers_503_not_500(self, empty_lake):
        c = TestClient(create_app(empty_lake, api_key=""))
        assert c.get("/health").json()["layers"]["forecast"] is False
        assert c.get("/v1/forecast", params={"product": "GASOLINA"}).status_code == 503
        assert c.get("/v1/models").status_code == 404

    def test_api_key_protects_data_but_not_health(self, lake):
        c = TestClient(create_app(lake, api_key="s3cret"))
        assert c.get("/health").status_code == 200
        assert c.get("/v1/series").status_code == 401
        assert c.get("/v1/series", headers={"x-api-key": "wrong"}).status_code == 401
        assert c.get("/v1/series", headers={"x-api-key": "s3cret"}).status_code == 200

    def test_a_new_pipeline_run_is_picked_up_without_restarting(self, lake):
        c = TestClient(create_app(lake, api_key=""))
        before = c.get("/v1/forecast", params={"product": "GASOLINA"}).json()["points"][0]["forecast"]
        path = lake.forecasts / "latest.parquet"
        original = pd.read_parquet(path)
        try:
            changed = original.copy()
            changed["forecast"] = changed["forecast"] * 1.5
            changed.to_parquet(path)
            import os
            import time

            os.utime(path, (time.time() + 5, time.time() + 5))  # ensure a distinct mtime
            after = c.get("/v1/forecast", params={"product": "GASOLINA"}).json()["points"][0]["forecast"]
        finally:
            original.to_parquet(path)
        assert after == pytest.approx(before * 1.5, rel=1e-3)


class TestReport:
    def test_report_is_self_contained_and_reflects_the_run(self, lake):
        html = (lake.reports / "forecast_report.html").read_text(encoding="utf-8")
        metrics = load_metrics(lake)
        assert html.startswith("<!doctype html>") and metrics["champion"] in html and metrics["run_id"] in html
        assert html.count("data:image/png;base64,") >= 2 and "<script" not in html
        assert "Data quality" in html and "Lineage" in html


class TestCli:
    def test_parse_horizons(self):
        assert parse_horizons("1-4") == (1, 2, 3, 4) and parse_horizons("1,2,8") == (1, 2, 8)
        for bad in ("0-3", "", "a-b"):
            with pytest.raises(ValueError):
                parse_horizons(bad)

    def test_default_end_is_the_last_complete_month(self):
        from datetime import date

        assert default_end(date(2026, 9, 30)) == "2026-08" and default_end(date(2026, 1, 15)) == "2025-12"

    def test_status_and_sql_and_graph(self, lake, capsys):
        assert main(["status", "--root", str(lake.root)]) == 0
        out = capsys.readouterr().out
        assert "bronze:" in out and "silver:" in out and "model:" in out
        assert main(["sql", "--root", str(lake.root), "SELECT count(*) AS n FROM weekly_prices"]) == 0
        assert int(capsys.readouterr().out.split()[-1]) > 500
        assert main(["graph"]) == 0 and "quality_silver --> gold" in capsys.readouterr().out

    def test_errors_are_reported_cleanly_with_a_nonzero_exit(self, lake, tmp_path, capsys):
        assert main(["sql", "--root", str(lake.root), "SELECT nope FROM weekly_prices"]) == 1
        assert "error:" in capsys.readouterr().err
        assert main(["run", "--root", str(tmp_path / "x"), "--local", str(tmp_path / "missing")]) == 1
        assert main(["demo", "--root", str(tmp_path / "d"), "--sample", str(tmp_path / "nope")]) == 1

    def test_rerunning_selected_steps_on_an_existing_lake(self, lake, synth_dir, capsys):
        code = main(
            [
                "run",
                "--root",
                str(lake.root),
                "--local",
                str(synth_dir),
                "--only",
                "bronze,quality_bronze",
                "--as-of",
                "2025-07-01",
            ]
        )
        out = capsys.readouterr().out
        assert code == 0 and "success  bronze" in out
        assert len(list(lake.runs.glob("*.json"))) >= 2  # every run leaves a log

    def test_the_run_log_records_every_task(self, lake):
        log = json.loads(sorted(lake.runs.glob("*.json"))[0].read_text())
        assert log["ok"] and list(log["tasks"]) == [
            "bronze",
            "quality_bronze",
            "silver",
            "quality_silver",
            "gold",
            "quality_gold",
            "train",
            "forecast",
            "report",
        ]
        assert all(t["status"] == "success" for t in log["tasks"].values())
