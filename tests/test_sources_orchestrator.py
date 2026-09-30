from __future__ import annotations

import json

import pytest

from fuellake.orchestrator import CycleError, Pipeline, Task
from fuellake.sources import half_of, last4weeks_url, month_range, monthly_candidates, parse_month, semester_url


class TestSources:
    def test_month_range_is_inclusive_and_crosses_years(self):
        assert month_range("2023-11", "2024-02") == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]
        assert month_range("2024-05", "2024-05") == [(2024, 5)]

    @pytest.mark.parametrize("bad", ["2024", "2024-13", "2024-00", "abcd-01", "2024-1-1", ""])
    def test_parse_month_rejects_garbage(self, bad):
        with pytest.raises(ValueError):
            parse_month(bad)

    def test_start_after_end_is_an_error(self):
        with pytest.raises(ValueError, match="after"):
            month_range("2024-05", "2024-04")

    def test_monthly_candidates_cover_both_known_url_shapes(self):
        old, new = monthly_candidates("gasolina-etanol", 2025, 3)
        assert old.endswith("/dsan/2025/precos-gasolina-etanol-03.csv")
        assert new.endswith("/dsan/2025/03-dados-abertos-precos-2025-03-gasolina-etanol.csv")

    def test_semester_and_rolling_urls(self):
        assert semester_url(2026, 1).endswith("/dsas/ca/ca-2026-01.zip")
        assert last4weeks_url("diesel-gnv").endswith("/qus/ultimas-4-semanas-diesel-gnv.csv")

    @pytest.mark.parametrize("month,half", [(1, 1), (6, 1), (7, 2), (12, 2)])
    def test_half_of(self, month, half):
        assert half_of(month) == half


class TestOrchestrator:
    @staticmethod
    def recorder():
        calls: list[str] = []

        def make(name, fail_times=0):
            state = {"n": 0}

            def fn():
                calls.append(name)
                state["n"] += 1
                if state["n"] <= fail_times:
                    raise RuntimeError(f"{name} boom {state['n']}")
                return {"task": name}

            return fn

        return calls, make

    def test_runs_in_dependency_order_regardless_of_declaration_order(self):
        calls, make = self.recorder()
        pipeline = Pipeline([Task("c", make("c"), deps=("b",)), Task("a", make("a")), Task("b", make("b"), deps=("a",))])
        run = pipeline.run()
        assert calls == ["a", "b", "c"]
        assert run.ok and run.results["b"].output == {"task": "b"}

    def test_failure_skips_everything_downstream_but_not_independent_branches(self):
        calls, make = self.recorder()
        pipeline = Pipeline(
            [
                Task("a", make("a")),
                Task("b", make("b", fail_times=99), deps=("a",)),
                Task("c", make("c"), deps=("b",)),
                Task("d", make("d"), deps=("a",)),
            ]
        )
        run = pipeline.run()
        assert not run.ok and run.failed == ["b"]
        assert run.results["c"].status == "skipped" and "b" in run.results["c"].error
        assert run.results["d"].status == "success"
        assert "c" not in calls

    def test_retries_then_succeeds_with_backoff(self):
        calls, make = self.recorder()
        sleeps: list[float] = []
        pipeline = Pipeline([Task("flaky", make("flaky", fail_times=2), retries=3, retry_delay=5)])
        run = pipeline.run(sleep=sleeps.append)
        assert run.ok and run.results["flaky"].attempts == 3
        assert sleeps == [5, 10]  # linear backoff

    def test_gives_up_after_retries_and_records_the_error(self):
        _, make = self.recorder()
        run = Pipeline([Task("bad", make("bad", fail_times=99), retries=1)]).run(sleep=lambda s: None)
        assert run.results["bad"].status == "failed" and run.results["bad"].attempts == 2
        assert "RuntimeError: bad boom" in run.results["bad"].error

    def test_cycle_and_unknown_dependency_are_rejected(self):
        with pytest.raises(CycleError, match="cycle"):
            Pipeline([Task("a", lambda: 1, deps=("b",)), Task("b", lambda: 1, deps=("a",))])
        with pytest.raises(ValueError, match="unknown task"):
            Pipeline([Task("a", lambda: 1, deps=("ghost",))])
        with pytest.raises(ValueError, match="Duplicate"):
            Pipeline([Task("a", lambda: 1), Task("a", lambda: 2)])

    def test_only_and_start_from_selection(self):
        _, make = self.recorder()
        p = Pipeline(
            [Task("a", make("a")), Task("b", make("b"), deps=("a",)), Task("c", make("c"), deps=("b",)), Task("x", make("x"))]
        )
        assert p.select(only=["c", "a"]) == ["a", "c"]
        assert p.select(start_from="b") == ["b", "c"]
        order = p.select()
        assert sorted(order) == ["a", "b", "c", "x"]
        assert order.index("a") < order.index("b") < order.index("c")  # any valid topological order
        with pytest.raises(ValueError):
            p.select(only=["nope"])
        with pytest.raises(ValueError):
            p.select(start_from="nope")

    def test_run_log_is_written_as_json_and_survives_unserialisable_output(self, tmp_path):
        p = Pipeline([Task("a", lambda: object())])
        run = p.run(run_dir=tmp_path)
        written = json.loads((tmp_path / f"{run.run_id}.json").read_text())
        assert written["ok"] and written["tasks"]["a"]["status"] == "success"
        assert isinstance(written["tasks"]["a"]["output"], str)

    def test_mermaid_diagram_lists_edges(self):
        p = Pipeline([Task("a", lambda: 1), Task("b", lambda: 1, deps=("a",))])
        assert "a --> b" in p.to_mermaid()
