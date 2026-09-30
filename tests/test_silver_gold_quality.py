from __future__ import annotations

import gzip
import io
import json
import shutil
import zipfile
from datetime import UTC, date, datetime

import duckdb
import pandas as pd
import pytest

from fuellake.bronze import Manifest
from fuellake.config import Settings
from fuellake.gold import open_lake, read_weekly_prices
from fuellake.pipeline import RunParams, build_pipeline
from fuellake.quality import QualityConfig, QualityError, run_quality
from fuellake.silver import build_silver, detect_encoding, map_header, read_header
from tests.conftest import AS_OF
from tests.synth import HEADER, csv_text, make_row


def add_source(settings: Settings, source_id: str, filename: str, content: bytes) -> None:
    """Put a raw file into bronze under a chosen source id (so priority rules can be tested)."""
    settings.ensure()
    target = settings.bronze / "anp" / source_id.replace(":", "_") / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    manifest = Manifest.load(settings.bronze_manifest)
    manifest.entries[source_id] = {
        "id": source_id,
        "url": "test",
        "path": target.relative_to(settings.bronze).as_posix(),
        "sha256": "x",
        "bytes": len(content),
        "fetched_at": datetime.now(UTC).isoformat(),
        "status": "ok",
        "origin": "test",
        "covers": [],
    }
    manifest.save()


def silver_rows(settings: Settings) -> pd.DataFrame:
    return duckdb.sql(
        f"SELECT * FROM read_parquet('{settings.silver_glob}', hive_partitioning = true) ORDER BY cnpj, product, collected_on"
    ).df()


@pytest.fixture
def s(tmp_path) -> Settings:
    return Settings(tmp_path / "lake")


class TestHeader:
    def test_maps_headers_to_canonical_names_ignoring_accents_and_case(self):
        names = map_header(
            [
                "REGIÃO - SIGLA",
                "estado - sigla",
                "Município",
                "Revenda",
                "CNPJ da Revenda",
                "Nome da Rua",
                "Numero Rua",
                "Complemento",
                "Bairro",
                "Cep",
                "Produto",
                "Data da Coleta",
                "Valor de Venda",
                "Valor de Compra",
                "Unidade de Medida",
                "Bandeira",
                "Extra Column",
            ]
        )
        assert names[0] == "region" and names[2] == "city" and names[-1] == "extra_16"

    def test_a_file_that_is_not_anp_is_rejected_with_the_missing_columns(self):
        with pytest.raises(ValueError, match="Not an ANP price file.*produto"):
            map_header(["a", "b", "c"])

    def test_reads_the_header_of_plain_and_gzipped_files(self, tmp_path):
        plain, gz = tmp_path / "a.csv", tmp_path / "a.csv.gz"
        plain.write_text("﻿" + HEADER + "\r\nrow", encoding="utf-8")
        gz.write_bytes(gzip.compress(HEADER.encode() + b"\r\nrow"))
        assert read_header(plain, "utf-8")[0] == "Regiao - Sigla"
        assert read_header(gz, "utf-8")[-1] == "Bandeira"

    def test_encoding_detection_handles_utf8_latin1_and_gzip(self, tmp_path):
        (tmp_path / "u.csv").write_bytes("AÇAÍ".encode())
        (tmp_path / "l.csv").write_bytes("AÇAÍ".encode("latin-1"))
        (tmp_path / "g.csv.gz").write_bytes(gzip.compress("AÇAÍ".encode()))
        assert detect_encoding(tmp_path / "u.csv") == "utf-8"
        assert detect_encoding(tmp_path / "l.csv") == "latin-1"
        assert detect_encoding(tmp_path / "g.csv.gz") == "utf-8"


class TestCleaning:
    def build(self, s, rows, source_id="monthly:2024-03:gasolina-etanol", name="a.csv", raw: bytes | None = None):
        add_source(s, source_id, name, raw if raw is not None else csv_text(rows).encode("utf-8"))
        return build_silver(s)

    def test_parses_prices_dates_and_cnpj_in_every_real_world_format(self, s):
        rows = [
            make_row(**{"Valor de Venda": "6,5", "CNPJ da Revenda": " 11.222.333/0001-44", "Data da Coleta": "05/03/2024"}),
            make_row(
                **{
                    "Valor de Venda": "4,199",
                    "Produto": "ETANOL",
                    "CNPJ da Revenda": "1222333000144",
                    "Data da Coleta": "2024-03-06",
                }
            ),
            make_row(**{"Valor de Venda": "5.99", "Produto": "DIESEL S10", "Data da Coleta": "07/03/2024"}),
        ]
        m = self.build(s, rows)
        df = silver_rows(s).set_index("product")
        assert m["rows_silver"] == 3 and m["rows_rejected"] == 0
        assert df.loc["GASOLINA", "price"] == 6.5 and df.loc["ETANOL", "price"] == 4.199 and df.loc["DIESEL S10", "price"] == 5.99
        assert set(df["cnpj"]) == {"11222333000144", "01222333000144"}  # 13 digits were left-padded
        assert df.loc["ETANOL", "collected_on"] == pd.Timestamp("2024-03-06")

    @pytest.mark.parametrize(
        "field,value,reason",
        [
            ("CNPJ da Revenda", "", "missing_cnpj"),
            ("CNPJ da Revenda", "123", "invalid_cnpj"),
            ("Data da Coleta", "31/02/2024", "invalid_date"),
            ("Data da Coleta", "lixo", "invalid_date"),
            ("Data da Coleta", "01/01/1990", "invalid_date"),
            ("Data da Coleta", "01/01/2099", "future_date"),
            ("Valor de Venda", "abc", "invalid_price"),
            ("Valor de Venda", "", "invalid_price"),
            ("Valor de Venda", "0,00", "price_out_of_range"),
            ("Valor de Venda", "9999,00", "price_out_of_range"),
            ("Valor de Venda", "1.234,56", "price_out_of_range"),  # thousands separator parsed, then rejected as implausible
            ("Estado - Sigla", "XX", "missing_field"),
            ("Bandeira", "", "missing_field"),
            ("Produto", "", "missing_field"),
        ],
    )
    def test_each_rejection_reason_is_counted_and_the_good_row_survives(self, s, field, value, reason):
        m = self.build(
            s,
            [
                make_row(),
                make_row(
                    **{field: value, "CNPJ da Revenda": " 99.888.777/0001-66"} if field != "CNPJ da Revenda" else {field: value}
                ),
            ],
        )
        assert m["rows_silver"] == 1 and m["rejects_by_reason"] == {reason: 1}
        assert len(pd.read_parquet(s.silver_rejects)) == 1

    def test_duplicates_collapse_and_the_authoritative_source_wins(self, s):
        add_source(s, "last4weeks:gasolina-etanol", "l.csv", csv_text([make_row(**{"Valor de Venda": "6,00"})]).encode())
        add_source(s, "monthly:2024-03:gasolina-etanol", "m.csv", csv_text([make_row(**{"Valor de Venda": "6,50"})]).encode())
        add_source(s, "semester:2024-H1", "s.csv", csv_text([make_row(**{"Valor de Venda": "7,00"})]).encode())
        m = build_silver(s)
        df = silver_rows(s)
        assert len(df) == 1 and df["price"].iloc[0] == 6.5 and df["source_id"].iloc[0].startswith("monthly")
        assert m["duplicates_removed"] == 2

    def test_a_latin1_file_is_decoded_correctly(self, s):
        raw = csv_text([make_row(Revenda="POSTO AÇAÍ LTDA")]).replace("﻿", "").encode("latin-1")
        self.build(s, [], raw=raw)
        assert silver_rows(s)["station_name"].iloc[0] == "POSTO AÇAÍ LTDA"

    def test_reads_csv_inside_a_zip_and_gz_files(self, s):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("Precos semestrais.csv", csv_text([make_row(**{"CNPJ da Revenda": " 11.111.111/0001-11"})]))
        add_source(s, "semester:2024-H1", "ca.zip", buffer.getvalue())
        add_source(
            s,
            "monthly:2024-03:diesel-gnv",
            "d.csv.gz",
            gzip.compress(csv_text([make_row(**{"CNPJ da Revenda": " 22.222.222/0001-22"})]).encode()),
        )
        assert build_silver(s)["rows_silver"] == 2

    def test_zip_without_a_csv_is_an_error(self, s):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("readme.txt", "no data here")
        add_source(s, "semester:2024-H1", "ca.zip", buffer.getvalue())
        with pytest.raises(ValueError, match="no CSV"):
            build_silver(s)

    def test_malformed_lines_are_counted_and_do_not_stop_the_load(self, s):
        good = csv_text([make_row(), make_row(**{"CNPJ da Revenda": " 55.555.555/0001-55"})]).splitlines()
        text = "\r\n".join([good[0], good[1], good[1] + ";;;;;extra;columns", good[2]]) + "\r\n"
        m = self.build(s, [], raw=text.encode())
        assert m["rows_silver"] == 2 and m["rows_malformed"] >= 1

    def test_silver_is_written_as_hive_partitions(self, s):
        self.build(s, [make_row(**{"Data da Coleta": "05/03/2024"}), make_row(**{"Data da Coleta": "05/04/2024"})])
        assert sorted(p.name for p in (s.silver_prices / "year=2024").iterdir()) == ["month=3", "month=4"]

    def test_rebuilding_is_idempotent(self, s):
        self.build(s, [make_row(), make_row(**{"Produto": "ETANOL"})])
        first = silver_rows(s)
        build_silver(s)
        pd.testing.assert_frame_equal(first, silver_rows(s))

    def test_a_failed_rebuild_leaves_the_previous_silver_intact(self, s):
        self.build(s, [make_row()])
        add_source(s, "monthly:2024-04:diesel-gnv", "bad.csv", b"not;an;anp;file\r\n1;2;3;4\r\n")
        with pytest.raises(ValueError):
            build_silver(s)
        assert len(silver_rows(s)) == 1

    def test_empty_bronze_is_an_error(self, s):
        with pytest.raises(RuntimeError, match="Bronze is empty"):
            build_silver(s)


class TestGoldAndQualityOnSyntheticLake:
    def test_weekly_prices_match_an_independent_pandas_calculation(self, lake):
        silver = silver_rows(lake)
        silver["week_start"] = silver["collected_on"].dt.to_period("W-SUN").dt.start_time
        expected = (
            silver.groupby(["week_start", "state", "product"])
            .agg(avg=("price", "mean"), n=("price", "size"), st=("cnpj", "nunique"))
            .reset_index()
        )
        gold = read_weekly_prices(lake)
        merged = expected.merge(gold, on=["week_start", "state", "product"])
        assert len(merged) == len(expected) > 500
        assert merged["avg"].sub(merged["avg_price"]).abs().max() < 1e-9
        assert (merged["n"] == merged["observations"]).all() and (merged["st"] == merged["stations"]).all()

    def test_national_rows_aggregate_every_state(self, lake):
        silver = silver_rows(lake)
        gold = read_weekly_prices(lake)
        week = gold["week_start"].iloc[len(gold) // 2]
        in_week = silver[
            (silver["collected_on"] >= week)
            & (silver["collected_on"] < week + pd.Timedelta(days=7))
            & (silver["product"] == "GASOLINA")
        ]
        national = gold[(gold["state"] == "BR") & (gold["week_start"] == week) & (gold["product"] == "GASOLINA")]
        assert national["avg_price"].iloc[0] == pytest.approx(in_week["price"].mean())
        assert national["observations"].iloc[0] == len(in_week)

    def test_station_dimension_has_one_row_per_station(self, lake):
        stations = pd.read_parquet(lake.gold / "stations.parquet")
        assert stations["cnpj"].is_unique and len(stations) == 6 * 28

    def test_ad_hoc_sql_over_the_lake(self, lake):
        con = open_lake(lake)
        assert con.execute("SELECT count(DISTINCT state) FROM weekly_prices WHERE state <> 'BR'").fetchone()[0] == 6
        assert con.execute("SELECT count(*) FROM silver_prices").fetchone()[0] > 10_000
        assert con.execute("SELECT count(*) FROM forecasts").fetchone()[0] > 0

    def test_the_synthetic_lake_passes_every_quality_gate(self, lake):
        report = run_quality(lake, today=AS_OF, write=False)
        assert not report.errors, [c.detail for c in report.errors]
        assert report.summary()["checks"] >= 15

    def test_the_quality_report_is_persisted_and_merges_layers(self, lake):
        stored = json.loads((lake.reports / "quality_report.json").read_text())
        assert {c["layer"] for c in stored["checks"]} == {"bronze", "silver", "gold"}


class TestQualityFailures:
    def test_tampering_with_bronze_is_an_error(self, lake, tmp_path):
        copy = tmp_path / "copy"
        shutil.copytree(lake.root, copy)
        settings = Settings(copy)
        victim = next(p for p in (settings.bronze / "anp").rglob("*.csv") if p.is_file())
        victim.write_bytes(victim.read_bytes() + b"tampered")
        report = run_quality(settings, layers=("bronze",), write=False)
        assert [c.name for c in report.errors] == ["bronze_checksums"]
        with pytest.raises(QualityError, match="checksum"):
            report.enforce()

    def test_stale_data_warns_then_errors(self, lake):
        assert {c.name for c in run_quality(lake, ("gold",), today=date(2025, 8, 1), write=False).warnings} >= {
            "gold_freshness_warn"
        }
        report = run_quality(lake, ("gold",), today=date(2026, 6, 1), write=False)
        assert "gold_freshness_error" in {c.name for c in report.errors}

    def test_a_missing_month_is_a_warning_not_a_failure_and_is_bridged_for_modelling(self, synth_dir, tmp_path):
        raw = tmp_path / "raw"
        shutil.copytree(synth_dir, raw)
        for f in list(raw.glob("2024-03_*")):
            f.unlink()
        settings = Settings(tmp_path / "lake")
        params = RunParams(local_dir=raw, as_of=AS_OF)
        pipe = build_pipeline(settings, params, log=lambda m: None)
        run = pipe.run(only=["bronze", "quality_bronze", "silver", "quality_silver", "gold", "quality_gold"])
        assert run.ok  # warnings never stop the pipeline
        report = json.loads((settings.reports / "quality_report.json").read_text())
        coverage = next(c for c in report["checks"] if c["name"] == "gold_weekly_coverage")
        assert not coverage["passed"] and coverage["value"]["missing_weeks"] >= 3

        from fuellake.ml.dataset import build_panel

        panel = build_panel(read_weekly_prices(settings))
        assert panel["imputed"].sum() > 0 and panel["price"].notna().all()

    def test_a_high_reject_rate_stops_the_pipeline_before_gold(self, tmp_path):
        rows = [make_row(**{"CNPJ da Revenda": f" 00.000.000/{i:04d}-00"}) for i in range(10)]
        rows += [make_row(**{"Valor de Venda": "abc", "CNPJ da Revenda": f" 11.000.000/{i:04d}-00"}) for i in range(10)]
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "bad.csv").write_text(csv_text(rows), encoding="utf-8", newline="")
        settings = Settings(tmp_path / "lake")
        run = build_pipeline(settings, RunParams(local_dir=raw, as_of=date(2024, 4, 1)), log=lambda m: None).run(
            only=["bronze", "quality_bronze", "silver", "quality_silver", "gold"]
        )
        assert run.results["quality_silver"].status == "failed" and "reject" in run.results["quality_silver"].error
        assert run.results["gold"].status == "skipped"

    def test_price_jump_detection_ignores_gaps(self, lake, tmp_path):
        settings = Settings(tmp_path / "g")
        (settings.gold).mkdir(parents=True)
        weeks = pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15", "2024-03-04"])  # a 6-week hole before the last one
        df = pd.DataFrame(
            {
                "week_start": weeks,
                "state": "BR",
                "product": "GASOLINA",
                "avg_price": [5.0, 5.0, 8.0, 20.0],
                "stations": 100,
                "days_covered": 5,
                "observations": 100,
                "median_price": 5.0,
                "min_price": 5.0,
                "max_price": 9.0,
                "std_price": 0.1,
            }
        )
        df.to_parquet(settings.gold / "weekly_prices.parquet")
        from fuellake.quality import QualityReport, check_gold

        report = QualityReport("now")
        check_gold(settings, report, today=date(2024, 3, 10), config=QualityConfig())
        jumps = next(c for c in report.checks if c.name == "gold_weekly_price_jumps")
        assert (
            not jumps.passed and len(jumps.value) == 1 and jumps.value[0]["week"] == "2024-01-15"
        )  # the +60%, not the +150% across a hole
        gaps = next(c for c in report.checks if c.name == "gold_weekly_coverage")
        assert gaps.value["missing_weeks"] == 6
