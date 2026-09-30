from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from fuellake.bronze import Downloader, DownloadError, Manifest, fetch_bronze, import_local, verify_bronze
from fuellake.config import Settings
from fuellake.sources import last4weeks_url, monthly_candidates, semester_url
from tests.synth import HEADER

BODY = (HEADER + "\r\n").encode()
TODAY = date(2026, 9, 30)


class FakeAnp:
    """An in-memory ANP: url -> (status, bytes). Records every request made."""

    def __init__(self, routes: dict[str, tuple[int, bytes]] | None = None):
        self.routes = routes or {}
        self.requests: list[str] = []

    def client(self) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            self.requests.append(url)
            status, content = self.routes.get(url, (404, b"not found"))
            return httpx.Response(status, content=content)

        return httpx.Client(transport=httpx.MockTransport(handler))


def fetch(settings, fake, start, end, **kw):
    return fetch_bronze(
        settings,
        start,
        end,
        client=fake.client(),
        today=kw.pop("today", TODAY),
        include_last4weeks=kw.pop("include_last4weeks", False),
        sleep=lambda s: None,
        **kw,
    )


def zip_bytes(name="data.csv", content=BODY) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(tmp_path / "lake")


class TestFetch:
    def test_downloads_the_old_style_monthly_files_and_records_checksums(self, settings):
        fake = FakeAnp(
            {
                monthly_candidates("gasolina-etanol", 2024, 3)[0]: (200, BODY + b"a"),
                monthly_candidates("diesel-gnv", 2024, 3)[0]: (200, BODY + b"b"),
            }
        )
        report = fetch(settings, fake, "2024-03", "2024-03")
        assert sorted(report.downloaded) == ["monthly:2024-03:diesel-gnv", "monthly:2024-03:gasolina-etanol"]
        assert report.gaps == [] and report.bytes_downloaded == 2 * (len(BODY) + 1)
        entry = Manifest.load(settings.bronze_manifest).entries["monthly:2024-03:gasolina-etanol"]
        assert entry["sha256"] == hashlib.sha256(BODY + b"a").hexdigest()
        assert (settings.bronze / entry["path"]).read_bytes() == BODY + b"a"

    def test_falls_back_to_the_newer_url_shape(self, settings):
        fake = FakeAnp(
            {
                monthly_candidates("gasolina-etanol", 2026, 6)[1]: (200, BODY),
                monthly_candidates("diesel-gnv", 2026, 6)[1]: (200, BODY),
            }
        )
        report = fetch(settings, fake, "2026-06", "2026-06")
        assert len(report.downloaded) == 2 and not report.gaps
        assert fake.requests[0] == monthly_candidates("gasolina-etanol", 2026, 6)[0]  # tried the old shape first

    def test_uses_the_semester_archive_when_monthly_files_are_missing(self, settings):
        fake = FakeAnp({semester_url(2026, 1): (200, zip_bytes())})
        report = fetch(settings, fake, "2026-02", "2026-04")
        assert report.downloaded == ["semester:2026-H1"]  # one archive covers three months x two fuels
        assert report.gaps == []
        entry = Manifest.load(settings.bronze_manifest).entries["semester:2026-H1"]
        assert entry["covers"] == ["2026-02", "2026-03", "2026-04"]
        assert fake.requests.count(semester_url(2026, 1)) == 1

    def test_reports_a_gap_when_nothing_is_published_and_retries_it_next_time(self, settings):
        fake = FakeAnp()
        report = fetch(settings, fake, "2026-07", "2026-07")
        assert report.gaps == ["2026-07"] and report.downloaded == []
        manifest = Manifest.load(settings.bronze_manifest)
        assert {e["status"] for e in manifest.entries.values()} == {"missing"}
        before = len(fake.requests)
        fetch(settings, fake, "2026-07", "2026-07")
        assert len(fake.requests) > before  # missing months are re-checked on every run

    def test_a_published_month_is_not_downloaded_twice(self, settings):
        fake = FakeAnp(
            {
                monthly_candidates("gasolina-etanol", 2024, 3)[0]: (200, BODY),
                monthly_candidates("diesel-gnv", 2024, 3)[0]: (200, BODY),
            }
        )
        fetch(settings, fake, "2024-03", "2024-03")
        n = len(fake.requests)
        again = fetch(settings, fake, "2024-03", "2024-03")
        assert len(fake.requests) == n and len(again.cached) == 2 and again.downloaded == []

    def test_force_redownloads(self, settings):
        fake = FakeAnp({monthly_candidates(f, 2024, 3)[0]: (200, BODY) for f in ("gasolina-etanol", "diesel-gnv")})
        fetch(settings, fake, "2024-03", "2024-03")
        assert len(fetch(settings, fake, "2024-03", "2024-03", force=True).downloaded) == 2

    def test_files_still_being_updated_are_refreshed_after_a_day(self, settings):
        urls = {monthly_candidates(f, 2026, 9)[0]: (200, BODY) for f in ("gasolina-etanol", "diesel-gnv")}
        fake = FakeAnp(urls)
        fetch(settings, fake, "2026-09", "2026-09")  # current month: mutable
        assert len(fetch(settings, fake, "2026-09", "2026-09").cached) == 2  # fresh
        manifest = Manifest.load(settings.bronze_manifest)
        for entry in manifest.entries.values():
            entry["fetched_at"] = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        manifest.save()
        assert len(fetch(settings, fake, "2026-09", "2026-09").downloaded) == 2  # stale -> refreshed

    def test_last_four_weeks_are_fetched_when_requested(self, settings):
        fake = FakeAnp({last4weeks_url("gasolina-etanol"): (200, BODY), last4weeks_url("diesel-gnv"): (200, BODY)})
        report = fetch(settings, fake, "2024-03", "2024-03", include_last4weeks=True)
        assert {"last4weeks:gasolina-etanol", "last4weeks:diesel-gnv"} <= set(report.downloaded)

    def test_an_interrupted_run_resumes_where_it_stopped(self, settings):
        good = monthly_candidates("gasolina-etanol", 2024, 3)[0]
        fake = FakeAnp({good: (200, BODY), monthly_candidates("diesel-gnv", 2024, 3)[0]: (500, b"")})
        with pytest.raises(DownloadError):
            fetch(settings, fake, "2024-03", "2024-03")
        assert "monthly:2024-03:gasolina-etanol" in Manifest.load(settings.bronze_manifest).entries  # saved per file
        fake.routes[monthly_candidates("diesel-gnv", 2024, 3)[0]] = (200, BODY)
        resumed = fetch(settings, fake, "2024-03", "2024-03")
        assert resumed.downloaded == ["monthly:2024-03:diesel-gnv"] and resumed.cached == ["monthly:2024-03:gasolina-etanol"]


class TestDownloader:
    def test_retries_server_errors_then_succeeds(self, tmp_path):
        attempts = {"n": 0}
        sleeps: list[float] = []

        def handler(request):
            attempts["n"] += 1
            return httpx.Response(503) if attempts["n"] < 3 else httpx.Response(200, content=b"ok")

        d = Downloader(httpx.Client(transport=httpx.MockTransport(handler)), retries=3, sleep=sleeps.append)
        path, digest, size = d.fetch("https://x.test/a/file.csv", tmp_path)
        assert path.read_bytes() == b"ok" and size == 2 and attempts["n"] == 3 and len(sleeps) == 2

    def test_gives_up_and_leaves_no_partial_file(self, tmp_path):
        d = Downloader(
            httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))), retries=2, sleep=lambda s: None
        )
        with pytest.raises(DownloadError, match="Giving up"):
            d.fetch("https://x.test/file.csv", tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_404_means_not_published_and_is_not_retried(self, tmp_path):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(404)

        d = Downloader(httpx.Client(transport=httpx.MockTransport(handler)), retries=3, sleep=lambda s: None)
        assert d.fetch("https://x.test/file.csv", tmp_path) is None and calls["n"] == 1

    def test_empty_response_is_an_error(self, tmp_path):
        d = Downloader(httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b""))), retries=1)
        with pytest.raises(DownloadError, match="Empty"):
            d.fetch("https://x.test/file.csv", tmp_path)

    def test_network_errors_are_retried(self, tmp_path):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("boom")
            return httpx.Response(200, content=b"x")

        d = Downloader(httpx.Client(transport=httpx.MockTransport(handler)), retries=2, sleep=lambda s: None)
        assert d.fetch("https://x.test/f.csv", tmp_path) is not None


class TestIntegrity:
    def _one_file(self, settings) -> Path:
        fake = FakeAnp({monthly_candidates(f, 2024, 3)[0]: (200, BODY) for f in ("gasolina-etanol", "diesel-gnv")})
        fetch(settings, fake, "2024-03", "2024-03")
        return settings.bronze / Manifest.load(settings.bronze_manifest).entries["monthly:2024-03:diesel-gnv"]["path"]

    def test_clean_bronze_verifies(self, settings):
        self._one_file(settings)
        assert verify_bronze(settings) == []

    def test_tampering_and_deletion_are_detected(self, settings):
        path = self._one_file(settings)
        path.write_bytes(b"corrupted")
        problems = verify_bronze(settings)
        assert len(problems) == 1 and "checksum mismatch" in problems[0]
        path.unlink()
        assert "missing" in verify_bronze(settings)[0]

    def test_manifest_is_valid_json_on_disk(self, settings):
        self._one_file(settings)
        assert "entries" in json.loads(settings.bronze_manifest.read_text())


class TestImportLocal:
    def test_imports_and_is_idempotent(self, settings, tmp_path):
        src = tmp_path / "raw"
        src.mkdir()
        (src / "a.csv").write_bytes(BODY)
        (src / "b.csv.gz").write_bytes(b"\x1f\x8b fake")
        (src / "ignore.txt").write_text("x")
        first = import_local(settings, src)
        assert sorted(first.downloaded) == ["local:a.csv", "local:b.csv.gz"]
        second = import_local(settings, src)
        assert second.downloaded == [] and len(second.cached) == 2

    def test_changed_file_is_reimported(self, settings, tmp_path):
        src = tmp_path / "raw"
        src.mkdir()
        (src / "a.csv").write_bytes(BODY)
        import_local(settings, src)
        (src / "a.csv").write_bytes(BODY + b"more")
        assert import_local(settings, src).downloaded == ["local:a.csv"]

    def test_empty_folder_is_an_error(self, settings, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            import_local(settings, tmp_path / "empty")
