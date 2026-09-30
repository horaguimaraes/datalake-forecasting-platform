"""Bronze layer: raw files, byte for byte, with a manifest.

Nothing here interprets the data. The point of bronze is that every later layer can be rebuilt
from it, so it stores exactly what the source published, plus enough metadata to prove it
(URL, SHA-256, size, when it was fetched) and to know what is missing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import Settings
from .sources import FUELS, half_of, last4weeks_url, month_range, monthly_candidates, semester_url

USER_AGENT = "fuellake/1.0 (open-data research project)"
MAX_AGE_HOURS = 24  # how long a still-changing file (current month, last 4 weeks) is trusted


class DownloadError(RuntimeError):
    """A source failed in a way that is not 'file does not exist' (which is normal and handled)."""


# --------------------------------------------------------------------------- manifest


class Manifest:
    """JSON catalog of every file in bronze. Written atomically, so a crash can't corrupt it."""

    def __init__(self, path: Path, entries: dict[str, dict[str, Any]] | None = None):
        self.path = path
        self.entries = entries or {}

    @classmethod
    def load(cls, path: Path) -> Manifest:
        if path.exists():
            return cls(path, json.loads(path.read_text(encoding="utf-8")).get("entries", {}))
        return cls(path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"entries": self.entries}, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def ok_entries(self) -> list[dict[str, Any]]:
        return [e for _, e in sorted(self.entries.items()) if e.get("status") == "ok"]

    def missing_entries(self) -> list[dict[str, Any]]:
        return [e for _, e in sorted(self.entries.items()) if e.get("status") == "missing"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(source_id: str) -> str:
    return source_id.replace(":", "_").replace("/", "_")


# --------------------------------------------------------------------------- downloading


class Downloader:
    """Streams a URL to disk with retries. A 404 means 'not published', anything else odd is an error."""

    def __init__(self, client: httpx.Client, retries: int = 3, sleep: Callable[[float], None] = time.sleep):
        self.client, self.retries, self.sleep = client, retries, sleep

    def fetch(self, url: str, target_dir: Path) -> tuple[Path, str, int] | None:
        """Download `url` into `target_dir`. Returns (path, sha256, bytes), or None if it does not exist."""
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / (Path(urlparse(url).path).name or "download")
        part = target.with_name(target.name + ".part")
        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                with self.client.stream("GET", url) as response:
                    if response.status_code == 404:
                        return None
                    if response.status_code >= 500 or response.status_code == 429:
                        raise DownloadError(f"HTTP {response.status_code} from {url}")
                    if response.status_code != 200:
                        raise DownloadError(f"Unexpected HTTP {response.status_code} from {url}")
                    digest, size = hashlib.sha256(), 0
                    with part.open("wb") as handle:
                        for chunk in response.iter_bytes(1 << 20):
                            handle.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                if size == 0:
                    raise DownloadError(f"Empty response from {url}")
                part.replace(target)
                return target, digest.hexdigest(), size
            except (httpx.TransportError, DownloadError) as error:
                last_error = error
                part.unlink(missing_ok=True)
                if attempt < self.retries:
                    self.sleep(2**attempt)
        raise DownloadError(f"Giving up on {url} after {self.retries} attempts: {last_error}")


def make_client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=20.0), headers={"User-Agent": USER_AGENT})


# --------------------------------------------------------------------------- planning + fetching


@dataclass
class FetchReport:
    downloaded: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)  # months with no usable source
    bytes_downloaded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"n_downloaded": len(self.downloaded), "n_cached": len(self.cached), "n_gaps": len(self.gaps)}


def _is_stale(entry: dict[str, Any] | None, mutable: bool, now: datetime, force: bool) -> bool:
    """A file needs (re)fetching when absent, forced, missing on disk, or mutable and older than a day."""
    if entry is None or entry.get("status") != "ok" or force:
        return True
    if not mutable:
        return False
    age = now - datetime.fromisoformat(entry["fetched_at"])
    return age.total_seconds() > MAX_AGE_HOURS * 3600


def fetch_bronze(
    settings: Settings,
    start: str,
    end: str,
    client: httpx.Client | None = None,
    today: date | None = None,
    include_last4weeks: bool = True,
    force: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda message: None,
) -> FetchReport:
    """Make sure bronze holds every ANP file needed for [start, end]. Idempotent and resumable."""
    settings.ensure()
    today = today or datetime.now(UTC).date()
    now = datetime.now(UTC)
    own_client = client is None
    client = client or make_client()
    downloader = Downloader(client, sleep=sleep)
    manifest = Manifest.load(settings.bronze_manifest)
    report = FetchReport()
    months = month_range(start, end)
    semesters: dict[tuple[int, int], dict[str, Any] | None] = {}

    def store(source_id: str, url: str, fetched: tuple[Path, str, int], covers: list[str]) -> dict[str, Any]:
        path, digest, size = fetched
        entry = {
            "id": source_id,
            "url": url,
            "path": path.relative_to(settings.bronze).as_posix(),
            "sha256": digest,
            "bytes": size,
            "fetched_at": datetime.now(UTC).isoformat(),
            "status": "ok",
            "origin": "download",
            "covers": covers,
        }
        manifest.entries[source_id] = entry
        manifest.save()  # after every file, so an interrupted run resumes where it stopped
        report.downloaded.append(source_id)
        report.bytes_downloaded += size
        log(f"  downloaded {source_id} ({size / 1e6:.1f} MB)")
        return entry

    def get_semester(year: int, half: int) -> dict[str, Any] | None:
        key = (year, half)
        if key in semesters:
            return semesters[key]
        source_id = f"semester:{year}-H{half}"
        in_range = [f"{y}-{m:02d}" for y, m in months if y == year and half_of(m) == half]
        mutable = (year, half) == (today.year, half_of(today.month))
        entry = manifest.entries.get(source_id)
        if not _is_stale(entry, mutable, now, force):
            report.cached.append(source_id)
            semesters[key] = entry
            return entry
        url = semester_url(year, half)
        fetched = downloader.fetch(url, settings.bronze / "anp" / _safe(source_id))
        semesters[key] = store(source_id, url, fetched, in_range) if fetched else None
        return semesters[key]

    try:
        for year, month in months:
            ym = f"{year}-{month:02d}"
            for fuel in FUELS:
                source_id = f"monthly:{ym}:{fuel}"
                mutable = (year, month) >= (today.year, today.month)
                if not _is_stale(manifest.entries.get(source_id), mutable, now, force):
                    report.cached.append(source_id)
                    continue

                fetched, used_url = None, ""
                for url in monthly_candidates(fuel, year, month):
                    fetched = downloader.fetch(url, settings.bronze / "anp" / _safe(source_id))
                    if fetched:
                        used_url = url
                        break
                if fetched:
                    store(source_id, used_url, fetched, [ym])
                    continue

                # Not published as a monthly file: the semester archive may still contain it.
                semester = get_semester(year, half_of(month))
                if semester is not None:
                    manifest.entries.pop(source_id, None)
                    continue
                manifest.entries[source_id] = {
                    "id": source_id,
                    "status": "missing",
                    "covers": [ym],
                    "checked_at": datetime.now(UTC).isoformat(),
                    "note": "no monthly file and no semester archive published",
                }
                manifest.save()
                if ym not in report.gaps:
                    report.gaps.append(ym)
                    log(f"  GAP {ym}: not published by ANP")

        if include_last4weeks:
            for fuel in FUELS:
                source_id = f"last4weeks:{fuel}"
                if not _is_stale(manifest.entries.get(source_id), True, now, force):
                    report.cached.append(source_id)
                    continue
                url = last4weeks_url(fuel)
                fetched = downloader.fetch(url, settings.bronze / "anp" / _safe(source_id))
                if fetched:
                    store(source_id, url, fetched, [])

        # A month covered by a semester archive is not a gap, even when its monthly files were missing.
        covered = {m for e in manifest.ok_entries() for m in e.get("covers", [])}
        report.gaps = sorted(m for m in report.gaps if m not in covered)
        manifest.save()
    finally:
        if own_client:
            client.close()
    return report


def import_local(settings: Settings, directory: Path, log: Callable[[str], None] = lambda message: None) -> FetchReport:
    """Register files from a local folder as bronze (offline mode). Idempotent by SHA-256."""
    settings.ensure()
    manifest = Manifest.load(settings.bronze_manifest)
    report = FetchReport()
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".gz", ".zip"})
    if not files:
        raise FileNotFoundError(f"No .csv, .csv.gz or .zip files found in {directory}")
    for path in files:
        source_id = f"local:{path.name}"
        digest = sha256_file(path)
        existing = manifest.entries.get(source_id)
        if existing and existing.get("sha256") == digest and (settings.bronze / existing["path"]).exists():
            report.cached.append(source_id)
            continue
        target = settings.bronze / "anp" / _safe(source_id) / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest.entries[source_id] = {
            "id": source_id,
            "url": path.as_posix(),
            "path": target.relative_to(settings.bronze).as_posix(),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "fetched_at": datetime.now(UTC).isoformat(),
            "status": "ok",
            "origin": "local",
            "covers": [],
        }
        report.downloaded.append(source_id)
        report.bytes_downloaded += path.stat().st_size
        log(f"  imported {path.name}")
    manifest.save()
    return report


def verify_bronze(settings: Settings) -> list[str]:
    """Integrity problems: files that vanished or whose bytes no longer match the recorded SHA-256."""
    problems = []
    for entry in Manifest.load(settings.bronze_manifest).ok_entries():
        path = settings.bronze / entry["path"]
        if not path.exists():
            problems.append(f"{entry['id']}: file is missing ({entry['path']})")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"{entry['id']}: checksum mismatch, file was modified after download")
    return problems
