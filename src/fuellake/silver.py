"""Silver layer: cleaned, typed, deduplicated observations as partitioned Parquet.

Silver is a *pure function of bronze*: it is rebuilt from scratch on every run, so it can never
drift from the raw files, and fixing a cleaning rule is just re-running this step.
"""

from __future__ import annotations

import codecs
import json
import re
import shutil
import tempfile
import unicodedata
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb

from .bronze import Manifest
from .config import Settings

VALID_STATES = (
    "AC",
    "AL",
    "AM",
    "AP",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MG",
    "MS",
    "MT",
    "PA",
    "PB",
    "PE",
    "PI",
    "PR",
    "RJ",
    "RN",
    "RO",
    "RR",
    "RS",
    "SC",
    "SE",
    "SP",
    "TO",
)
MIN_PRICE, MAX_PRICE = 0.1, 500.0  # wide on purpose: LPG cylinders cost ~R$100, fuels ~R$3-10
EARLIEST_DATE = "2000-01-01"

# canonical column name -> folded header text used by ANP
HEADER_ALIASES = {
    "region": "regiao sigla",
    "state": "estado sigla",
    "city": "municipio",
    "station_name": "revenda",
    "cnpj_raw": "cnpj da revenda",
    "street": "nome da rua",
    "street_number": "numero rua",
    "complement": "complemento",
    "neighborhood": "bairro",
    "zip": "cep",
    "product": "produto",
    "date_raw": "data da coleta",
    "price_raw": "valor de venda",
    "purchase_raw": "valor de compra",
    "unit": "unidade de medida",
    "brand": "bandeira",
}
_BY_ALIAS = {alias: name for name, alias in HEADER_ALIASES.items()}

SOURCE_PRIORITY = {"monthly": 3, "semester": 2, "last4weeks": 1, "local": 0}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def detect_encoding(path: Path) -> str:
    """'utf-8' when the whole file decodes cleanly, otherwise 'latin-1' (older ANP archives)."""
    import gzip

    decoder = codecs.getincrementaldecoder("utf-8")()
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rb") as handle:  # type: ignore[operator]
            for chunk in iter(lambda: handle.read(4 << 20), b""):
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def read_header(path: Path, encoding: str) -> list[str]:
    import gzip

    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding=encoding, newline="") as handle:  # type: ignore[operator]
        line = handle.readline().lstrip("﻿").strip("\r\n")
    return [cell.strip().strip('"') for cell in line.split(";")]


def map_header(header: list[str]) -> list[str]:
    """Canonical column names in file order. Unknown columns become extra_N; missing ones are an error."""
    names = [_BY_ALIAS.get(_fold(cell), f"extra_{i}") for i, cell in enumerate(header)]
    missing = [alias for name, alias in HEADER_ALIASES.items() if name not in names]
    if missing:
        raise ValueError(f"Not an ANP price file. Missing column(s): {', '.join(missing)}")
    return names


def source_kind(source_id: str) -> str:
    return source_id.split(":", 1)[0]


def _csv_files(entry: dict[str, Any], bronze: Path, scratch: Path) -> list[Path]:
    """The CSV(s) behind a manifest entry. ZIP archives are unpacked into `scratch`."""
    path = bronze / entry["path"]
    if path.suffix.lower() != ".zip":
        return [path]
    out = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.filename.lower().endswith(".csv") and not info.is_dir():
                target = scratch / f"{entry['id'].replace(':', '_')}__{Path(info.filename).name}"
                with archive.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                out.append(target)
    if not out:
        raise ValueError(f"{path.name} contains no CSV file")
    return out


_STAGING_COLUMNS = [*HEADER_ALIASES.keys()]


def _clean_sql(states: str) -> str:
    return f"""
    CREATE OR REPLACE TEMP TABLE parsed AS
    WITH base AS (
      SELECT
        source_id, priority,
        regexp_replace(coalesce(cnpj_raw, ''), '[^0-9]', '', 'g') AS cnpj_digits,
        upper(trim(coalesce(product, '')))  AS product,
        upper(trim(coalesce(state, '')))    AS state,
        upper(trim(coalesce(city, '')))     AS city,
        upper(trim(coalesce(brand, '')))    AS brand,
        upper(trim(coalesce(region, '')))   AS region,
        trim(regexp_replace(coalesce(station_name, ''), '\\s+', ' ', 'g')) AS station_name,
        trim(coalesce(neighborhood, '')) AS neighborhood,
        trim(coalesce(street, ''))       AS street,
        trim(coalesce(street_number, '')) AS street_number,
        trim(coalesce(complement, ''))   AS complement,
        trim(coalesce(zip, ''))          AS zip,
        trim(coalesce(unit, ''))         AS unit,
        CAST(COALESCE(TRY_STRPTIME(trim(date_raw), '%d/%m/%Y'), TRY_STRPTIME(trim(date_raw), '%Y-%m-%d')) AS DATE) AS collected_on,
        TRY_CAST(CASE WHEN trim(price_raw) LIKE '%,%'
                      THEN replace(replace(trim(price_raw), '.', ''), ',', '.') ELSE trim(price_raw) END AS DOUBLE) AS price,
        TRY_CAST(CASE WHEN trim(purchase_raw) LIKE '%,%'
                      THEN replace(replace(trim(purchase_raw), '.', ''), ',', '.') ELSE trim(purchase_raw) END AS DOUBLE) AS purchase_price
      FROM staging
    ), typed AS (
      SELECT *, CASE WHEN length(cnpj_digits) = 13 THEN lpad(cnpj_digits, 14, '0') ELSE cnpj_digits END AS cnpj
      FROM base
    )
    SELECT *,
      CASE
        WHEN cnpj_digits = ''                                        THEN 'missing_cnpj'
        WHEN length(cnpj) <> 14                                      THEN 'invalid_cnpj'
        WHEN collected_on IS NULL OR collected_on < DATE '{EARLIEST_DATE}' THEN 'invalid_date'
        WHEN collected_on > current_date + INTERVAL 1 DAY            THEN 'future_date'
        WHEN price IS NULL                                           THEN 'invalid_price'
        WHEN price < {MIN_PRICE} OR price > {MAX_PRICE}              THEN 'price_out_of_range'
        WHEN product = '' OR station_name = '' OR city = '' OR brand = '' OR unit = ''
             OR state NOT IN ({states})                              THEN 'missing_field'
      END AS reject_reason
    FROM typed
    """


def build_silver(settings: Settings, log: Callable[[str], None] = lambda message: None) -> dict[str, Any]:
    """Rebuild silver from every OK entry in the bronze manifest. Returns a metrics dict."""
    settings.ensure()
    entries = Manifest.load(settings.bronze_manifest).ok_entries()
    if not entries:
        raise RuntimeError("Bronze is empty: nothing to build silver from. Run the bronze step first.")

    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TEMP TABLE staging (source_id VARCHAR, priority INTEGER, "
        + ", ".join(f"{c} VARCHAR" for c in _STAGING_COLUMNS)
        + ")"
    )

    sources: dict[str, dict[str, int]] = {}
    with tempfile.TemporaryDirectory(prefix="fuellake_") as scratch_dir:
        scratch = Path(scratch_dir)
        for entry in entries:
            priority = SOURCE_PRIORITY.get(source_kind(entry["id"]), 0)
            read = malformed = 0
            for csv_path in _csv_files(entry, settings.bronze, scratch):
                encoding = detect_encoding(csv_path)
                names = map_header(read_header(csv_path, encoding))
                names_sql = "[" + ", ".join(f"'{n}'" for n in names) + "]"
                before = con.execute("SELECT count(*) FROM staging").fetchone()[0]
                con.execute("DROP TABLE IF EXISTS reject_errors")
                con.execute("DROP TABLE IF EXISTS reject_scans")
                cols = ", ".join(_STAGING_COLUMNS)
                con.execute(
                    f"""INSERT INTO staging
                        SELECT ?, ?, {cols} FROM read_csv(?, delim=';', header=true, names={names_sql},
                            all_varchar=true, quote='"', escape='"', encoding='{encoding}',
                            ignore_errors=true, store_rejects=true, null_padding=false)""",
                    [entry["id"], priority, csv_path.as_posix()],
                )
                after = con.execute("SELECT count(*) FROM staging").fetchone()[0]
                read += after - before
                if con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'reject_errors'").fetchone()[0]:
                    malformed += con.execute("SELECT count(DISTINCT (scan_id, line)) FROM reject_errors").fetchone()[0]
            sources[entry["id"]] = {"rows_read": read, "rows_malformed": malformed}
            log(f"  {entry['id']}: {read:,} rows" + (f", {malformed:,} malformed" if malformed else ""))

    states = ", ".join(f"'{s}'" for s in VALID_STATES)
    con.execute(_clean_sql(states))
    total_parsed = con.execute("SELECT count(*) FROM parsed").fetchone()[0]

    rejects = dict(
        con.execute("SELECT reject_reason, count(*) FROM parsed WHERE reject_reason IS NOT NULL GROUP BY 1").fetchall()
    )
    con.execute(
        """CREATE TEMP TABLE deduped AS
           SELECT * EXCLUDE (reject_reason, cnpj_digits) FROM parsed WHERE reject_reason IS NULL
           QUALIFY row_number() OVER (PARTITION BY cnpj, product, collected_on ORDER BY priority DESC, source_id) = 1"""
    )
    clean_rows = total_parsed - sum(rejects.values())
    rows_out = con.execute("SELECT count(*) FROM deduped").fetchone()[0]

    # Write to a temp dir, then swap: readers never see a half-written silver.
    tmp = settings.silver.with_name("silver.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "prices").mkdir(parents=True)
    con.execute(
        f"""COPY (SELECT cnpj, station_name, brand, region, state, city, neighborhood, street, street_number, complement, zip,
                          product, unit, collected_on, price, purchase_price, source_id,
                          year(collected_on) AS year, month(collected_on) AS month
                   FROM deduped ORDER BY collected_on, state, cnpj)
            TO '{(tmp / "prices").as_posix()}' (FORMAT parquet, PARTITION_BY (year, month), COMPRESSION zstd)"""
    )
    con.execute(
        f"""COPY (SELECT source_id, reject_reason, cnpj_digits AS cnpj, product, state, collected_on, price
                   FROM parsed WHERE reject_reason IS NOT NULL LIMIT 100000)
            TO '{(tmp / "_rejects.parquet").as_posix()}' (FORMAT parquet)"""
    )
    span = con.execute("SELECT min(collected_on), max(collected_on) FROM deduped").fetchone()
    metrics = {
        "rows_read": sum(s["rows_read"] for s in sources.values()),
        "rows_malformed": sum(s["rows_malformed"] for s in sources.values()),
        "rows_rejected": sum(rejects.values()),
        "rejects_by_reason": rejects,
        "rows_clean": clean_rows,
        "duplicates_removed": clean_rows - rows_out,
        "rows_silver": rows_out,
        "date_from": span[0].isoformat() if span[0] else None,
        "date_to": span[1].isoformat() if span[1] else None,
        "stations": con.execute("SELECT count(DISTINCT cnpj) FROM deduped").fetchone()[0],
        "products": sorted(r[0] for r in con.execute("SELECT DISTINCT product FROM deduped").fetchall()),
        "sources": sources,
    }
    (tmp / "_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    con.close()

    if settings.silver.exists():
        shutil.rmtree(settings.silver)
    tmp.replace(settings.silver)
    log(f"  silver: {rows_out:,} rows ({metrics['duplicates_removed']:,} duplicates, {metrics['rows_rejected']:,} rejected)")
    return metrics
