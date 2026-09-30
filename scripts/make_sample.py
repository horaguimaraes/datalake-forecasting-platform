"""Build the small offline sample in data/sample from a real bronze layer.

It keeps the *raw lines* of K randomly chosen stations per state (so the sample exercises the real
parsing path, including its quirks), only for the three products the model uses, and writes one
gzipped CSV per source file.

    python scripts/make_sample.py --root data/lake --per-state 20
"""

from __future__ import annotations

import argparse
import gzip
import zipfile
from pathlib import Path

import duckdb

from fuellake.bronze import Manifest
from fuellake.config import Settings
from fuellake.ml.dataset import MODEL_PRODUCTS
from fuellake.silver import HEADER_ALIASES, detect_encoding, map_header


def choose_stations(settings: Settings, per_state: int, seed: int = 42) -> set[str]:
    """A seeded *random* sample of stations per state.

    Random on purpose: picking the best-covered stations would keep only those surveyed every week,
    which removes the week-to-week sampling noise the real survey has and makes the data unrepresentative.
    """
    con = duckdb.connect(":memory:")
    rows = con.execute(
        f"""SELECT cnpj FROM (
                SELECT cnpj, state,
                       row_number() OVER (PARTITION BY state ORDER BY hash(cnpj || '{seed}')) AS rk
                FROM (SELECT DISTINCT cnpj, state
                      FROM read_parquet('{settings.silver_glob}', hive_partitioning = true)
                      WHERE product IN ({", ".join(repr(p) for p in MODEL_PRODUCTS)}))
            ) WHERE rk <= {per_state}"""
    ).fetchall()
    return {r[0] for r in rows}


def filter_lines(text: str, header_line: str, names: list[str], keep: set[str]) -> list[str]:
    cnpj_i, product_i = names.index("cnpj_raw"), names.index("product")
    out = [header_line]
    for line in text.splitlines()[1:]:
        cells = line.split(";")
        if len(cells) <= max(cnpj_i, product_i):
            continue
        digits = "".join(c for c in cells[cnpj_i] if c.isdigit())
        digits = digits.zfill(14) if len(digits) == 13 else digits
        if digits in keep and cells[product_i].strip().upper() in MODEL_PRODUCTS:
            out.append(line)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/lake")
    parser.add_argument("--per-state", type=int, default=20)
    parser.add_argument("--out", default="data/sample")
    args = parser.parse_args()

    settings = Settings(Path(args.root))
    keep = choose_stations(settings, args.per_state)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv.gz"):
        old.unlink()

    total_rows = 0
    for entry in Manifest.load(settings.bronze_manifest).ok_entries():
        path = settings.bronze / entry["path"]
        members: list[tuple[str, bytes]] = []
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = [
                    (Path(i.filename).stem, archive.read(i)) for i in archive.infolist() if i.filename.lower().endswith(".csv")
                ]
        else:
            members = [(path.stem, path.read_bytes())]
        for _stem, raw in members:
            tmp = out_dir / "_tmp.csv"
            tmp.write_bytes(raw)
            encoding = detect_encoding(tmp)
            tmp.unlink()
            text = raw.decode(encoding).lstrip("﻿")
            header_line = text.splitlines()[0]
            names = map_header([c.strip().strip('"') for c in header_line.split(";")])
            lines = filter_lines(text, header_line, names, keep)
            if len(lines) <= 1:
                continue
            safe = entry["id"].replace(":", "_").replace("/", "_")
            target = out_dir / f"{safe}.csv.gz"
            with gzip.open(target, "wt", encoding="utf-8", newline="") as handle:
                handle.write("\r\n".join(lines) + "\r\n")
            total_rows += len(lines) - 1
    files = sorted(out_dir.glob("*.csv.gz"))
    size = sum(f.stat().st_size for f in files)
    print(f"{len(keep)} stations, {total_rows:,} rows, {len(files)} files, {size / 1e6:.1f} MB -> {out_dir}")


if __name__ == "__main__":
    assert HEADER_ALIASES
    main()
