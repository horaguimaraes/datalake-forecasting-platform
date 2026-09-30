"""Deterministic synthetic ANP files, so tests exercise the *real* parsing path without network access.

The price process is designed so the truth is known:
  - a common national level (random walk with drift) shared by all states,
  - a fixed offset per state,
  - independent noise on every station price, and each station is surveyed in only some weeks,
    so the weekly state average is a *noisy* measure of the level (like the real ANP survey).
That noise is exactly what a model can partly correct, which gives tests a real effect to detect.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HEADER = (
    "Regiao - Sigla;Estado - Sigla;Municipio;Revenda;CNPJ da Revenda;Nome da Rua;Numero Rua;Complemento;"
    "Bairro;Cep;Produto;Data da Coleta;Valor de Venda;Valor de Compra;Unidade de Medida;Bandeira"
)
PRODUCT_FILE = {"GASOLINA": "gasolina-etanol", "ETANOL": "gasolina-etanol", "DIESEL S10": "diesel-gnv"}
BASE_PRICE = {"GASOLINA": 6.0, "ETANOL": 4.2, "DIESEL S10": 6.5}
REGION = {"SP": "SE", "RJ": "SE", "MG": "SE", "PR": "S", "BA": "NE", "GO": "CO"}
BRANDS = ["VIBRA", "IPIRANGA", "RAIZEN", "BRANCA"]


@dataclass(frozen=True)
class SynthConfig:
    start: date = date(2023, 1, 2)  # a Monday
    weeks: int = 130
    states: tuple[str, ...] = ("SP", "RJ", "MG", "PR", "BA", "GO")
    stations_per_state: int = 28
    p_survey: float = 0.5  # chance a station is surveyed in a given week
    station_noise: float = 0.06  # R$ std of a single station price around the state level
    level_vol: float = 0.006  # weekly std of the national log level
    drift: float = 0.0008
    seed: int = 7


def _cnpj(index: int) -> str:
    d = f"{index:014d}"
    return f" {d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:]}"


DEFAULT = SynthConfig()


def generate_rows(cfg: SynthConfig = DEFAULT) -> list[dict[str, str]]:
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, str]] = []
    level = {p: 0.0 for p in BASE_PRICE}
    state_offset = {(p, s): rng.normal(0, 0.03) for p in BASE_PRICE for s in cfg.states}
    stations = [(i, s) for i, s in enumerate((s for s in cfg.states for _ in range(cfg.stations_per_state)), start=1)]

    for week in range(cfg.weeks):
        monday = cfg.start + timedelta(weeks=week)
        for product in BASE_PRICE:
            level[product] += cfg.drift + rng.normal(0, cfg.level_vol)
        for idx, state in stations:
            for product, base in BASE_PRICE.items():
                if rng.random() > cfg.p_survey:
                    continue
                true = base * np.exp(level[product] + state_offset[(product, state)])
                price = max(true + rng.normal(0, cfg.station_noise), 0.5)
                day = monday + timedelta(days=int(rng.integers(0, 5)))
                rows.append(
                    {
                        "Regiao - Sigla": REGION[state],
                        "Estado - Sigla": state,
                        "Municipio": f"CITY {state}",
                        "Revenda": f"POSTO {idx:04d} LTDA",
                        "CNPJ da Revenda": _cnpj(idx),
                        "Nome da Rua": "RUA A",
                        "Numero Rua": "10",
                        "Complemento": "",
                        "Bairro": "CENTRO",
                        "Cep": "00000-000",
                        "Produto": product,
                        "Data da Coleta": day.strftime("%d/%m/%Y"),
                        "Valor de Venda": f"{price:.2f}".replace(".", ","),
                        "Valor de Compra": "",
                        "Unidade de Medida": "R$ / litro",
                        "Bandeira": BRANDS[idx % len(BRANDS)],
                    }
                )
    return rows


def _line(row: dict[str, str]) -> str:
    return ";".join(row[c] for c in HEADER.split(";"))


def write_monthly_files(directory: Path, cfg: SynthConfig = DEFAULT, gz_every: int = 3) -> list[Path]:
    """Write one ANP-style CSV per (month, fuel kind), exactly like the portal (BOM, CRLF, `;`)."""
    directory.mkdir(parents=True, exist_ok=True)
    buckets: dict[tuple[str, str], list[str]] = {}
    for row in generate_rows(cfg):
        d = row["Data da Coleta"]
        ym = f"{d[6:10]}-{d[3:5]}"
        buckets.setdefault((ym, PRODUCT_FILE[row["Produto"]]), []).append(_line(row))
    paths = []
    for n, ((ym, kind), lines) in enumerate(sorted(buckets.items())):
        text = "﻿" + "\r\n".join([HEADER, *lines]) + "\r\n"
        if n % gz_every == 0:
            path = directory / f"{ym}_{kind}.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                handle.write(text)
        else:
            path = directory / f"{ym}_{kind}.csv"
            path.write_text(text, encoding="utf-8", newline="")
        paths.append(path)
    return paths


def csv_text(rows: list[dict[str, str]], header: str = HEADER) -> str:
    return "﻿" + "\r\n".join([header, *(_line(r) for r in rows)]) + "\r\n"


def make_row(**overrides: str) -> dict[str, str]:
    """A single valid ANP row; override any field by its raw column name."""
    row = {
        "Regiao - Sigla": "SE",
        "Estado - Sigla": "SP",
        "Municipio": "SAO PAULO",
        "Revenda": "POSTO TESTE LTDA",
        "CNPJ da Revenda": " 11.222.333/0001-44",
        "Nome da Rua": "RUA A",
        "Numero Rua": "10",
        "Complemento": "",
        "Bairro": "CENTRO",
        "Cep": "01000-000",
        "Produto": "GASOLINA",
        "Data da Coleta": "05/03/2024",
        "Valor de Venda": "6,50",
        "Valor de Compra": "",
        "Unidade de Medida": "R$ / litro",
        "Bandeira": "VIBRA",
    }
    row.update(overrides)
    return row
