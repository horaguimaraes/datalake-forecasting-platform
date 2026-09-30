"""Where the ANP files live.

ANP publishes the same survey in several shapes, and the naming has changed over the years
(verified against the live portal):

  2023-01 .. 2025-12   dsan/{year}/precos-{fuel}-{MM}.csv                       (monthly)
  2026-06, 2026-08     dsan/{year}/{MM}-dados-abertos-precos-{year}-{MM}-{fuel}.csv
  a whole semester     dsas/ca/ca-{year}-0{1|2}.zip                              (all automotive fuels)
  latest 4 weeks       qus/ultimas-4-semanas-{fuel}.csv                          (rolling)

Some months are simply not published (e.g. 2026-07 at the time of writing), so the resolver
tries every known shape and falls back to the semester archive before declaring a gap.
"""

from __future__ import annotations

BASE = "https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/arquivos/shpc"
FUELS = ("gasolina-etanol", "diesel-gnv")


def parse_month(value: str) -> tuple[int, int]:
    """'2024-03' -> (2024, 3)."""
    try:
        year_text, month_text = value.split("-")
        year, month = int(year_text), int(month_text)
    except ValueError as error:
        raise ValueError(f"Invalid month '{value}', expected YYYY-MM") from error
    if not (1 <= month <= 12 and 1990 <= year <= 2100):
        raise ValueError(f"Invalid month '{value}', expected YYYY-MM")
    return year, month


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) between two 'YYYY-MM' strings."""
    (y, m), (end_y, end_m) = parse_month(start), parse_month(end)
    if (y, m) > (end_y, end_m):
        raise ValueError(f"start {start} is after end {end}")
    months = []
    while (y, m) <= (end_y, end_m):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def monthly_candidates(fuel: str, year: int, month: int) -> tuple[str, ...]:
    """Every URL shape a monthly file has used, most likely first."""
    mm = f"{month:02d}"
    return (
        f"{BASE}/dsan/{year}/precos-{fuel}-{mm}.csv",
        f"{BASE}/dsan/{year}/{mm}-dados-abertos-precos-{year}-{mm}-{fuel}.csv",
    )


def semester_url(year: int, half: int) -> str:
    return f"{BASE}/dsas/ca/ca-{year}-0{half}.zip"


def last4weeks_url(fuel: str) -> str:
    return f"{BASE}/qus/ultimas-4-semanas-{fuel}.csv"


def half_of(month: int) -> int:
    return 1 if month <= 6 else 2
