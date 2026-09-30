"""Self-contained HTML report: forecasts, model comparison, data quality and lineage.

Charts are embedded as PNGs, and there is no JavaScript, so the file can be emailed or attached
to a ticket and still open anywhere.
"""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import UTC, datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .bronze import Manifest  # noqa: E402
from .config import Settings  # noqa: E402
from .ml.dataset import PanelConfig, build_panel  # noqa: E402
from .ml.train import load_metrics  # noqa: E402

ACCENT, WARN, MUTED = "#0e75b6", "#e08a00", "#7f8c99"
PRODUCT_LABEL = {"GASOLINA": "Gasoline", "ETANOL": "Ethanol", "DIESEL S10": "Diesel S10"}


def _png(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _forecast_chart(panel: pd.DataFrame, forecast: pd.DataFrame, history_weeks: int = 70) -> str:
    products = [p for p in PRODUCT_LABEL if p in set(forecast["product"])]
    fig, axes = plt.subplots(1, len(products), figsize=(4.6 * len(products), 3.4), squeeze=False)
    for ax, product in zip(axes[0], products, strict=True):
        hist = panel[(panel["product"] == product) & (panel["state"] == "BR")].tail(history_weeks)
        fc = forecast[(forecast["product"] == product) & (forecast["state"] == "BR")].sort_values("horizon")
        ax.plot(hist["week_start"], hist["price"], color=ACCENT, lw=1.8, label="History")
        bridged = hist[hist["imputed"]]
        if not bridged.empty:
            ax.scatter(bridged["week_start"], bridged["price"], s=10, color=WARN, zorder=3, label="Gap bridged")
        start = pd.DataFrame({"target_week": [fc["origin_week"].iloc[0]], "forecast": [fc["price_origin"].iloc[0]]})
        line = pd.concat([start, fc[["target_week", "forecast"]]])
        ax.plot(line["target_week"], line["forecast"], color=ACCENT, lw=1.8, ls="--", label="Forecast")
        ax.fill_between(fc["target_week"], fc["lo"], fc["hi"], color=ACCENT, alpha=0.15, label="80% interval")
        ax.set_title(f"{PRODUCT_LABEL[product]}, Brazil (R$/L)", fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        fig.autofmt_xdate(rotation=30)
    axes[0][0].legend(frameon=False, fontsize=7, loc="upper left")
    return _png(fig)


def _skill_chart(metrics: dict[str, Any]) -> str:
    by_h = pd.DataFrame(metrics["by_horizon"])
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for model, group in by_h[by_h["model"].isin(["blend", "gbm", "ridge"])].groupby("model"):
        emphasis = model == metrics["champion"]
        ax.plot(
            group["horizon"],
            group["skill_vs_naive"] * 100,
            marker="o",
            ms=4,
            lw=2.2 if emphasis else 1.2,
            color=ACCENT if emphasis else MUTED,
            alpha=1 if emphasis else 0.7,
            label=model,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title('Above 0 = beats "no change"', fontsize=9, loc="left", color=MUTED)
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("Error reduction vs naive (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def _card(label: str, value: str, note: str = "") -> str:
    extra = f"<small>{html.escape(note)}</small>" if note else ""
    return f"<div class='card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong>{extra}</div>"


_CSS = """
:root{--bg:#f6f8fa;--fg:#0f2027;--muted:#586069;--line:#e1e4e8;--accent:#0e75b6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
header{background:linear-gradient(135deg,#0F2027,#2C5364);color:#fff;padding:34px 7vw}
header h1{margin:0 0 4px;font-size:26px}header p{margin:0;opacity:.8}
main{padding:22px 7vw 60px;max-width:1280px;margin:auto}h2{margin:34px 0 10px;font-size:19px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card span{display:block;color:var(--muted);font-size:13px}.card strong{font-size:22px}.card small{display:block;color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}
th,td{padding:7px 12px;text-align:left;border-bottom:1px solid var(--line)}th{background:#eef2f6}th.num,td.num{text-align:right;font-variant-numeric:tabular-nums}
img{max-width:100%;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px}
.pos{color:#c62828}.neg{color:#1b8a4b}.pill{border-radius:999px;padding:1px 9px;font-size:12px;background:#eef2f6}
.warn{color:#b26a00}.ok{color:#1b8a4b}p.note{color:var(--muted);font-size:13px;margin:6px 0 0}
footer{color:var(--muted);font-size:13px;text-align:center;padding:24px}
"""


def build_report(settings: Settings, weekly: pd.DataFrame, forecast: pd.DataFrame) -> str:
    metrics = load_metrics(settings)
    if metrics is None:
        raise FileNotFoundError("No trained model: run the train step first.")
    panel = build_panel(
        weekly, PanelConfig(**{**metrics["config"]["panel"], "products": tuple(metrics["config"]["panel"]["products"])})
    )
    silver = json.loads(settings.silver_metrics.read_text(encoding="utf-8"))
    quality_path = settings.reports / "quality_report.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None
    manifest = Manifest.load(settings.bronze_manifest)
    ok_files = manifest.ok_entries()

    ci = metrics.get("champion_skill_ci")
    ranking = pd.DataFrame(metrics["ranking"])
    champ_row = ranking[ranking["model"] == metrics["champion"]].iloc[0]
    coverage = metrics["interval_coverage"]

    cards = [
        _card("Champion model", metrics["champion"], f"MAE R$ {champ_row['mae']:.3f} avg over horizons"),
        _card(
            "Error vs naive",
            f"{champ_row['skill_vs_naive']:+.1%}",
            f"95% CI {ci['lo']:+.1%} to {ci['hi']:+.1%}" if ci else "baseline",
        ),
        _card("80% interval coverage", f"{coverage['empirical']:.0%}", f"on {coverage['n']:,} unseen forecasts"),
        _card("Series modelled", f"{metrics['n_series']}", f"data through {metrics['data_through']}"),
        _card("Silver rows", f"{silver['rows_silver']:,}", f"{silver['stations']:,} stations"),
    ]

    origin = forecast["origin_week"].iloc[0]
    four = forecast[(forecast["horizon"] == 4) & (forecast["state"] != "BR")].copy()
    four["change"] = four["forecast"] / four["price_origin"] - 1
    movers = pd.concat([four.nlargest(6, "change"), four.nsmallest(6, "change")]).drop_duplicates(["product", "state"])
    mover_rows = "".join(
        f"<tr><td>{html.escape(PRODUCT_LABEL.get(r.product, r.product))}</td><td>{r.state}</td>"
        f"<td class='num'>{r.price_origin:.2f}</td><td class='num'>{r.forecast:.2f}</td>"
        f"<td class='num {'pos' if r.change > 0 else 'neg'}'>{r.change:+.1%}</td></tr>"
        for r in movers.sort_values("change", ascending=False).itertuples()
    )

    model_rows = "".join(
        f"<tr><td>{'<b>' if r.model == metrics['champion'] else ''}{html.escape(r.model)}{'</b>' if r.model == metrics['champion'] else ''}</td>"
        f"<td class='num'>{r.mae:.4f}</td><td class='num'>{r.mape:.2%}</td><td class='num'>{r.skill_vs_naive:+.1%}</td></tr>"
        for r in ranking.itertuples()
    )

    quality_html = "<p>No quality report found.</p>"
    if quality:
        s = quality["summary"]
        flagged = [c for c in quality["checks"] if not c["passed"]]
        rows = (
            "".join(
                f"<tr><td class='{'warn' if c['severity'] == 'warn' else 'pos'}'>{c['severity']}</td><td>{html.escape(c['name'])}</td>"
                f"<td>{html.escape(str(c['detail'])[:300])}</td></tr>"
                for c in flagged
            )
            or "<tr><td colspan='3' class='ok'>Every check passed.</td></tr>"
        )
        quality_html = (
            f"<p><b>{s['passed']}/{s['checks']}</b> checks passed, {s['errors']} errors, {s['warnings']} warnings.</p>"
            f"<table><thead><tr><th>Severity</th><th>Check</th><th>Detail</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    rejects = ", ".join(f"{k}: {v:,}" for k, v in silver["rejects_by_reason"].items()) or "none"
    lineage = (
        f"<table><tbody>"
        f"<tr><td>Bronze files (raw, checksummed)</td><td class='num'>{len(ok_files)}</td></tr>"
        f"<tr><td>Bronze size</td><td class='num'>{sum(e['bytes'] for e in ok_files) / 1e6:,.0f} MB</td></tr>"
        f"<tr><td>Rows read</td><td class='num'>{silver['rows_read']:,}</td></tr>"
        f"<tr><td>Rejected ({html.escape(rejects)})</td><td class='num'>{silver['rows_rejected']:,}</td></tr>"
        f"<tr><td>Duplicates removed (overlapping sources)</td><td class='num'>{silver['duplicates_removed']:,}</td></tr>"
        f"<tr><td>Silver rows</td><td class='num'>{silver['rows_silver']:,}</td></tr>"
        f"<tr><td>Period</td><td class='num'>{silver['date_from']} to {silver['date_to']}</td></tr>"
        f"</tbody></table>"
    )

    body = f"""
<header><h1>Fuel price forecast</h1>
<p>Origin week {origin:%Y-%m-%d} · model run {html.escape(str(metrics["run_id"]))} · generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC</p></header>
<main>
<div class='cards'>{"".join(cards)}</div>
<h2>National forecast</h2>
<img src='{_forecast_chart(panel, forecast)}' alt='National forecast by product'>
<p class='note'>Solid: weekly average price. Dashed: forecast up to 8 weeks ahead with an 80% interval. Orange dots are weeks
ANP did not publish; they were bridged by interpolation and never used as ground truth.</p>
<h2>How much better than "no change"?</h2>
<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;align-items:start'>
<div><table><thead><tr><th>Model</th><th class='num'>MAE (R$)</th><th class='num'>MAPE</th><th class='num'>vs naive</th></tr></thead><tbody>{model_rows}</tbody></table>
<p class='note'>Rolling-origin backtest, {metrics["backtest"]["n_folds"]} folds, origins {metrics["backtest"]["test_origins_from"]} to
{metrics["backtest"]["test_origins_to"]}, horizons 1-{max(metrics["backtest"]["horizons"])} weeks. Training never sees data after each fold's cutoff.</p></div>
<img src='{_skill_chart(metrics)}' alt='Skill by horizon'></div>
<h2>Biggest expected moves in 4 weeks (states)</h2>
<table><thead><tr><th>Product</th><th>State</th><th class='num'>Now (R$)</th><th class='num'>Forecast (R$)</th><th class='num'>Change</th></tr></thead><tbody>{mover_rows}</tbody></table>
<h2>Data quality</h2>{quality_html}
<h2>Lineage</h2>{lineage}
</main><footer>Made with fuellake · data: ANP open data</footer>"""
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Fuel price forecast</title><style>{_CSS}</style></head><body>{body}</body></html>"


def write_report(settings: Settings, weekly: pd.DataFrame, forecast: pd.DataFrame) -> str:
    settings.reports.mkdir(parents=True, exist_ok=True)
    path = settings.reports / "forecast_report.html"
    path.write_text(build_report(settings, weekly, forecast), encoding="utf-8")
    return str(path)
