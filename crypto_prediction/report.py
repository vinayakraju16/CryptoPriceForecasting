"""Generate a self-contained HTML report - no Flask, no server, no network.

``python -m crypto_prediction.report --out reports/latest.html`` produces a
static page with the screening table, per-coin forecasts and risk stats. It is
the artefact to attach to a PR or drop into a portfolio page, and it works on a
host with nothing but the standard library.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .core import discover_datasets, load_series, screen_universe
from .core.indicators import summary_stats
from .core.signals import recommend

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_CSV_DIR = os.path.join(os.path.dirname(ROOT), "CSV")

_SIGNAL_CLASS = {
    "strong_buy": "sig-strong-buy",
    "buy": "sig-buy",
    "hold": "sig-hold",
    "sell": "sig-sell",
    "strong_sell": "sig-strong-sell",
}


def _fmt(value, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}{suffix}"
        return f"{value:.{digits}f}{suffix}"
    return f"{html.escape(str(value))}{suffix}"


def _pct(value) -> str:
    if not isinstance(value, float):
        return "n/a"
    return f"{value * 100:+.2f}%"


def build_report(csv_dir: str, model: str = "ensemble", top: Optional[int] = None) -> Dict[str, object]:
    datasets: Dict[str, object] = {}
    quality: Dict[str, dict] = {}
    for symbol, path in discover_datasets(csv_dir).items():
        series, report = load_series(path, symbol)
        quality[symbol] = report.to_dict()
        if report.is_valid:
            datasets[symbol] = series

    rows = screen_universe(datasets, model, top=top)
    proposals = recommend(datasets, model, top=top or 3)
    stats = []
    for symbol, series in sorted(datasets.items()):
        summary = summary_stats(series.closes)
        summary["symbol"] = symbol
        summary["bars"] = len(series)
        summary["start"] = str(series.start)
        summary["end"] = str(series.end)
        stats.append(summary)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "csv_dir": csv_dir,
        "coins": len(datasets),
        "quality": quality,
        "screen": rows,
        "recommendations": proposals,
        "stats": stats,
    }


def render_html(report: Dict[str, object]) -> str:
    rows = report["screen"]
    stats = report["stats"]

    screen_rows = []
    for row in rows:
        if "error" in row:
            screen_rows.append(
                f"<tr><td>{html.escape(str(row['symbol']))}</td>"
                f"<td colspan='8' class='muted'>skipped: {html.escape(str(row['error']))}</td></tr>"
            )
            continue
        signal = str(row.get("signal", "hold"))
        screen_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['symbol']))}</td>"
            f"<td class='{_SIGNAL_CLASS.get(signal, 'sig-hold')}'>{html.escape(signal)}</td>"
            f"<td>{_fmt(row.get('last_price'), 4)}</td>"
            f"<td>{_fmt(row.get('predicted_next_price'), 4)}</td>"
            f"<td>{_pct(row.get('expected_return'))}</td>"
            f"<td>{_fmt(row.get('confidence'), 2)}</td>"
            f"<td>{_fmt(row.get('directional_accuracy'), 3)}</td>"
            f"<td>{_fmt(row.get('volatility_annual'), 2)}</td>"
            f"<td>{_fmt(row.get('max_drawdown'), 3)}</td>"
            "</tr>"
        )

    stat_rows = []
    for row in stats:
        stat_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['symbol']))}</td>"
            f"<td>{row.get('bars')}</td>"
            f"<td>{html.escape(str(row.get('start')))}</td>"
            f"<td>{html.escape(str(row.get('end')))}</td>"
            f"<td>{_pct(row.get('total_return'))}</td>"
            f"<td>{_fmt(row.get('volatility_annual'), 2)}</td>"
            f"<td>{_fmt(row.get('max_drawdown'), 3)}</td>"
            f"<td>{_fmt(row.get('sharpe'), 2)}</td>"
            f"<td>{_fmt(row.get('var_95'), 4)}</td>"
            "</tr>"
        )

    quality_rows = []
    for symbol, row in report.get("quality", {}).items():
        status = "OK" if row.get("is_valid") else "REJECTED"
        quality_rows.append(
            "<tr>"
            f"<td>{html.escape(symbol)}</td>"
            f"<td class='{'ok' if row.get('is_valid') else 'bad'}'>{status}</td>"
            f"<td>{row.get('bars_out', 0)}</td>"
            f"<td>{row.get('dropped_invalid', 0)}</td>"
            f"<td>{row.get('dropped_duplicates', 0)}</td>"
            f"<td class='muted'>{html.escape('; '.join(row.get('errors', []) + row.get('warnings', [])[:2]))}</td>"
            "</tr>"
        )

    rec_rows = []
    for row in report.get("recommendations", []):
        allocation = row.get("allocation", {})
        signal = str(row.get("signal", "hold"))
        rec_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('symbol')))}</td>"
            f"<td class='{_SIGNAL_CLASS.get(signal, 'sig-hold')}'>{html.escape(signal)}</td>"
            f"<td>{_fmt(row.get('last_price'), 4)}</td>"
            f"<td>{_fmt(row.get('predicted_next_price'), 4)}</td>"
            f"<td>{_pct(row.get('expected_return'))}</td>"
            f"<td>{_fmt(allocation.get('cash'), 2)}</td>"
            f"<td>{_fmt(allocation.get('weight', 0) * 100, 1, '%')}</td>"
            f"<td>{_fmt(allocation.get('risk_cash'), 2)}</td>"
            "</tr>"
        )

    return _TEMPLATE.format(
        generated_at=html.escape(str(report["generated_at"])),
        model=html.escape(str(report["model"])),
        coins=report["coins"],
        csv_dir=html.escape(str(report["csv_dir"])),
        screen_rows="\n".join(screen_rows) or "<tr><td colspan='9'>no data</td></tr>",
        rec_rows="\n".join(rec_rows) or "<tr><td colspan='8' class='muted'>nothing cleared the bar - staying flat is a valid call</td></tr>",
        stat_rows="\n".join(stat_rows),
        quality_rows="\n".join(quality_rows),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="crypto_prediction.report")
    parser.add_argument("--csv-dir", default=os.environ.get("CRYPTO_CSV_DIR", DEFAULT_CSV_DIR))
    parser.add_argument("--model", default="ensemble")
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--out", default="reports/latest.html")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    report = build_report(args.csv_dir, args.model, args.top)
    document = render_html(report)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(document)
    print(f"wrote {out_path} ({len(document):,} bytes, {report['coins']} coins)")

    if args.json_out:
        json_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"wrote {json_path}")
    return 0


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crypto Forecasting Report</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 2rem;
         background: #0f1216; color: #e6e9ef; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 .25rem; }}
  h2 {{ font-size: 1.15rem; margin: 2rem 0 .6rem; color: #9fb3c8; }}
  .meta {{ color: #8b98a9; font-size: .85rem; margin-bottom: 1rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; background: #151a21;
           border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: .55rem .7rem; text-align: right; border-bottom: 1px solid #222a34; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #1b222c; color: #9fb3c8; font-weight: 600; position: sticky; top: 0; }}
  tr:hover td {{ background: #1a212b; }}
  .sig-strong-buy {{ color: #4ade80; font-weight: 700; }}
  .sig-buy {{ color: #86efac; }}
  .sig-hold {{ color: #cbd5e1; }}
  .sig-sell {{ color: #fdba74; }}
  .sig-strong-sell {{ color: #f87171; font-weight: 700; }}
  .ok {{ color: #4ade80; }} .bad {{ color: #f87171; }}
  .muted {{ color: #7c8998; text-align: left; }}
  .note {{ margin-top: 2rem; padding: 1rem; background: #171d26; border-left: 3px solid #3b82f6;
           border-radius: 6px; font-size: .85rem; color: #a8b6c7; }}
  @media (max-width: 760px) {{ body {{ padding: 1rem; }} table {{ font-size: .75rem; }} }}
</style>
</head>
<body>
  <h1>Crypto Forecasting Report</h1>
  <div class="meta">generated {generated_at} &middot; model <b>{model}</b> &middot; {coins} coins &middot; data: {csv_dir}</div>

  <h2>Screening (ranked by forecast edge / model error)</h2>
  <table>
    <thead><tr>
      <th>Coin</th><th>Signal</th><th>Last</th><th>Next (pred)</th><th>Exp. return</th>
      <th>Conf.</th><th>Dir. acc.</th><th>Vol (ann.)</th><th>Max DD</th>
    </tr></thead>
    <tbody>
{screen_rows}
    </tbody>
  </table>

  <h2 style="color:#4ade80">Recommended allocations (positive edge only)</h2>
  <table>
    <thead><tr>
      <th>Coin</th><th>Signal</th><th>Last</th><th>Next (pred)</th><th>Exp. return</th>
      <th>Allocate</th><th>Weight</th><th>Risk at stake</th>
    </tr></thead>
    <tbody>
{rec_rows}
    </tbody>
  </table>

  <h2>Risk statistics (full history)</h2>
  <table>
    <thead><tr>
      <th>Coin</th><th>Bars</th><th>From</th><th>To</th><th>Total ret.</th>
      <th>Vol (ann.)</th><th>Max DD</th><th>Sharpe</th><th>VaR 95%</th>
    </tr></thead>
    <tbody>
{stat_rows}
    </tbody>
  </table>

  <h2>Data quality</h2>
  <table>
    <thead><tr>
      <th>Coin</th><th>Status</th><th>Bars kept</th><th>Invalid dropped</th><th>Dupes</th><th>Notes</th>
    </tr></thead>
    <tbody>
{quality_rows}
    </tbody>
  </table>

  <div class="note">
    <b>Read this before acting on it.</b> These numbers are statistical projections from
    historical prices only. They ship with a confidence band precisely because single-point
    price forecasts are unreliable, and crypto is the most volatile mainstream asset class.
    Nothing here is financial advice.
  </div>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
