"""Dependency-free HTTP server exposing the same JSON API as the Flask app.

Why this exists: the Flask UI needs Flask, Werkzeug, Jinja2 and friends. This
server needs nothing but the standard library, so the project is usable - and
demonstrable - on a bare host, in CI, or in a container where you do not want a
web framework. It serves the same ``/api/*`` contract, and a single-page
dashboard that talks to it.

    python -m crypto_prediction.server --port 8501
    curl 'http://127.0.0.1:8501/api/forecast?coin=Bitcoin&model=drift'

Endpoints
---------
GET /healthz                      liveness + dataset/model counts
GET /api/datasets                 coins with row counts and quality reports
GET /api/models                   built-in models, with descriptions
GET /api/forecast?coin&model&steps
GET /api/screen?model&top
GET /api/recommend?model&capital&risk
GET /api/backtest?coin&model
GET /api/stats?coin
GET /                             minimal interactive dashboard

This server is for local use and demos. It binds 127.0.0.1 by default; it has no
authentication, so do not expose it to the internet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

from .core import (
    MODEL_REGISTRY,
    build_model,
    build_prediction,
    discover_datasets,
    load_series,
    make_windows,
    screen_universe,
    walk_forward,
    MinMax,
)
from .core.dataset import DataError
from .core.indicators import summary_stats
from .core.models import available_models
from .core.signals import recommend

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_DIR = os.environ.get("CRYPTO_CSV_DIR", os.path.join(os.path.dirname(HERE), "CSV"))


class ForecastService:
    """Loads datasets once and answers every API call."""

    def __init__(self, csv_dir: str) -> None:
        self.csv_dir = csv_dir
        self.datasets: Dict[str, object] = {}
        self.reports: Dict[str, dict] = {}
        self.reload()

    def reload(self) -> None:
        self.datasets = {}
        self.reports = {}
        for symbol, path in discover_datasets(self.csv_dir).items():
            series, report = load_series(path, symbol)
            self.reports[symbol] = report.to_dict()
            if report.is_valid:
                self.datasets[symbol] = series

    def coin(self, symbol: str):
        series = self.datasets.get(symbol)
        if series is None:
            raise DataError(f"unknown coin {symbol!r}; available: {sorted(self.datasets)}")
        return series

    # ------------------------------------------------------------- handlers
    def health(self) -> dict:
        return {
            "status": "ok",
            "coins": len(self.datasets),
            "rejected": sum(1 for r in self.reports.values() if not r["is_valid"]),
            "models": sorted(MODEL_REGISTRY),
            "csv_dir": self.csv_dir,
        }

    def datasets_payload(self) -> dict:
        rows = []
        for symbol, series in sorted(self.datasets.items()):
            rows.append(
                {
                    "symbol": symbol,
                    "start": str(series.start),
                    "end": str(series.end),
                    "bars": len(series),
                    "quality": self.reports.get(symbol, {}),
                }
            )
        return {"count": len(rows), "datasets": rows}

    def forecast(self, params: dict) -> dict:
        symbol = params.get("coin", ["Bitcoin"])[0]
        model = params.get("model", ["ensemble"])[0]
        steps = max(1, int(params.get("steps", ["1"])[0]))
        lookback = max(5, int(params.get("lookback", ["30"])[0]))
        prediction = build_prediction(self.coin(symbol), model, lookback=lookback, horizon=steps)
        return prediction.to_dict()

    def screen(self, params: dict) -> dict:
        model = params.get("model", ["ensemble"])[0]
        top = params.get("top", [None])[0]
        rows = screen_universe(self.datasets, model, top=int(top) if top else None)
        return {"model": model, "count": len(rows), "rows": rows}

    def recommend(self, params: dict) -> dict:
        model = params.get("model", ["drift"])[0]
        capital = float(params.get("capital", ["1000"])[0])
        risk = float(params.get("risk", ["0.02"])[0])
        proposals = recommend(self.datasets, model, capital=capital, risk_per_trade=risk)
        return {"model": model, "capital": capital, "count": len(proposals), "proposals": proposals}

    def backtest(self, params: dict) -> dict:
        symbol = params.get("coin", ["Bitcoin"])[0]
        model = params.get("model", ["drift"])[0]
        lookback = max(5, int(params.get("lookback", ["30"])[0]))
        series = self.coin(symbol)
        closes = series.closes
        scaler = MinMax()
        scaler.fit(closes[: max(2, int(len(closes) * 0.7))])
        xs, ys = make_windows(scaler.transform(closes), lookback, 1)
        result = walk_forward(
            build_model(model), xs, ys, scaler, series.dates[lookback:], min_train=120
        )
        return result.to_dict()

    def stats(self, params: dict) -> dict:
        symbol = params.get("coin", ["Bitcoin"])[0]
        return {"symbol": symbol, **summary_stats(self.coin(symbol).closes)}

    def models(self) -> dict:
        return {"models": available_models(), "default": "ensemble"}


def make_handler(service: ForecastService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "crypto-forecast/2.0"

        def log_message(self, fmt, *args):  # keep the console readable
            if os.environ.get("VERBOSE"):
                super().log_message(fmt, *args)

        def _send(self, payload, status: int = 200, content_type: str = "application/json"):
            body = (
                payload.encode("utf-8")
                if isinstance(payload, str)
                else json.dumps(payload, default=str).encode("utf-8")
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _dispatch(self, path: str, params: dict) -> Optional[tuple]:
            if path == "/healthz":
                return service.health(), 200
            if path == "/api/datasets":
                return service.datasets_payload(), 200
            if path == "/api/models":
                return service.models(), 200
            if path == "/api/forecast":
                return service.forecast(params), 200
            if path == "/api/screen":
                return service.screen(params), 200
            if path == "/api/recommend":
                return service.recommend(params), 200
            if path == "/api/backtest":
                return service.backtest(params), 200
            if path == "/api/stats":
                return service.stats(params), 200
            return None

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            params = parse_qs(parsed.query)
            try:
                if path in ("/", "/index.html", "/dashboard"):
                    return self._send(DASHBOARD_HTML, 200, "text/html")
                if path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                    return
                routed = self._dispatch(path, params)
                if routed is None:
                    return self._send({"error": "not found", "path": path}, 404)
                payload, status = routed
                return self._send(payload, status)
            except DataError as exc:
                return self._send({"error": str(exc)}, 404)
            except KeyError as exc:
                return self._send({"error": str(exc)}, 400)
            except ValueError as exc:
                return self._send({"error": str(exc)}, 400)
            except Exception as exc:  # pragma: no cover - never leak a stack trace
                return self._send({"error": f"internal error: {exc}"}, 500)

        def do_POST(self):
            return self._send({"error": "read-only API; use GET"}, 405)

    return Handler


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crypto Forecasting</title>
<style>
 body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:0;
      background:#0f1216;color:#e6e9ef}
 nav{padding:.85rem 1.5rem;background:#151a21;border-bottom:1px solid #222a34;font-weight:700}
 main{max-width:1050px;margin:0 auto;padding:1.5rem}
 h1{font-size:1.4rem;margin:0 0 .3rem} h2{font-size:1.05rem;color:#9fb3c8;margin:1.8rem 0 .6rem}
 .sub{color:#8b98a9;font-size:.85rem;margin-bottom:1rem}
 .controls{display:flex;gap:.6rem;flex-wrap:wrap;background:#151a21;border:1px solid #222a34;
           border-radius:10px;padding:.9rem}
 select,input,button{font-size:.9rem;padding:.5rem .7rem;border-radius:7px;border:1px solid #2c3644;
                     background:#1b222c;color:#e6e9ef}
 button{cursor:pointer;border:none;font-weight:600;background:#2563eb;color:#fff}
 table{width:100%;border-collapse:collapse;font-size:.85rem;background:#151a21;
       border:1px solid #222a34;border-radius:10px;overflow:hidden;margin-top:.5rem}
 th,td{padding:.55rem .7rem;text-align:right;border-bottom:1px solid #222a34}
 th:first-child,td:first-child{text-align:left}
 th{background:#1b222c;color:#9fb3c8}
 .sig-strong_buy{color:#4ade80;font-weight:700}.sig-buy{color:#86efac}.sig-hold{color:#cbd5e1}
 .sig-sell{color:#fdba74}.sig-strong_sell{color:#f87171;font-weight:700}
 .note{margin-top:1.5rem;padding:.9rem;background:#171d26;border-left:3px solid #3b82f6;
       border-radius:6px;font-size:.82rem;color:#a8b6c7}
</style></head><body>
<nav>Crypto Forecasting</nav>
<main>
 <h1>Forecast &amp; Screening</h1>
 <p class="sub">Served by the standard-library server - no Flask required.</p>
 <div class="controls">
   <select id="coin"></select>
   <select id="model"></select>
   <input id="capital" type="number" value="1000" style="width:6.5rem">
   <button id="go">Run</button>
 </div>
 <h2>Forecast</h2>
 <table id="forecast"><tbody><tr><td>loading…</td></tr></tbody></table>
 <h2>Screening (ranked by edge / model error)</h2>
 <table id="screen"><thead><tr><th>Coin</th><th>Signal</th><th>Last</th><th>Next</th>
   <th>Exp. return</th><th>Conf.</th><th>Dir. acc.</th></tr></thead>
   <tbody><tr><td colspan="7">loading…</td></tr></tbody></table>
 <h2>Recommended allocations</h2>
 <table id="rec"><thead><tr><th>Coin</th><th>Signal</th><th>Exp. return</th><th>Allocate</th>
   <th>Weight</th></tr></thead><tbody><tr><td colspan="5">loading…</td></tr></tbody></table>
 <div class="note">Statistical projections from historical prices only. Not financial advice.</div>
</main>
<script>
const $=id=>document.getElementById(id);
const money=v=>v==null?"n/a":(v>=1000?"$"+v.toLocaleString(undefined,{maximumFractionDigits:2})
  :"$"+Number(v).toFixed(6).replace(/0+$/,"").replace(/\\.$/,""));
const pct=v=>v==null?"n/a":(v*100).toFixed(2)+"%";
const num=(v,d=3)=>v==null?"n/a":Number(v).toFixed(d);
async function j(u){const r=await fetch(u);const d=await r.json();if(d.error)throw new Error(d.error);return d;}
async function init(){
  const ds=await j("/api/datasets");
  ds.datasets.forEach(d=>$("coin").append(new Option(d.symbol,d.symbol)));
  const ms=await j("/api/models");
  ms.models.forEach(m=>$("model").append(new Option(m.name,m.name)));
  $("model").value="drift";
  run();
}
async function run(){
  const coin=$("coin").value, model=$("model").value, capital=$("capital").value;
  const f=await j(`/api/forecast?coin=${encodeURIComponent(coin)}&model=${encodeURIComponent(model)}`);
  $("forecast").innerHTML=`<tbody>
    <tr><td>Coin</td><td>${f.symbol}</td></tr>
    <tr><td>Last close</td><td>${money(f.last_price)}</td></tr>
    <tr><td>Predicted next</td><td>${money(f.predicted_next_price)}
        <span style="color:#8b98a9">(${money(f.predicted_low)} – ${money(f.predicted_high)})</span></td></tr>
    <tr><td>Expected return</td><td>${pct(f.expected_return)}</td></tr>
    <tr><td>Signal</td><td class="sig-${f.signal}">${f.signal.replace("_"," ")}</td></tr>
    <tr><td>Directional accuracy</td><td>${num(f.accuracy.directional_accuracy)}</td></tr>
    <tr><td>Annualised volatility</td><td>${num(f.risk.volatility_annual,2)}</td></tr></tbody>`;

  const s=await j(`/api/screen?model=${encodeURIComponent(model)}`);
  $("screen").querySelector("tbody").innerHTML=s.rows.map(r=>r.error
    ?`<tr><td>${r.symbol}</td><td colspan="6">skipped: ${r.error}</td></tr>`
    :`<tr><td>${r.symbol}</td><td class="sig-${r.signal}">${r.signal.replace("_"," ")}</td>
      <td>${money(r.last_price)}</td><td>${money(r.predicted_next_price)}</td>
      <td>${pct(r.expected_return)}</td><td>${num(r.confidence,2)}</td>
      <td>${num(r.directional_accuracy)}</td></tr>`).join("");

  const rec=await j(`/api/recommend?model=${encodeURIComponent(model)}&capital=${capital}`);
  $("rec").querySelector("tbody").innerHTML=rec.proposals.length?rec.proposals.map(r=>
    `<tr><td>${r.symbol}</td><td class="sig-${r.signal}">${r.signal.replace("_"," ")}</td>
     <td>${pct(r.expected_return)}</td><td>${money(r.allocation.cash)}</td>
     <td>${(r.allocation.weight*100).toFixed(1)}%</td></tr>`).join("")
    :`<tr><td colspan="5">nothing cleared the bar - staying flat is a valid call</td></tr>`;
}
$("go").addEventListener("click",run);
init().catch(e=>{document.body.insertAdjacentHTML("beforeend",
  `<p style="color:#f87171;padding:1rem">${e.message}</p>`);});
</script></body></html>
"""


def serve(csv_dir: str = DEFAULT_CSV_DIR, host: str = "127.0.0.1", port: int = 8501):
    service = ForecastService(csv_dir)
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    print(
        f"crypto-forecast serving {len(service.datasets)} coins on "
        f"http://{host}:{port}  (Ctrl-C to stop)"
    )
    return httpd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="crypto_prediction.server")
    parser.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8501")))
    args = parser.parse_args(argv)

    httpd = serve(args.csv_dir, args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
