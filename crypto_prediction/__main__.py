"""Command-line interface: forecast, screen, backtest and validate.

Examples::

    python -m crypto_prediction forecast Bitcoin --model ensemble --steps 7
    python -m crypto_prediction screen --top 5 --model ensemble
    python -m crypto_prediction backtest Bitcoin --model ar
    python -m crypto_prediction validate
    python -m crypto_prediction compare Bitcoin

The CLI is the reproducible, reviewable surface of the project: a reviewer can
clone the repo, run one command and see real numbers without opening a notebook
or starting a web server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from .core import (
    MODEL_REGISTRY,
    build_model,
    build_prediction,
    discover_datasets,
    load_series,
    screen_universe,
    walk_forward,
    make_windows,
    MinMax,
)
from .core.dataset import DataError
from .core.indicators import summary_stats

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)
DEFAULT_CSV_DIR = os.path.join(DEFAULT_ROOT, "CSV")


def _load_all(csv_dir: str) -> Dict[str, object]:
    datasets: Dict[str, object] = {}
    for symbol, path in discover_datasets(csv_dir).items():
        series, report = load_series(path, symbol)
        if report.is_valid:
            datasets[symbol] = series
    return datasets


def _load_one(csv_dir: str, symbol: str):
    path = discover_datasets(csv_dir).get(symbol)
    if path is None:
        raise DataError(f"unknown coin {symbol!r}; run the validate command to list available coins")
    series, report = load_series(path, symbol)
    if not report.is_valid:
        raise DataError(f"{symbol}: {'; '.join(report.errors)}")
    return series


def _emit(payload: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return
    for key, value in payload.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        elif isinstance(value, list):
            print(f"{key}: {len(value)} item(s)")
        else:
            print(f"{key}: {value}")


def cmd_report(args) -> int:
    from .report import build_report, render_html

    report = build_report(args.csv_dir, args.model, getattr(args, "top", None))
    document = render_html(report)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(document)
    print(f"wrote {out_path} ({len(document):,} bytes, {report['coins']} coins)")
    if getattr(args, "json_out", None):
        json_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"wrote {json_path}")
    return 0


def cmd_live(args) -> int:
    """Fetch recent bars from a public provider and forecast on the merged series."""
    from .core.data_source import PROVIDERS, fetch_and_merge

    if args.provider not in PROVIDERS:
        print(f"error: unknown provider {args.provider!r}; choose from {sorted(PROVIDERS)}", file=sys.stderr)
        return 2

    base = None
    if not args.no_cache:
        try:
            base = _load_one(args.csv_dir, args.coin)
        except DataError:
            base = None
            print(f"note: no bundled history for {args.coin}; using fetched data only", file=sys.stderr)

    try:
        result = fetch_and_merge(
            base, args.coin, provider=args.provider, limit=args.limit
        )
    except DataError as exc:
        print(f"error: could not fetch live data: {exc}", file=sys.stderr)
        print(
            "hint: this host's egress may be restricted; run it where the provider is reachable, "
            "or use the bundled history with `forecast`.",
            file=sys.stderr,
        )
        return 3

    prediction = build_prediction(
        result.series, args.model, lookback=args.lookback, min_train=args.min_train
    ).to_dict()
    payload = {
        "fetch": result.to_dict(),
        "forecast": prediction,
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    fetch_meta = payload["fetch"]
    print(
        f"fetched {fetch_meta['fetched']} bar(s) from {fetch_meta['provider']} "
        f"-> {fetch_meta['bars']} total, {fetch_meta['start']} .. {fetch_meta['end']}"
    )
    _emit(prediction, False)
    return 0


def cmd_validate(args) -> int:
    datasets = discover_datasets(args.csv_dir)
    report: Dict[str, dict] = {}
    for symbol, path in datasets.items():
        _, validation = load_series(path, symbol, strict=False)
        report[symbol] = validation.to_dict()
    bad = [s for s, r in report.items() if not r["is_valid"]]
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{len(report)} dataset(s) found, {len(report) - len(bad)} valid, {len(bad)} rejected")
        for symbol, row in report.items():
            status = "OK " if row["is_valid"] else "BAD"
            print(
                f"  [{status}] {symbol:16s} rows={row['bars_out']:5d} "
                f"invalid={row['dropped_invalid']:3d} dupes={row['dropped_duplicates']:3d}"
            )
            for warning in row["warnings"][:2]:
                print(f"        ! {warning}")
            for error in row["errors"]:
                print(f"        x {error}")
    return 1 if bad else 0


def cmd_forecast(args) -> int:
    series = _load_one(args.csv_dir, args.coin)
    prediction = build_prediction(
        series, args.model, lookback=args.lookback, min_train=args.min_train, horizon=args.steps
    )
    payload = prediction.to_dict()
    if args.steps > 1:
        scaler = MinMax()
        closes = series.closes
        scaler.fit(closes[: max(2, int(len(closes) * 0.7))])
        scaled = scaler.transform(closes)
        xs, ys = make_windows(scaled, args.lookback, args.steps)
        model = build_model(args.model)
        model.fit(xs, ys)
        path = scaler.inverse_transform(model.forecast(scaled[-args.lookback:], steps=args.steps))
        payload["forecast_path"] = [round(float(p), 4) for p in path]
    _emit(payload, args.json)
    return 0


def cmd_screen(args) -> int:
    datasets = _load_all(args.csv_dir)
    rows = screen_universe(
        datasets, args.model, lookback=args.lookback, min_train=args.min_train, top=args.top
    )
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0
    print(f"{'symbol':16s} {'signal':11s} {'exp_ret%':>9s} {'conf':>5s} {'dir_acc':>8s} {'vol_ann':>8s}")
    for row in rows:
        if "error" in row:
            print(f"{row['symbol']:16s} skipped: {row['error']}")
            continue
        direction = row["directional_accuracy"]
        direction_text = f"{direction:.3f}" if isinstance(direction, float) else "n/a"
        volatility = row["volatility_annual"]
        volatility_text = f"{volatility:.2f}" if isinstance(volatility, float) else "n/a"
        print(
            f"{row['symbol']:16s} {row['signal']:11s} "
            f"{row['expected_return_pct']:9.2f} {row['confidence']:5.2f} "
            f"{direction_text:>8s} {volatility_text:>8s}"
        )
    return 0


def cmd_backtest(args) -> int:
    series = _load_one(args.csv_dir, args.coin)
    closes = series.closes
    scaler = MinMax()
    scaler.fit(closes[: max(2, int(len(closes) * 0.7))])
    scaled = scaler.transform(closes)
    xs, ys = make_windows(scaled, args.lookback, 1)
    model = build_model(args.model)
    result = walk_forward(
        model, xs, ys, scaler, series.dates[args.lookback:], min_train=args.min_train
    )
    payload = {"symbol": series.symbol, "model": args.model, "metrics": result.metrics, "strategy": result.strategy}
    _emit(payload, args.json)
    return 0


def cmd_compare(args) -> int:
    from .core.models import auto_select

    series = _load_one(args.csv_dir, args.coin)
    best, results = auto_select(series, lookback=args.lookback, min_train=args.min_train)
    if args.json:
        print(json.dumps({name: r.metrics for name, r in results.items()}, indent=2, default=str))
        return 0
    print(f"{args.coin}: {len(results)} model(s) evaluated, best by direction then sMAPE = {best}")
    print(f"{'model':10s} {'dir_acc':>8s} {'smape':>8s} {'mae':>12s} {'strategy_ret':>13s}")
    for name, result in sorted(
        results.items(),
        key=lambda kv: -(kv[1].metrics.get("directional_accuracy") or 0),
    ):
        metrics = result.metrics
        direction = metrics.get("directional_accuracy")
        smape_val = metrics.get("smape")
        mae_val = metrics.get("mae")
        strategy = result.strategy.get("total_return")
        print(
            f"{name:10s} "
            f"{(f'{direction:.3f}' if isinstance(direction, float) else 'n/a'):>8s} "
            f"{(f'{smape_val:.2f}' if isinstance(smape_val, float) else 'n/a'):>8s} "
            f"{(f'{mae_val:.2f}' if isinstance(mae_val, float) else 'n/a'):>12s} "
            f"{(f'{strategy:+.2%}' if isinstance(strategy, float) else 'n/a'):>13s}"
        )
    return 0


def cmd_stats(args) -> int:
    series = _load_one(args.csv_dir, args.coin)
    payload = {"symbol": series.symbol, **summary_stats(series.closes)}
    _emit(payload, args.json)
    return 0


def cmd_recommend(args) -> int:
    """Answer: what is worth buying, and how much?"""
    from .core.signals import recommend as build_recommendations

    datasets = _load_all(args.csv_dir)
    proposals = build_recommendations(
        datasets,
        args.model,
        lookback=args.lookback,
        min_train=args.min_train,
        capital=args.capital,
        risk_per_trade=args.risk,
        min_confidence=args.min_confidence,
        top=args.top,
    )
    if args.json:
        print(json.dumps(proposals, indent=2, default=str))
        return 0

    if not proposals:
        print(
            "No coins cleared the bar (positive expected return and enough of a directional "
            "edge). That is a valid answer - staying flat beats trading noise."
        )
        return 0

    print(f"model={args.model}  capital={args.capital:,.2f}  risk/trade={args.risk:.1%}")
    print(f"{'coin':16s} {'signal':11s} {'exp_ret%':>9s} {'allocate':>10s} {'weight':>8s} {'risk':>9s}")
    for row in proposals:
        allocation = row["allocation"]
        print(
            f"{row['symbol']:16s} {row['signal']:11s} {row['expected_return_pct']:9.2f} "
            f"{allocation['cash']:10.2f} {allocation['weight']:8.1%} {allocation['risk_cash']:9.2f}"
        )
    print("\nProjections from historical prices only; not financial advice.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    global_opts = argparse.ArgumentParser(add_help=False)
    global_opts.add_argument(
        "--csv-dir",
        default=os.environ.get("CRYPTO_CSV_DIR", DEFAULT_CSV_DIR),
        help="directory holding the coin_*.csv files",
    )
    global_opts.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    parser = argparse.ArgumentParser(
        prog="crypto_prediction",
        description="Dependency-light crypto forecasting, backtesting and screening.",
        parents=[global_opts],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False, parents=[global_opts])
    common.add_argument("--model", default="ensemble", choices=sorted(MODEL_REGISTRY))
    common.add_argument("--lookback", type=int, default=30)
    common.add_argument("--min-train", type=int, default=120)

    p = sub.add_parser(
        "validate",
        help="check every dataset parses and is usable",
        parents=[global_opts],
    )
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("forecast", help="forecast the next price for one coin", parents=[common])
    p.add_argument("coin")
    p.add_argument("--steps", type=int, default=1)
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("screen", help="rank every coin by risk-adjusted expected return", parents=[common])
    p.add_argument("--top", type=int, default=None)
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("backtest", help="walk-forward backtest one coin", parents=[common])
    p.add_argument("coin")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("compare", help="benchmark every model on one coin", parents=[common])
    p.add_argument("coin")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser(
        "stats",
        help="descriptive and risk statistics for one coin",
        parents=[global_opts],
    )
    p.add_argument("coin")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser(
        "recommend",
        help="propose sized buys: which coin, how much",
        parents=[common],
    )
    p.add_argument("--capital", type=float, default=1000.0)
    p.add_argument("--risk", type=float, default=0.02, help="fraction of capital risked per trade")
    p.add_argument("--min-confidence", type=float, default=0.02)
    p.add_argument("--top", type=int, default=3)
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser(
        "report",
        help="write a self-contained HTML + JSON report",
        parents=[common],
    )
    p.add_argument("--out", default="reports/latest.html")
    p.add_argument("--json-out", default=None)
    p.add_argument("--top", type=int, default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser(
        "live",
        help="fetch recent bars from a public provider, then forecast",
        parents=[common],
    )
    p.add_argument("coin")
    p.add_argument("--provider", default="binance", choices=["binance", "coingecko"])
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--no-cache", action="store_true", help="ignore the bundled history")
    p.set_defaults(func=cmd_live)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
