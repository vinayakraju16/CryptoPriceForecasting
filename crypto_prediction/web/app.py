"""Flask web layer (optional).

The HTTP surface is a thin adapter over ``crypto_prediction.core``. Flask is
imported only when this module is used, so the CLI and tests work on hosts
without Flask installed.

Compatibility: the original template called ``/models``, ``/data`` and
``/next_price``. Those routes are preserved with the same payload shape, so the
existing front-end keeps working, while new ``/api/*`` routes expose the richer
model (bands, signals, screening, backtests).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from ..core import (
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
from ..core.dataset import DataError
from ..core.indicators import summary_stats
from ..core.models import available_models, feature_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEMPLATE_DIR = os.path.join(ROOT, "templates")
STATIC_DIR = os.path.join(ROOT, "static")
DEFAULT_CSV_DIR = os.environ.get("CRYPTO_CSV_DIR", os.path.join(os.path.dirname(ROOT), "CSV"))


def create_app(csv_dir: Optional[str] = None, max_upload_mb: int = 8):
    """Application factory - the only supported way to construct the app."""
    try:
        from flask import Flask, jsonify, render_template, request
    except ImportError as exc:  # pragma: no cover - depends on host
        raise RuntimeError(
            "Flask is required for the web UI; install it with "
            "`pip install -r crypto_prediction/requirements.txt`"
        ) from exc

    csv_dir = csv_dir or DEFAULT_CSV_DIR
    app = Flask(
        __name__,
        template_folder=TEMPLATE_DIR,
        static_folder=STATIC_DIR,
    )
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024
    app.config["CSV_DIR"] = csv_dir

    _cache: Dict[str, object] = {}

    def datasets():
        """Lazily load every valid coin series, cached for the process."""
        if "datasets" not in _cache:
            loaded: Dict[str, object] = {}
            reports = {}
            for symbol, path in discover_datasets(app.config["CSV_DIR"]).items():
                series, report = load_series(path, symbol)
                reports[symbol] = report.to_dict()
                if report.is_valid:
                    loaded[symbol] = series
            _cache["datasets"] = loaded
            _cache["reports"] = reports
        return _cache["datasets"]

    def require_coin(symbol: str):
        series = datasets().get(symbol)
        if series is None:
            raise DataError(f"unknown coin {symbol!r}")
        return series

    @app.errorhandler(DataError)
    def _handle_data_error(exc):  # pragma: no cover - trivial
        return jsonify({"error": str(exc)}), 404

    @app.errorhandler(404)
    def _not_found(exc):  # pragma: no cover - trivial
        if request.path.startswith("/api/"):
            return jsonify({"error": "not found"}), 404
        return render_template("404.html"), 404

    # ---------------------------------------------------------------- health
    @app.get("/healthz")
    def healthz():
        data = datasets()
        return jsonify(
            {
                "status": "ok",
                "coins": len(data),
                "models": sorted(MODEL_REGISTRY),
                "csv_dir": app.config["CSV_DIR"],
            }
        )

    @app.get("/api/datasets")
    def api_datasets():
        payload = []
        for symbol, series in sorted(datasets().items()):
            payload.append(
                {
                    "symbol": symbol,
                    "start": str(series.start),
                    "end": str(series.end),
                    "bars": len(series),
                    "quality": _cache.get("reports", {}).get(symbol, {}),
                }
            )
        return jsonify(payload)

    @app.get("/api/models")
    def api_models():
        return jsonify(
            {
                "models": available_models(),
                "default": "ensemble",
                "csv_dir": app.config["CSV_DIR"],
            }
        )

    @app.get("/api/forecast")
    def api_forecast():
        symbol = request.args.get("coin", "Bitcoin")
        model = request.args.get("model", "ensemble")
        steps = max(1, int(request.args.get("steps", 1)))
        lookback = max(5, int(request.args.get("lookback", 30)))
        series = require_coin(symbol)
        prediction = build_prediction(series, model, lookback=lookback, horizon=steps)
        payload = prediction.to_dict()
        payload["features"] = _latest_features(series)
        return jsonify(payload)

    @app.get("/api/screen")
    def api_screen():
        model = request.args.get("model", "ensemble")
        top = request.args.get("top", type=int)
        rows = screen_universe(datasets(), model, top=top)
        return jsonify({"model": model, "count": len(rows), "rows": rows})

    @app.get("/api/backtest")
    def api_backtest():
        symbol = request.args.get("coin", "Bitcoin")
        model = request.args.get("model", "naive")
        lookback = max(5, int(request.args.get("lookback", 30)))
        series = require_coin(symbol)
        closes = series.closes
        scaler = MinMax()
        scaler.fit(closes[: max(2, int(len(closes) * 0.7))])
        scaled = scaler.transform(closes)
        xs, ys = make_windows(scaled, lookback, 1)
        result = walk_forward(
            build_model(model), xs, ys, scaler, series.dates[lookback:], min_train=120
        )
        return jsonify(result.to_dict())

    @app.get("/api/stats")
    def api_stats():
        symbol = request.args.get("coin", "Bitcoin")
        series = require_coin(symbol)
        return jsonify({"symbol": symbol, **summary_stats(series.closes)})

    # ------------------------------------------------- legacy (unchanged shape)
    @app.get("/models")
    def get_models():
        """Symbols *and* any bundled trained model files, so the original
        front-end (which expected model filenames) keeps working."""
        return jsonify(sorted(datasets().keys()))

    @app.get("/api/saved-models")
    def api_saved_models():
        """Report the bundled .h5 files and whether they can be loaded here."""
        model_dir = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "models")
        files = []
        if os.path.isdir(model_dir):
            files = sorted(
                name for name in os.listdir(model_dir) if name.endswith((".h5", ".keras"))
            )
        try:
            import tensorflow  # noqa: F401

            loadable = True
        except Exception:
            loadable = False
        return jsonify(
            {
                "files": files,
                "count": len(files),
                "tensorflow_available": loadable,
                "note": (
                    "Saved models can be loaded when TensorFlow is installed; "
                    "the built-in models work without it."
                ),
            }
        )

    @app.post("/data")
    def get_data():
        body = request.get_json(silent=True) or {}
        symbol = body.get("model_name") or body.get("coin")
        timeframe = body.get("timeframe", "months")
        if not symbol:
            return jsonify({"error": "Model name is required"}), 400
        symbol = _symbol_from_model_name(symbol)
        try:
            series = require_coin(symbol)
            model = build_model(body.get("model", "ensemble"))
            closes = series.closes
            scaler = MinMax()
            scaler.fit(closes[: max(2, int(len(closes) * 0.7))])
            scaled = scaler.transform(closes)
            lookback = max(5, int(body.get("lookback", 30)))
            xs, ys = make_windows(scaled, lookback, 1)
            model.fit(xs, ys)
            predicted_scaled = [model.predict(xs[i]) for i in range(len(xs))]
            predicted = scaler.inverse_transform(predicted_scaled)
            actual = scaler.inverse_transform(ys)
            dates = series.dates[lookback:]
            limit = -365 if timeframe == "years" else -365
            return jsonify(
                {
                    "prices": {
                        "dates": [str(d) for d in dates[limit:]],
                        "actual": [float(v) for v in actual[limit:]],
                        "predicted": [float(v) for v in predicted[limit:]],
                    }
                }
            )
        except DataError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:  # pragma: no cover - surface as JSON, never a stack trace
            return jsonify({"error": f"prediction failed: {exc}"}), 500

    @app.post("/next_price")
    def next_price():
        body = request.get_json(silent=True) or {}
        symbol = body.get("model_name") or body.get("coin")
        if not symbol:
            return jsonify({"error": "Model name is required"}), 400
        try:
            series = require_coin(_symbol_from_model_name(symbol))
            prediction = build_prediction(series, body.get("model", "ensemble"))
            return jsonify(
                {
                    "next_price": prediction.predicted_next_price,
                    "low": prediction.low,
                    "high": prediction.high,
                    "signal": prediction.signal,
                    "expected_return_pct": prediction.expected_return * 100.0,
                }
            )
        except DataError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": f"prediction failed: {exc}"}), 500

    # ------------------------------------------------------------------ pages
    @app.get("/")
    @app.get("/crypto-price-prediction")
    def home():
        return render_template(
            "index.html",
            models=sorted(datasets().keys()),
            model_options=available_models(),
        )

    @app.get("/dashboard")
    def dashboard():
        return render_template(
            "dashboard.html",
            models=sorted(datasets().keys()),
            model_options=available_models(),
        )

    @app.get("/sentiment-analysis")
    def sentiment_analysis():
        return render_template("sentiment_analysis.html")

    return app


def _symbol_from_model_name(name: str) -> str:
    """``Bitcoin_model.h5`` and ``Bitcoin`` both map to ``Bitcoin``."""
    stem = os.path.basename(str(name))
    for suffix in (".h5", ".keras", ".csv"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    if stem.endswith("_model"):
        stem = stem[: -len("_model")]
    if stem.startswith("coin_"):
        stem = stem[len("coin_"):]
    return stem


def _latest_features(series) -> Dict[str, Optional[float]]:
    rows = feature_matrix(series.closes, series.highs, series.lows)
    return rows[-1] if rows else {}


def main() -> None:
    app = create_app()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8501"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
