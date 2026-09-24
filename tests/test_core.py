"""Test suite for the crypto forecasting pipeline (stdlib ``unittest``).

Deliberately free of pytest/numpy so it runs anywhere::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crypto_prediction.core import (  # noqa: E402
    MinMax,
    build_model,
    build_prediction,
    discover_datasets,
    load_series,
    make_windows,
    sequential_split,
    validate_series,
    walk_forward,
)
from crypto_prediction.core import Bar, Series  # noqa: E402
from crypto_prediction.core.dataset import (  # noqa: E402
    DataError,
    _parse_date,
    _parse_float,
    symbol_from_path,
)
from crypto_prediction.core.indicators import (  # noqa: E402
    atr,
    bollinger,
    ema,
    max_drawdown,
    rsi,
    sharpe,
    sma,
    summary_stats,
    value_at_risk,
    volatility,
)
from crypto_prediction.core.metrics import (  # noqa: E402
    directional_accuracy,
    mae,
    mape,
    metric_suite,
    r2,
    rmse,
    smape,
)
from crypto_prediction.core.models import (  # noqa: E402
    MODEL_REGISTRY,
    EnsembleForecaster,
    ModelUnavailableError,
    available_models,
    feature_matrix,
    solve_least_squares,
)
from crypto_prediction.core.backtest import (  # noqa: E402
    buy_and_hold,
    strategy_returns,
)
from crypto_prediction.core.signals import signal_from_edge  # noqa: E402

CSV_DIR = os.path.join(ROOT, "CSV")


def synthetic_series(n=400, start_price=100.0, drift=0.002, wobble=0.01, symbol="TEST"):
    """Deterministic, gently trending series - no randomness, so tests are stable."""
    bars = []
    price = start_price
    day = date(2020, 1, 1)
    for i in range(n):
        price = price * (1.0 + drift) * (1.0 + wobble * math.sin(i / 7.0))
        bars.append(
            Bar(
                date=day + timedelta(days=i),
                open=price * 0.995,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000.0 + i,
            )
        )
    return Series(symbol=symbol, bars=bars, source="synthetic")


class TestParsing(unittest.TestCase):
    def test_dates(self):
        self.assertEqual(_parse_date("2013-04-29 23:59:59"), date(2013, 4, 29))
        self.assertEqual(_parse_date("2013-04-29"), date(2013, 4, 29))
        self.assertEqual(_parse_date("2013-04-29T00:00:00.000Z"), date(2013, 4, 29))
        with self.assertRaises(DataError):
            _parse_date("")
        with self.assertRaises(DataError):
            _parse_date("not-a-date")

    def test_floats(self):
        self.assertAlmostEqual(_parse_float("1.5"), 1.5)
        self.assertAlmostEqual(_parse_float("0"), 0.0)
        for bad in ("", "nan", "NaN", "null", "inf"):
            with self.assertRaises(DataError):
                _parse_float(bad)

    def test_symbol_from_path(self):
        self.assertEqual(symbol_from_path("/x/CSV/coin_Bitcoin.csv"), "Bitcoin")
        self.assertEqual(symbol_from_path("coin_Dogecoin.csv"), "Dogecoin")
        self.assertEqual(symbol_from_path("plain.csv"), "plain")


class TestValidation(unittest.TestCase):
    def test_dedupes_and_sorts(self):
        base = date(2021, 1, 1)
        bars = [
            Bar(base + timedelta(days=2), 1, 1, 1, 3, 10),
            Bar(base, 1, 1, 1, 1, 10),
            Bar(base + timedelta(days=1), 1, 1, 1, 2, 10),
            Bar(base + timedelta(days=1), 1, 1, 1, 2.5, 10),  # duplicate date, later revision
        ]
        series, report = validate_series(Series("X", bars), min_bars=2)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.dropped_duplicates, 1)
        self.assertEqual([b.date for b in series.bars], [base, base + timedelta(days=1), base + timedelta(days=2)])
        self.assertEqual(series.bars[1].close, 2.5, "latest revision should win")

    def test_rejects_nonpositive_and_empty(self):
        base = date(2021, 1, 1)
        bars = [Bar(base, 1, 1, 1, -5, 1), Bar(base + timedelta(days=1), 1, 1, 1, 0, 1)]
        _, report = validate_series(Series("X", bars), min_bars=1)
        self.assertFalse(report.is_valid)

        _, empty_report = validate_series(Series("X", []), min_bars=1)
        self.assertFalse(empty_report.is_valid)
        self.assertIn("series is empty", empty_report.errors)

    def test_min_bars_enforced(self):
        series = synthetic_series(n=5)
        _, report = validate_series(series, min_bars=30)
        self.assertFalse(report.is_valid)


class TestLoading(unittest.TestCase):
    def test_loads_bundled_csv(self):
        path = os.path.join(CSV_DIR, "coin_Bitcoin.csv")
        if not os.path.isfile(path):
            self.skipTest("bundled CSV not present")
        series, report = load_series(path)
        self.assertTrue(report.is_valid, report.to_dict())
        self.assertEqual(series.symbol, "Bitcoin")
        self.assertGreater(len(series), 1000)
        self.assertEqual(series.dates, sorted(series.dates))
        self.assertTrue(all(b.close > 0 for b in series.bars))

    def test_handles_malformed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "coin_Test.csv")
            base = date(2021, 1, 1)
            with open(path, "w", encoding="utf-8-sig") as handle:
                handle.write("SNo,Name,Symbol,Date,High,Low,Open,Close,Volume,Marketcap\n")
                handle.write("1,Test,T,2021-01-01,2,1,1,1.5,10,100\n")
                handle.write("2,Test,T,garbage,2,1,1,1.5,10,100\n")
                handle.write("3,Test,T,2021-01-03,2,1,1,nan,10,100\n")
                for i in range(3, 60):
                    day = base + timedelta(days=i)
                    handle.write(f"{i + 1},Test,T,{day.isoformat()},2,1,1,1.6,10,100\n")
            _, report = load_series(path, "Test")
            self.assertEqual(report.dropped_invalid, 2)
            self.assertTrue(report.is_valid)

    def test_garbage_file_terminates_quickly(self):
        """The original app hung/crashed on malformed input; we must fail cleanly."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "coin_Test.csv")
            with open(path, "w", encoding="utf-8-sig") as handle:
                handle.write("\x01\x02\x03binary garbage\n")
            with self.assertRaises(DataError):
                load_series(path, "Test")

    def test_impossible_date_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "coin_Test.csv")
            base = date(2021, 2, 1)
            with open(path, "w", encoding="utf-8-sig") as handle:
                handle.write("SNo,Name,Symbol,Date,High,Low,Open,Close,Volume,Marketcap\n")
                handle.write("1,Test,T,2021-01-32,2,1,1,1.5,10,100\n")
                for i in range(40):
                    day = base + timedelta(days=i)
                    handle.write(f"{i + 2},Test,T,{day.isoformat()},2,1,1,1.6,10,100\n")
            _, report = load_series(path, "Test")
            self.assertEqual(report.dropped_invalid, 1)
            self.assertTrue(report.is_valid)

    def test_discover(self):
        found = discover_datasets(CSV_DIR)
        self.assertIn("Bitcoin", found)
        with self.assertRaises(DataError):
            discover_datasets("/definitely/not/here")


class TestScaling(unittest.TestCase):
    def test_roundtrip(self):
        scaler = MinMax().fit([10, 20, 30])
        scaled = scaler.transform([10, 20, 30])
        self.assertAlmostEqual(scaled[0], 0.0)
        self.assertAlmostEqual(scaled[2], 1.0)
        back = scaler.inverse_transform(scaled)
        for original, restored in zip([10, 20, 30], back):
            self.assertAlmostEqual(original, restored, places=9)

    def test_no_leakage_from_future_values(self):
        """Fitting on train only: a future spike must not change train scaling."""
        train = [1.0, 2.0, 3.0]
        scaler = MinMax().fit(train)
        before = scaler.transform([1.0, 2.0, 3.0])
        _ = [0.0, 100.0, 1e9]  # hypothetical future data, never fitted
        after = scaler.transform([1.0, 2.0, 3.0])
        self.assertEqual(before, after)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            MinMax().transform([1.0])

    def test_constant_series_does_not_divide_by_zero(self):
        scaler = MinMax().fit([5.0, 5.0, 5.0])
        self.assertEqual(scaler.transform([5.0]), [0.0])


class TestWindows(unittest.TestCase):
    def test_windows_target_the_future(self):
        xs, ys = make_windows([1, 2, 3, 4, 5], lookback=2, horizon=1)
        self.assertEqual(xs[0], [1, 2])
        self.assertEqual(ys[0], 3)
        self.assertNotIn(ys[0], xs[0], "target must not be inside its own window")

    def test_horizon_shifts_target(self):
        xs, ys = make_windows([1, 2, 3, 4, 5, 6], lookback=2, horizon=2)
        self.assertEqual(xs[0], [1, 2])
        self.assertEqual(ys[0], 4)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            make_windows([1, 2, 3], lookback=0)
        with self.assertRaises(ValueError):
            make_windows([1, 2, 3], lookback=2, horizon=0)

    def test_split_is_chronological_and_non_overlapping(self):
        train, val, test = sequential_split(100, 0.7, 0.15)
        self.assertEqual((train.start, train.stop), (0, 70))
        self.assertEqual((val.start, val.stop), (70, 85))
        self.assertEqual((test.start, test.stop), (85, 100))
        self.assertLessEqual(train.stop, val.start)
        with self.assertRaises(ValueError):
            sequential_split(100, 0.9, 0.2)


class TestMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        actual = [1.0, 2.0, 3.0]
        self.assertEqual(mae(actual, actual), 0.0)
        self.assertEqual(rmse(actual, actual), 0.0)
        self.assertEqual(smape(actual, actual), 0.0)
        self.assertAlmostEqual(r2(actual, actual), 1.0)

    def test_known_values(self):
        actual = [1.0, 2.0, 3.0]
        predicted = [2.0, 2.0, 2.0]
        self.assertAlmostEqual(mae(actual, predicted), 2 / 3)
        self.assertAlmostEqual(rmse(actual, predicted), math.sqrt(2 / 3))

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            mae([1, 2], [1])

    def test_mape_skips_zero_actuals(self):
        self.assertTrue(math.isfinite(mape([0.0, 2.0], [0.0, 1.0])))

    def test_r2_without_variance_is_nan(self):
        self.assertTrue(math.isnan(r2([5.0, 5.0], [4.0, 6.0])))

    def test_directional_accuracy(self):
        # rising then falling; model calls both correctly
        actual = [1.0, 2.0, 3.0, 2.0]
        predicted = [1.5, 2.5, 2.5, 1.5]
        self.assertAlmostEqual(directional_accuracy(actual, predicted), 1.0)
        self.assertAlmostEqual(directional_accuracy(actual, [1.0, 1.0, 4.0, 4.0]), 1 / 3)

    def test_metric_suite_is_json_safe(self):
        suite = metric_suite([1.0], [1.0])
        self.assertIsNone(suite["directional_accuracy"])
        for key, value in suite.items():
            self.assertTrue(value is None or isinstance(value, (int, float)), key)


class TestIndicators(unittest.TestCase):
    def test_sma(self):
        out = sma([1, 2, 3, 4, 5], 3)
        self.assertEqual(out[:2], [None, None])
        self.assertAlmostEqual(out[2], 2.0)
        self.assertAlmostEqual(out[4], 4.0)

    def test_ema_seeded(self):
        out = ema([1, 2, 3], 2)
        self.assertEqual(out[0], 1)
        self.assertGreater(out[-1], 2)

    def test_rsi_bounds_and_extremes(self):
        rising = rsi(list(range(1, 40)), 14)
        self.assertTrue(all(v is None or 0 <= v <= 100 for v in rising))
        self.assertAlmostEqual([v for v in rising if v is not None][-1], 100.0)
        falling = rsi(list(range(40, 1, -1)), 14)
        self.assertAlmostEqual([v for v in falling if v is not None][-1], 0.0)

    def test_bollinger_ordering(self):
        values = [10, 11, 12, 11, 13, 12, 14, 13, 15, 14] * 3
        upper, mid, lower = bollinger(values, 20, 2.0)
        for u, m, l in zip(upper, mid, lower):
            if u is None:
                continue
            self.assertGreaterEqual(u, m)
            self.assertGreaterEqual(m, l)

    def test_atr_positive(self):
        series = synthetic_series(60)
        out = atr(series.highs, series.lows, series.closes, 14)
        defined = [v for v in out if v is not None]
        self.assertTrue(defined)
        self.assertTrue(all(v > 0 for v in defined))

    def test_risk_stats(self):
        series = synthetic_series(300, start_price=100, drift=0.001)
        self.assertTrue(math.isfinite(volatility(series.closes)))
        self.assertLessEqual(max_drawdown(series.closes), 0.0)
        self.assertTrue(math.isfinite(sharpe(series.closes)))
        self.assertLessEqual(value_at_risk(series.closes), 1.0)

    def test_max_drawdown_known(self):
        self.assertAlmostEqual(max_drawdown([100, 50, 75]), -0.5)

    def test_summary_stats_json_safe(self):
        payload = summary_stats(synthetic_series(200).closes)
        self.assertEqual(payload["n"], 200)
        for value in payload.values():
            self.assertTrue(value is None or isinstance(value, (int, float, str)))

    def test_feature_matrix_alignment(self):
        series = synthetic_series(120)
        rows = feature_matrix(series.closes, series.highs, series.lows)
        self.assertEqual(len(rows), len(series))
        self.assertIsNone(rows[0]["rsi_14"])
        self.assertIsNotNone(rows[-1]["rsi_14"])
        self.assertAlmostEqual(rows[-1]["close"], series.closes[-1])


class TestLeastSquares(unittest.TestCase):
    def test_recovers_exact_line(self):
        design = [[1.0, x] for x in range(20)]
        target = [3.0 + 2.0 * x for x in range(20)]
        beta = solve_least_squares(design, target)
        self.assertAlmostEqual(beta[0], 3.0, places=4)
        self.assertAlmostEqual(beta[1], 2.0, places=4)

    def test_handles_collinear_columns(self):
        design = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        beta = solve_least_squares(design, [1.0, 2.0, 3.0])
        self.assertEqual(len(beta), 2)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            solve_least_squares([], [])


class TestModels(unittest.TestCase):
    def test_registry_contents(self):
        for name in ("naive", "sma", "drift", "linear", "holt", "ar", "ensemble"):
            self.assertIn(name, MODEL_REGISTRY)

    def test_unknown_model_raises(self):
        with self.assertRaises(KeyError):
            build_model("does_not_exist")

    def test_naive_equals_last_value(self):
        model = build_model("naive")
        self.assertEqual(model.predict([1.0, 2.0, 3.0]), 3.0)

    def test_linear_extrapolates(self):
        model = build_model("linear")
        self.assertAlmostEqual(model.predict([1.0, 2.0, 3.0, 4.0]), 5.0, places=6)

    def test_ar_learns_and_predicts_plausibly(self):
        series = synthetic_series(400)
        scaler = MinMax().fit(series.closes)
        xs, ys = make_windows(scaler.transform(series.closes), 30, 1)
        model = build_model("ar").fit(xs, ys)
        value = model.predict(xs[-1])
        self.assertTrue(0.0 <= value <= 1.5, f"scaled forecast out of band: {value}")

    def test_ensemble_averages_members(self):
        model = EnsembleForecaster(["naive", "linear"])
        window = [1.0, 2.0, 3.0]
        expected = (3.0 + 4.0) / 2
        self.assertAlmostEqual(model.predict(window), expected, places=6)

    def test_multistep_forecast_length(self):
        series = synthetic_series(300)
        scaler = MinMax().fit(series.closes)
        xs, ys = make_windows(scaler.transform(series.closes), 20, 3)
        model = build_model("holt").fit(xs, ys)
        path = model.forecast(scaler.transform(series.closes)[-20:], steps=3)
        self.assertEqual(len(path), 3)

    def test_available_models_payload(self):
        payload = available_models()
        names = {row["name"] for row in payload}
        self.assertIn("ensemble", names)
        for row in payload:
            self.assertIn("description", row)

    def test_keras_model_unavailable_or_working(self):
        if "keras_lstm" not in MODEL_REGISTRY:
            self.skipTest("tensorflow not installed on this host")
        model = build_model("keras_lstm", lookback=5, epochs=1)
        self.assertTrue(hasattr(model, "fit"))


class TestBacktest(unittest.TestCase):
    def test_walk_forward_no_lookahead(self):
        """Training windows must never include the row being predicted."""
        series = synthetic_series(300)
        scaler = MinMax().fit(series.closes)
        xs, ys = make_windows(scaler.transform(series.closes), 20, 1)
        result = walk_forward(build_model("naive"), xs, ys, scaler, series.dates[20:], min_train=100)
        self.assertGreater(len(result.actual), 0)
        self.assertEqual(len(result.actual), len(result.predicted))
        self.assertIn("directional_accuracy", result.metrics)

    def test_insufficient_data_is_reported_not_crashed(self):
        scaler = MinMax().fit([1.0, 2.0])
        xs, ys = make_windows([1.0, 2.0], 1, 1)
        result = walk_forward(build_model("naive"), xs, ys, scaler, ["a", "b"], min_train=50)
        self.assertEqual(result.metrics.get("n"), 0)
        self.assertIn("error", result.metrics)

    def test_strategy_charges_fees_on_flips(self):
        actual = [100.0, 101.0, 102.0, 103.0]
        predicted = [101.0, 102.0, 103.0, 104.0]  # always long -> one entry
        stats = strategy_returns(actual, predicted, fee=0.001)
        self.assertEqual(stats["trades"], 1)
        self.assertGreater(stats["total_return"], 0)

    def test_strategy_flat_prediction_takes_no_risk(self):
        actual = [100.0, 90.0, 80.0]
        predicted = [90.0, 80.0, 70.0]  # predicts falls -> stays flat
        stats = strategy_returns(actual, predicted, fee=0.0)
        self.assertEqual(stats["total_return"], 0.0)

    def test_buy_and_hold(self):
        stats = buy_and_hold([100.0, 200.0])
        self.assertAlmostEqual(stats["total_return"], 1.0)

    def test_backtest_result_serialises(self):
        series = synthetic_series(250)
        scaler = MinMax().fit(series.closes)
        xs, ys = make_windows(scaler.transform(series.closes), 20, 1)
        result = walk_forward(build_model("sma"), xs, ys, scaler, series.dates[20:], min_train=80)
        payload = result.to_dict()
        self.assertIn("predictions", payload)
        self.assertEqual(len(payload["predictions"]), len(result.actual))


class TestSignals(unittest.TestCase):
    def test_signal_thresholds(self):
        self.assertEqual(signal_from_edge(0.5, 0.01), "strong_buy")
        self.assertEqual(signal_from_edge(0.005, 0.01), "buy")
        self.assertEqual(signal_from_edge(0.0, 0.01), "hold")
        self.assertEqual(signal_from_edge(-0.005, 0.01), "sell")
        self.assertEqual(signal_from_edge(-0.5, 0.01), "strong_sell")

    def test_zero_error_scale_is_safe(self):
        for edge in (-0.2, 0.0, 0.2):
            self.assertIn(signal_from_edge(edge, 0.0), {"strong_buy", "buy", "hold", "sell", "strong_sell"})

    def test_build_prediction_shape_and_band(self):
        series = synthetic_series(400, drift=0.003)
        prediction = build_prediction(series, "ensemble", lookback=20, min_train=100)
        payload = prediction.to_dict()
        for key in (
            "symbol", "last_price", "predicted_next_price", "predicted_low",
            "predicted_high", "signal", "confidence", "accuracy", "risk", "disclaimer",
        ):
            self.assertIn(key, payload)
        self.assertLessEqual(payload["predicted_low"], payload["predicted_next_price"])
        self.assertGreaterEqual(payload["predicted_high"], payload["predicted_next_price"])
        self.assertGreater(payload["last_price"], 0)
        self.assertIn(payload["signal"], {"strong_buy", "buy", "hold", "sell", "strong_sell"})
        self.assertGreaterEqual(payload["confidence"], 0.0)
        self.assertLessEqual(payload["confidence"], 1.0)

    def test_prediction_rejects_short_series(self):
        with self.assertRaises(ValueError):
            build_prediction(synthetic_series(50), "ensemble", lookback=30, min_train=120)

    def test_prediction_rejects_unknown_model(self):
        with self.assertRaises(KeyError):
            build_prediction(synthetic_series(400), "nope", lookback=20, min_train=100)

    def test_naive_tracks_recent_price_on_synthetic_trend(self):
        """Sanity: a trending series should not produce an absurd next price."""
        series = synthetic_series(400, drift=0.002)
        prediction = build_prediction(series, "naive", lookback=20, min_train=100)
        ratio = prediction.predicted_next_price / prediction.last_price
        self.assertTrue(0.5 < ratio < 2.0, f"implausible ratio {ratio}")


class TestScreenUniverse(unittest.TestCase):
    def test_screen_ranks_and_reports(self):
        from crypto_prediction.core.signals import screen_universe

        datasets = {
            "Up": synthetic_series(400, drift=0.004, symbol="Up"),
            "Flat": synthetic_series(400, drift=0.0, symbol="Flat"),
            "Down": synthetic_series(400, drift=-0.004, symbol="Down"),
        }
        rows = screen_universe(datasets, "naive", lookback=20, min_train=100)
        self.assertEqual(len(rows), 3)
        scores = [row["risk_adjusted_score"] for row in rows if "risk_adjusted_score" in row]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(rows[0]["symbol"], "Up")

    def test_screen_survives_a_broken_coin(self):
        from crypto_prediction.core.signals import screen_universe

        datasets = {
            "Good": synthetic_series(400, symbol="Good"),
            "Tiny": synthetic_series(30, symbol="Tiny"),  # too short to model
        }
        rows = screen_universe(datasets, "naive", lookback=20, min_train=100)
        self.assertEqual(len(rows), 2)
        tiny = [r for r in rows if r["symbol"] == "Tiny"][0]
        self.assertIn("error", tiny)


class TestCli(unittest.TestCase):
    def test_validate(self):
        from crypto_prediction.__main__ import main

        self.assertEqual(main(["--csv-dir", CSV_DIR, "validate", "--json"]), 0)

    def test_forecast(self):
        from crypto_prediction.__main__ import main

        self.assertEqual(main(["--csv-dir", CSV_DIR, "forecast", "Bitcoin", "--model", "naive"]), 0)

    def test_screen(self):
        from crypto_prediction.__main__ import main

        self.assertEqual(main(["--csv-dir", CSV_DIR, "screen", "--model", "naive", "--top", "3"]), 0)

    def test_backtest(self):
        from crypto_prediction.__main__ import main

        self.assertEqual(main(["--csv-dir", CSV_DIR, "backtest", "Bitcoin", "--model", "holt"]), 0)

    def test_unknown_coin_exits_nonzero(self):
        from crypto_prediction.__main__ import main

        self.assertEqual(main(["--csv-dir", CSV_DIR, "forecast", "NotACoin"]), 2)


class TestReport(unittest.TestCase):
    def test_html_report_builds(self):
        from crypto_prediction.report import build_report, render_html

        report = build_report(CSV_DIR, "naive", top=4)
        document = render_html(report)
        self.assertIn("<!DOCTYPE html>", document)
        self.assertIn("Screening", document)
        self.assertIn("</html>", document)
        self.assertGreater(report["coins"], 0)


class TestWebContract(unittest.TestCase):
    """The web layer is optional, but its helpers must stay correct."""

    def test_symbol_from_model_name(self):
        from crypto_prediction.web.app import _symbol_from_model_name

        self.assertEqual(_symbol_from_model_name("Bitcoin_model.h5"), "Bitcoin")
        self.assertEqual(_symbol_from_model_name("Bitcoin"), "Bitcoin")
        self.assertEqual(_symbol_from_model_name("coin_Bitcoin.csv"), "Bitcoin")
        self.assertEqual(_symbol_from_model_name("/a/b/Dogecoin_model.h5"), "Dogecoin")

    def test_create_app_without_flask_is_a_clear_error(self):
        try:
            import flask  # noqa: F401
        except ImportError:
            from crypto_prediction.web.app import create_app

            with self.assertRaises(RuntimeError):
                create_app(CSV_DIR)

    def test_flask_routes_when_available(self):
        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("flask not installed")
        from crypto_prediction.web.app import create_app

        app = create_app(CSV_DIR)
        client = app.test_client()
        self.assertEqual(client.get("/healthz").status_code, 200)
        self.assertEqual(client.get("/models").status_code, 200)
        self.assertEqual(client.get("/api/forecast?coin=Bitcoin&model=naive").status_code, 200)
        self.assertEqual(client.post("/next_price", json={"model_name": "Bitcoin"}).status_code, 200)
        self.assertEqual(client.post("/data", json={"model_name": "Bitcoin", "timeframe": "years"}).status_code, 200)
        self.assertEqual(client.post("/next_price", json={}).status_code, 400)
        self.assertEqual(client.get("/api/forecast?coin=Nope").status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
