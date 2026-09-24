"""Forecasters, all pure-Python except the optional Keras LSTM.

Every model implements the same tiny interface so the backtester, the API and
the CLI can swap models without caring what is underneath::

    model = build_model("holt")
    model.fit(xs, ys)          # xs: list of float windows, ys: next values
    value = model.predict(window)   # one-step forecast

``MODEL_REGISTRY`` is the public list of what is available in this build.
The ``keras_lstm`` entry is only offered when TensorFlow is importable, so a
missing heavy dependency degrades the menu instead of breaking the app.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence

from .indicators import atr, ema, macd, rsi, sma
from .timeseries import MinMax, make_windows

__all__ = [
    "Forecaster",
    "MODEL_REGISTRY",
    "ModelUnavailableError",
    "build_model",
    "available_models",
    "feature_matrix",
    "auto_select",
    "solve_least_squares",
    "load_saved_keras_model",
]


class ModelUnavailableError(RuntimeError):
    """Raised when a model's optional dependency is not installed."""


def solve_least_squares(design: Sequence[Sequence[float]], target: Sequence[float], ridge: float = 1e-8) -> List[float]:
    """Solve ``X b ~= y`` via normal equations with a small ridge term.

    Pure-Python Gaussian elimination: keeps coefficient fitting available
    without numpy, and the ridge term keeps the system solvable when columns
    are collinear (common with lagged price features).
    """
    n = len(design)
    if n == 0:
        raise ValueError("empty design matrix")
    p = len(design[0])
    if p == 0:
        raise ValueError("design matrix has no columns")
    if len(target) != n:
        raise ValueError("design/target length mismatch")

    # A = X^T X + ridge*I , b = X^T y
    a = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for row, y in zip(design, target):
        if len(row) != p:
            raise ValueError("ragged design matrix")
        for i in range(p):
            b[i] += row[i] * y
            for j in range(p):
                a[i][j] += row[i] * row[j]
    for i in range(p):
        a[i][i] += ridge

    # forward elimination with partial pivoting
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            continue
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            b[col], b[pivot] = b[pivot], b[col]
        for r in range(col + 1, p):
            factor = a[r][col] / a[col][col]
            if factor == 0:
                continue
            for c in range(col, p):
                a[r][c] -= factor * a[col][c]
            b[r] -= factor * b[col]

    # back substitution
    beta = [0.0] * p
    for i in range(p - 1, -1, -1):
        if abs(a[i][i]) < 1e-12:
            beta[i] = 0.0
            continue
        total = b[i] - sum(a[i][j] * beta[j] for j in range(i + 1, p))
        beta[i] = total / a[i][i]
    return beta


class Forecaster:
    """Base class: naive persistence, also the fallback for subclasses."""

    name = "naive"
    requires: Optional[str] = None

    def fit(self, xs: Sequence[Sequence[float]], ys: Sequence[float]) -> "Forecaster":
        return self

    def predict(self, window: Sequence[float]) -> float:
        if not window:
            raise ValueError("empty window")
        return float(window[-1])

    def forecast(self, window: Sequence[float], steps: int = 1) -> List[float]:
        """Recursive multi-step forecast, feeding each prediction back in."""
        history = [float(v) for v in window]
        outputs: List[float] = []
        for _ in range(max(1, steps)):
            nxt = self.predict(history)
            outputs.append(nxt)
            history = history[1:] + [nxt]
        return outputs


class MovingAverageForecaster(Forecaster):
    name = "sma"

    def __init__(self, window: int = 5) -> None:
        self.window = max(1, window)

    def predict(self, window: Sequence[float]) -> float:
        tail = [float(v) for v in window[-self.window:]]
        return sum(tail) / len(tail)


class DriftForecaster(Forecaster):
    """Last value plus the average per-step drift across the window."""

    name = "drift"

    def predict(self, window: Sequence[float]) -> float:
        vals = [float(v) for v in window]
        if len(vals) < 2:
            return vals[-1] if vals else 0.0
        drift = (vals[-1] - vals[0]) / (len(vals) - 1)
        return vals[-1] + drift


class LinearTrendForecaster(Forecaster):
    """Ordinary least squares on time index, extrapolated one step."""

    name = "linear"

    def predict(self, window: Sequence[float]) -> float:
        vals = [float(v) for v in window]
        n = len(vals)
        if n < 2:
            return vals[-1] if vals else 0.0
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(vals) / n
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom == 0:
            return vals[-1]
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, vals)) / denom
        intercept = mean_y - slope * mean_x
        return intercept + slope * n  # the next index


class HoltForecaster(Forecaster):
    """Holt's linear (double exponential) smoothing with a trend term."""

    name = "holt"

    def __init__(self, alpha: float = 0.4, beta: float = 0.1) -> None:
        self.alpha = alpha
        self.beta = beta

    def predict(self, window: Sequence[float]) -> float:
        vals = [float(v) for v in window]
        if not vals:
            return 0.0
        level = vals[0]
        trend = (vals[1] - vals[0]) if len(vals) > 1 else 0.0
        for v in vals[1:]:
            prev_level = level
            level = self.alpha * v + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
        return level + trend


class AutoregressiveForecaster(Forecaster):
    """AR(p) on log-returns: predict the next return, then reprice.

    Working in returns rather than raw levels is what keeps the model sane for
    assets that move by orders of magnitude over the sample (DOGE, BTC).
    """

    name = "ar"

    def __init__(self, order: int = 5) -> None:
        self.order = max(1, order)
        self.coefficients: Optional[List[float]] = None
        self.mean_return: float = 0.0

    @staticmethod
    def _returns(window: Sequence[float]) -> List[float]:
        vals = [float(v) for v in window]
        return [
            math.log(cur / prev)
            for prev, cur in zip(vals, vals[1:])
            if prev > 0 and cur > 0
        ]

    def fit(self, xs: Sequence[Sequence[float]], ys: Sequence[float]) -> "AutoregressiveForecaster":
        returns_series = [self._returns(w) for w in xs]
        design: List[List[float]] = []
        target: List[float] = []
        for rets in returns_series:
            if len(rets) < self.order + 1:
                continue
            design.append([1.0] + rets[-self.order:][::-1])
            target.append(rets[-1])
        if len(design) < self.order + 2:
            self.coefficients = None
            return self
        if target:
            self.mean_return = sum(target) / len(target)
        self.coefficients = solve_least_squares(design, target)
        return self

    def predict(self, window: Sequence[float]) -> float:
        vals = [float(v) for v in window]
        if not vals:
            return 0.0
        last = vals[-1]
        rets = self._returns(vals)
        if self.coefficients is None or len(rets) < self.order:
            return last * math.exp(self.mean_return)
        lags = rets[-self.order:][::-1]
        predicted_return = self.coefficients[0] + sum(
            c * lag for c, lag in zip(self.coefficients[1:], lags)
        )
        # clamp to a plausible daily band; guards against extrapolating a spike
        predicted_return = max(-0.5, min(0.5, predicted_return))
        return last * math.exp(predicted_return)


class EnsembleForecaster(Forecaster):
    """Average of the component forecasts - a cheap variance reducer."""

    name = "ensemble"

    def __init__(self, members: Optional[Sequence[str]] = None) -> None:
        names = list(members) if members else ["sma", "drift", "linear", "holt"]
        self.members: List[Forecaster] = [build_model(n) for n in names if n in MODEL_REGISTRY]
        if not self.members:
            self.members = [MovingAverageForecaster()]

    def fit(self, xs: Sequence[Sequence[float]], ys: Sequence[float]) -> "EnsembleForecaster":
        for member in self.members:
            member.fit(xs, ys)
        return self

    def predict(self, window: Sequence[float]) -> float:
        preds = [m.predict(window) for m in self.members]
        return sum(preds) / len(preds)


class KerasLSTMForecaster(Forecaster):
    """Optional LSTM. Only registered when TensorFlow/Keras import cleanly.

    The important difference from the original ``app.py``: the scaler is fitted
    on the training window only, and ``predict`` receives the *unscaled* window
    and normalises it with that training scaler.
    """

    name = "keras_lstm"
    requires = "tensorflow"

    def __init__(self, lookback: int = 30, units: int = 50, epochs: int = 20) -> None:
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.scaler = MinMax()
        self._model = None

    def _require(self):
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow.keras import layers, models
        except Exception as exc:  # pragma: no cover - depends on host
            raise ModelUnavailableError(
                "keras_lstm needs tensorflow; install it or pick a different model"
            ) from exc
        return tf, layers, models

    def fit(self, xs, ys) -> "KerasLSTMForecaster":
        tf, layers, models = self._require()
        tf.random.set_seed(42)
        train_values = [v for window in xs for v in window]
        self.scaler.fit(train_values)
        x_train = [[self.scaler.transform([v])[0] for v in window] for window in xs]
        y_train = [self.scaler.transform([y])[0] for y in ys]
        model = models.Sequential(
            [
                layers.Input(shape=(len(x_train[0]), 1)),
                layers.LSTM(self.units, return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(
            tf.constant(x_train, dtype=tf.float32),
            tf.constant(y_train, dtype=tf.float32),
            epochs=self.epochs,
            batch_size=32,
            verbose=0,
        )
        self._model = model
        return self

    def predict(self, window: Sequence[float]) -> float:
        if self._model is None:
            raise ModelUnavailableError("keras_lstm must be fitted before predict")
        row = [self.scaler.transform([v])[0] for v in window]
        import numpy as np  # local import: only needed when actually predicting

        pred = self._model.predict(np.array([row], dtype="float32"), verbose=0)
        return float(self.scaler.inverse_transform([float(pred[0][0])])[0])


MODEL_REGISTRY: Dict[str, type] = {
    "naive": Forecaster,
    "sma": MovingAverageForecaster,
    "drift": DriftForecaster,
    "linear": LinearTrendForecaster,
    "holt": HoltForecaster,
    "ar": AutoregressiveForecaster,
    "ensemble": EnsembleForecaster,
}


def load_saved_keras_model(path: str, train_values: Sequence[float]):
    """Load one of the bundled ``models/*.h5`` files with a leak-free scaler.

    The repository ships ten pre-trained LSTMs. Loading them directly is
    supported, but the scaler must be fitted on the *training* slice only -
    the original code re-fitted it over the whole series, which both leaked the
    test window and made the output drift between requests.

    Requires TensorFlow; raises ``ModelUnavailableError`` when it is absent.
    The loaded model is expected to accept a window shaped
    ``(1, sequence_length, 1)`` and to return a min-max scaled value.
    """
    try:
        from tensorflow.keras.models import load_model
    except Exception as exc:  # pragma: no cover - depends on host
        raise ModelUnavailableError(
            "loading bundled .h5 models needs tensorflow; "
            "install the 'ml' extra or use a built-in model"
        ) from exc

    if not os.path.isfile(path):
        raise FileNotFoundError(f"no saved model at {path}")

    keras_model = load_model(path)
    lookback = int(keras_model.input_shape[1])

    class _SavedKerasForecaster(Forecaster):
        name = "saved_keras"

        def __init__(self) -> None:
            self._model = keras_model
            self.lookback = lookback
            self.scaler = MinMax().fit(list(train_values))

        def fit(self, xs, ys):  # already trained; keep the fitted scaler
            return self

        def predict(self, window):
            import numpy as np

            row = [self.scaler.transform([float(v)])[0] for v in window[-self.lookback:]]
            out = self._model.predict(np.array([row], dtype="float32"), verbose=0)
            return float(self.scaler.inverse_transform([float(out[0][0])])[0])

    return _SavedKerasForecaster()


def _keras_available() -> bool:
    try:  # pragma: no cover - depends on host
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


if _keras_available():  # pragma: no cover - depends on host
    MODEL_REGISTRY["keras_lstm"] = KerasLSTMForecaster
    EnsembleForecaster  # bind so linters keep the symbol


def available_models() -> List[Dict[str, object]]:
    """Model catalogue for the API/UI, including which optional extras exist."""
    out: List[Dict[str, object]] = []
    for name, cls in sorted(MODEL_REGISTRY.items()):
        if name == "ensemble":
            members = ["sma", "drift", "linear", "holt"]
        elif name == "keras_lstm":
            members = []
        else:
            members = []
        out.append(
            {
                "name": name,
                "class": cls.__name__,
                "requires": getattr(cls, "requires", None),
                "members": members,
                "description": _DESCRIPTIONS.get(name, ""),
            }
        )
    return out


_DESCRIPTIONS = {
    "naive": "Persistence: tomorrow equals today. The baseline every model must beat.",
    "sma": "Simple moving average of the last k steps.",
    "drift": "Last value plus the average per-step drift across the window.",
    "linear": "OLS straight line on the window, extrapolated one step.",
    "holt": "Holt's linear exponential smoothing (level + trend).",
    "ar": "Autoregressive model on log-returns, repriced from the last close.",
    "ensemble": "Equal-weight average of sma, drift, linear and holt.",
    "keras_lstm": "Optional LSTM trained with a train-only scaler (no leakage).",
}


def build_model(name: str, **kwargs) -> Forecaster:
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def feature_matrix(
    closes: Sequence[float],
    highs: Optional[Sequence[float]] = None,
    lows: Optional[Sequence[float]] = None,
) -> List[Dict[str, Optional[float]]]:
    """Technical feature table aligned to ``closes`` (rows with ``None`` warm-ups kept).

    This is the multivariate view of the market used for screening and for
    export to the notebooks (XGBoost / LSTM experiments read this shape).
    """
    closes = [float(c) for c in closes]
    n = len(closes)
    rsi14 = rsi(closes, 14)
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_line, signal_line, hist = macd(closes)
    sma20 = sma(closes, 20)
    if highs is None or lows is None:
        atr14: List[Optional[float]] = [None] * n
    else:
        atr14 = atr(highs, lows, closes, 14)

    rows: List[Dict[str, Optional[float]]] = []
    for i in range(n):
        prev = closes[i - 1] if i > 0 else None
        rows.append(
            {
                "close": closes[i],
                "return_1d": (closes[i] / prev - 1.0) if prev else None,
                "rsi_14": rsi14[i],
                "ema_12": ema12[i],
                "ema_26": ema26[i],
                "macd": macd_line[i],
                "macd_signal": signal_line[i],
                "macd_hist": hist[i],
                "sma_20": sma20[i],
                "atr_14": atr14[i],
                "above_sma20": (
                    None if sma20[i] is None else float(closes[i] > sma20[i])
                ),
            }
        )
    return rows


def auto_select(
    series,
    *,
    lookback: int = 30,
    min_train: int = 120,
    candidates: Optional[Sequence[str]] = None,
) -> tuple:
    """Walk-forward backtest every candidate and return the best performer.

    Returns ``(best_name, results)`` where ``results`` maps model name to its
    ``BacktestResult``. Ranking is by directional accuracy, then sMAPE - a
    model that is accurate in price but blind to direction is useless for
    trading, so direction breaks the tie first.
    """
    from .backtest import walk_forward

    names = list(candidates) if candidates else [
        n for n in ("naive", "sma", "drift", "linear", "holt", "ar", "ensemble")
        if n in MODEL_REGISTRY
    ]
    scaler = MinMax()
    closes = series.closes
    scaler.fit(closes[: max(2, int(len(closes) * 0.7))])
    scaled = scaler.transform(closes)
    xs, ys = make_windows(scaled, lookback, 1)

    results: Dict[str, object] = {}
    for name in names:
        model = build_model(name)
        try:
            result = walk_forward(model, xs, ys, scaler, series.dates[lookback:], min_train=min_train)
        except Exception:  # pragma: no cover - a broken candidate must not kill the run
            continue
        results[name] = result

    if not results:
        return "naive", results

    def rank(item):
        name, result = item
        direction = result.metrics.get("directional_accuracy")
        error = result.metrics.get("smape")
        return (
            -(direction if direction is not None else 0.0),
            error if error is not None else float("inf"),
            name,
        )

    best_name = min(results.items(), key=rank)[0]
    return best_name, results
