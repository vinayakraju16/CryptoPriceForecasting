"""Forecast accuracy metrics (standard library only).

``mape`` is included for compatibility with the original notebooks but is not
reliable for coins whose price approaches zero; ``smape`` is the safe default
and ``directional_accuracy`` is the metric that actually matters for trading.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

__all__ = ["mae", "rmse", "mape", "smape", "r2", "directional_accuracy", "metric_suite"]


def _paired(actual: Sequence[float], predicted: Sequence[float]) -> Tuple[List[float], List[float]]:
    if len(actual) != len(predicted):
        raise ValueError(f"length mismatch: actual={len(actual)} predicted={len(predicted)}")
    return [float(x) for x in actual], [float(y) for y in predicted]


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    a, p = _paired(actual, predicted)
    if not a:
        return float("nan")
    return sum(abs(x - y) for x, y in zip(a, p)) / len(a)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    a, p = _paired(actual, predicted)
    if not a:
        return float("nan")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, p)) / len(a))


def mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute percentage error, ignoring zero actuals."""
    a, p = _paired(actual, predicted)
    terms = [abs((x - y) / x) for x, y in zip(a, p) if x != 0]
    if not terms:
        return float("nan")
    return 100.0 * sum(terms) / len(terms)


def smape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Symmetric MAPE - stable when actuals are small, as they are for many coins."""
    a, p = _paired(actual, predicted)
    terms = [
        2.0 * abs(x - y) / (abs(x) + abs(y))
        for x, y in zip(a, p)
        if abs(x) + abs(y) != 0
    ]
    if not terms:
        return float("nan")
    return 100.0 * sum(terms) / len(terms)


def r2(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Coefficient of determination; ``nan`` when the actuals have no variance."""
    a, p = _paired(actual, predicted)
    if not a:
        return float("nan")
    mean = sum(a) / len(a)
    ss_tot = sum((x - mean) ** 2 for x in a)
    if ss_tot == 0:
        return float("nan")
    ss_res = sum((x - y) ** 2 for x, y in zip(a, p))
    return 1.0 - ss_res / ss_tot


def directional_accuracy(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Share of steps where the predicted move matches the actual move.

    Compares each prediction against the *previous actual* value, which is the
    decision a trader actually makes ("is it going up or down from here?").
    Returns a fraction in [0, 1]; ``nan`` with fewer than two observations.
    """
    a, p = _paired(actual, predicted)
    if len(a) < 2:
        return float("nan")
    hits = total = 0
    for i in range(1, len(a)):
        actual_move = a[i] - a[i - 1]
        predicted_move = p[i] - a[i - 1]
        if actual_move == 0 and predicted_move == 0:
            hits += 1
        elif actual_move != 0 and predicted_move != 0 and (actual_move > 0) == (predicted_move > 0):
            hits += 1
        total += 1
    return hits / total if total else float("nan")


def metric_suite(actual: Sequence[float], predicted: Sequence[float]) -> dict:
    """All metrics in one dict, ready for JSON serialisation.

    Non-finite values become ``None`` so the payload is always valid JSON.
    """
    values = {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
        "smape": smape(actual, predicted),
        "r2": r2(actual, predicted),
        "directional_accuracy": directional_accuracy(actual, predicted),
        "n": len(actual),
    }
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in values.items()}
