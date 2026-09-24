"""Walk-forward (expanding-window) backtesting.

Why this module exists: the notebooks trained on a random 70% of the data and
then measured error on the remaining 30%. That overstates accuracy on a time
series - future data leaks into training and the reported score is not what you
would have earned trading live.

``walk_forward`` instead replays history in order. At each step it trains on
everything up to that point and predicts the *next* bar, which is exactly the
situation the model faces in production. It also reports the equity curve of a
simple long/flat strategy driven by the same predictions, so accuracy and
profitability are not confused with one another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .metrics import metric_suite

__all__ = ["BacktestResult", "walk_forward", "strategy_returns", "buy_and_hold"]


@dataclass
class BacktestResult:
    """Outcome of a walk-forward run."""

    model: str
    actual: List[float] = field(default_factory=list)
    predicted: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    strategy: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "metrics": self.metrics,
            "strategy": self.strategy,
            "predictions": [
                {"date": d, "actual": a, "predicted": p}
                for d, a, p in zip(self.dates, self.actual, self.predicted)
            ],
        }


def walk_forward(
    model,
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    scaler,
    dates: Sequence,
    *,
    min_train: int = 60,
    retrain_every: int = 1,
    warmup: int = 0,
) -> BacktestResult:
    """Expanding-window evaluation.

    ``xs``/``ys`` are scaled window/target pairs; ``scaler`` maps predictions
    back to price space. ``min_train`` is how many windows must exist before
    the first prediction, ``retrain_every`` amortises refitting.
    """
    result = BacktestResult(model=getattr(model, "name", "model"))
    if len(xs) <= min_train:
        result.metrics = {"n": 0, "error": "not enough windows for a walk-forward run"}
        return result

    start = max(min_train, warmup)
    fitted = False
    for i in range(start, len(xs)):
        if not fitted or (i - start) % max(1, retrain_every) == 0:
            train_x = xs[:i]
            train_y = ys[:i]
            model.fit(train_x, train_y)
            fitted = True
        predicted_scaled = model.predict(xs[i])
        actual_scaled = ys[i]
        actual = scaler.inverse_transform([actual_scaled])[0]
        predicted = scaler.inverse_transform([predicted_scaled])[0]
        result.actual.append(float(actual))
        result.predicted.append(float(predicted))
        if i < len(dates):
            result.dates.append(str(dates[i]))

    result.metrics = metric_suite(result.actual, result.predicted)
    result.strategy = strategy_returns(result.actual, result.predicted)
    return result


def strategy_returns(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    fee: float = 0.001,
) -> Dict[str, float]:
    """Backtest a long/flat rule: hold when the model predicts a rise.

    A round-trip fee is charged whenever the position changes, so a model that
    flip-flops pays for it. Returns cumulative return, Sharpe-like ratio,
    max drawdown and the number of trades - realistic totals, not headline ones.
    """
    if len(actual) < 2 or len(actual) != len(predicted):
        return {"n": 0, "error": "insufficient data"}

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    trades = 0
    position = 0  # 0 = flat, 1 = long
    returns: List[float] = []

    for i in range(1, len(actual)):
        prev_actual = actual[i - 1]
        if prev_actual <= 0:
            continue
        signal = 1 if predicted[i] > prev_actual else 0
        bar_return = actual[i] / prev_actual - 1.0

        if signal != position:
            equity *= (1.0 - fee)
            trades += 1
            position = signal
        step = bar_return * position
        equity *= (1.0 + step)
        returns.append(step)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, (equity - peak) / peak)

    if not returns:
        return {"n": 0, "error": "no tradable steps"}

    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / max(1, len(returns) - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    sharpe = (mean / sd * math.sqrt(365)) if sd > 0 else 0.0

    return {
        "n": len(returns),
        "total_return": equity - 1.0,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": trades,
        "hit_rate": sum(1 for r in returns if r > 0) / len(returns),
        "fee_per_trade": fee,
    }


def buy_and_hold(actual: Sequence[float]) -> Dict[str, float]:
    """Reference strategy: buy the first bar and hold to the end."""
    if len(actual) < 2 or actual[0] <= 0:
        return {"n": 0, "error": "insufficient data"}
    peak = actual[0]
    max_dd = 0.0
    for value in actual:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, (value - peak) / peak)
    return {
        "n": len(actual),
        "total_return": actual[-1] / actual[0] - 1.0,
        "max_drawdown": max_dd,
        "trades": 1,
    }
