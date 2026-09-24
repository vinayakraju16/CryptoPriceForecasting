"""Turning a forecast into something actionable.

The honest position: nobody can reliably predict tomorrow's crypto price, and a
tool that claims to should not be trusted. What *is* defensible is to forecast,
quantify how wrong the forecast has recently been, and size the signal against
that uncertainty. That is what this module does.

``build_prediction`` produces, for one coin:

* ``predicted_next_price`` with a ``low``/``high`` band from recent residuals
  (so you see the spread, not a single number pretending to be certain),
* ``expected_return`` over the horizon,
* risk statistics and the recent accuracy of the model,
* a ``signal`` in {strong_buy, buy, hold, sell, strong_sell} derived from the
  forecast *relative to its own recent error* - not from an arbitrary threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .backtest import walk_forward
from .indicators import max_drawdown, sharpe, summary_stats, volatility, value_at_risk
from .metrics import directional_accuracy, mae, mape, rmse, smape
from .models import MODEL_REGISTRY, build_model, feature_matrix
from .timeseries import MinMax, make_windows

__all__ = ["Prediction", "build_prediction", "screen_universe", "signal_from_edge", "position_size", "recommend"]

DEFAULT_LOOKBACK = 30
DEFAULT_MIN_TRAIN = 120


def signal_from_edge(edge: float, error_scale: float) -> str:
    """Map a forecast edge to a discrete signal.

    ``edge`` is the expected fractional return; ``error_scale`` is the recent
    typical absolute error (as a fraction). The signal only fires when the
    expected move clears the model's own noise, scaled by a hand-set tolerance.
    """
    if error_scale <= 0 or not math.isfinite(error_scale):
        error_scale = 0.05
    z = edge / error_scale
    if z >= 1.0:
        return "strong_buy"
    if z >= 0.35:
        return "buy"
    if z <= -1.0:
        return "strong_sell"
    if z <= -0.35:
        return "sell"
    return "hold"


@dataclass
class Prediction:
    """A single-coin forecast bundle."""

    symbol: str
    model: str
    last_date: Optional[str]
    last_price: float
    predicted_next_price: float
    expected_return: float
    low: float
    high: float
    error_scale: float
    signal: str
    confidence: float
    accuracy: Dict[str, Optional[float]] = field(default_factory=dict)
    risk: Dict[str, Optional[float]] = field(default_factory=dict)
    horizon_days: int = 1
    disclaimer: str = (
        "Statistical projection from historical prices only. Not financial "
        "advice; crypto is highly volatile and forecasts can be badly wrong."
    )

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "model": self.model,
            "last_date": self.last_date,
            "last_price": self.last_price,
            "predicted_next_price": self.predicted_next_price,
            "predicted_low": self.low,
            "predicted_high": self.high,
            "expected_return": self.expected_return,
            "expected_return_pct": self.expected_return * 100.0,
            "error_scale": self.error_scale,
            "signal": self.signal,
            "confidence": self.confidence,
            "horizon_days": self.horizon_days,
            "accuracy": self.accuracy,
            "risk": self.risk,
            "disclaimer": self.disclaimer,
        }


def _residual_scale(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Median absolute relative residual - a robust error scale (a fraction)."""
    residuals = sorted(
        abs(p - a) / a
        for a, p in zip(actual, predicted)
        if a and math.isfinite(a) and math.isfinite(p)
    )
    if not residuals:
        return 0.0
    mid = len(residuals) // 2
    if len(residuals) % 2:
        return residuals[mid]
    return (residuals[mid - 1] + residuals[mid]) / 2.0


def build_prediction(
    series,
    model_name: str = "ensemble",
    *,
    lookback: int = DEFAULT_LOOKBACK,
    min_train: int = DEFAULT_MIN_TRAIN,
    horizon: int = 1,
) -> Prediction:
    """Fit a walk-forward-evaluated model to ``series`` and forecast forward.

    The model is judged on its out-of-sample residuals, and those same
    residuals set the confidence band - so a coin the model has handled badly
    automatically gets a wide band and a weaker signal.
    """
    closes = series.closes
    if len(closes) < lookback + min_train:
        raise ValueError(
            f"{series.symbol}: need at least {lookback + min_train} bars, have {len(closes)}"
        )
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {model_name!r}")

    scaler = MinMax()
    split = max(lookback + 1, int(len(closes) * 0.7))
    scaler.fit(closes[:split])
    scaled = scaler.transform(closes)
    xs, ys = make_windows(scaled, lookback, horizon)
    dates = series.dates[lookback + horizon - 1:]

    model = build_model(model_name)
    result = walk_forward(model, xs, ys, scaler, dates, min_train=min_train)
    actual = result.actual
    predicted = result.predicted

    error_scale = _residual_scale(actual, predicted)
    accuracy = {
        "mae": mae(actual, predicted) if actual else None,
        "rmse": rmse(actual, predicted) if actual else None,
        "mape": mape(actual, predicted) if actual else None,
        "smape": smape(actual, predicted) if actual else None,
        "directional_accuracy": directional_accuracy(actual, predicted) if actual else None,
        "n_evaluated": len(actual),
    }

    # Refit on everything available, then forecast the true next step.
    model_final = build_model(model_name)
    model_final.fit(xs, ys)
    window = scaled[-lookback:]
    forecast_scaled = model_final.forecast(window, steps=horizon)
    forecast_price = scaler.inverse_transform(forecast_scaled)
    next_price = float(forecast_price[-1])

    last_price = float(closes[-1])
    band = max(error_scale, 1e-6) * math.sqrt(horizon)
    diagnostics = summary_stats(closes)
    risk = {
        "volatility_annual": diagnostics.get("volatility_annual"),
        "max_drawdown": diagnostics.get("max_drawdown"),
        "sharpe": diagnostics.get("sharpe"),
        "var_95_daily": diagnostics.get("var_95"),
        "best_day": diagnostics.get("best_day"),
        "worst_day": diagnostics.get("worst_day"),
    }

    expected_return = next_price / last_price - 1.0 if last_price else 0.0
    confidence = 0.0
    if accuracy["directional_accuracy"] is not None:
        confidence = max(0.0, min(1.0, (accuracy["directional_accuracy"] - 0.5) * 2.0))

    return Prediction(
        symbol=series.symbol,
        model=model_name,
        last_date=str(series.end) if series.end else None,
        last_price=last_price,
        predicted_next_price=next_price,
        expected_return=expected_return,
        low=next_price * (1.0 - band),
        high=next_price * (1.0 + band),
        error_scale=error_scale,
        signal=signal_from_edge(expected_return, error_scale),
        confidence=confidence,
        accuracy=accuracy,
        risk=risk,
        horizon_days=horizon,
    )


def position_size(
    expected_return: float,
    error_scale: float,
    *,
    capital: float = 1000.0,
    risk_per_trade: float = 0.02,
    max_weight: float = 0.25,
) -> Dict[str, float]:
    """Risk-based position sizing - the number the forecast alone cannot give.

    The forecast says *direction*; sizing says *how much*. A Kelly-flavoured
    but conservative rule: risk a fixed fraction of capital per trade, where the
    per-unit risk is the model's own error scale. The result is capped at
    ``max_weight`` of capital so one noisy call cannot take the whole book.

    Returns the cash amount, the weight, and the risk actually assumed.
    """
    if capital <= 0:
        return {"cash": 0.0, "weight": 0.0, "risk_cash": 0.0, "note": "capital must be positive"}
    edge = max(0.0, float(expected_return))
    unit_risk = max(float(error_scale), 1e-9)
    if edge <= 0:
        return {"cash": 0.0, "weight": 0.0, "risk_cash": 0.0, "note": "no positive edge; stay flat"}

    risk_budget = capital * risk_per_trade
    cash = risk_budget / unit_risk
    weight = min(cash / capital, max_weight)
    cash = capital * weight
    return {
        "cash": cash,
        "weight": weight,
        "risk_cash": cash * unit_risk,
        "capital": capital,
        "risk_per_trade": risk_per_trade,
        "note": "size from model error; not financial advice",
    }


def recommend(
    datasets: Dict[str, object],
    model_name: str = "ensemble",
    *,
    lookback: int = DEFAULT_LOOKBACK,
    min_train: int = DEFAULT_MIN_TRAIN,
    capital: float = 1000.0,
    risk_per_trade: float = 0.02,
    min_confidence: float = 0.0,
    top: int = 3,
) -> List[dict]:
    """Answer the practical question: given this model, what is worth buying?

    Only coins with a positive expected return *and* a directional edge above
    ``min_confidence`` are proposed, each with a sized allocation. Coins the
    model merely cannot price are excluded rather than ranked low - silence is
    the honest output for those.
    """
    ranked = screen_universe(
        datasets, model_name, lookback=lookback, min_train=min_train
    )
    proposals: List[dict] = []
    for row in ranked:
        if "error" in row:
            continue
        if row.get("expected_return", 0.0) <= 0:
            continue
        if row.get("confidence", 0.0) < min_confidence:
            continue
        sizing = position_size(
            row["expected_return"],
            row.get("error_scale", 0.0),
            capital=capital,
            risk_per_trade=risk_per_trade,
        )
        proposal = dict(row)
        proposal["allocation"] = sizing
        proposals.append(proposal)

    proposals.sort(key=lambda r: r.get("risk_adjusted_score") or 0.0, reverse=True)
    return proposals[:top]


def screen_universe(
    datasets: Dict[str, object],
    model_name: str = "ensemble",
    *,
    lookback: int = DEFAULT_LOOKBACK,
    min_train: int = DEFAULT_MIN_TRAIN,
    top: Optional[int] = None,
) -> List[dict]:
    """Rank every coin by risk-adjusted expected return and return the list.

    Ranking uses the forecast edge divided by the model's own error scale (the
    same quantity the signal uses), so a big predicted jump on a coin the model
    cannot price reliably does not automatically top the leaderboard.
    """
    rows: List[dict] = []
    for symbol, series in datasets.items():
        try:
            prediction = build_prediction(
                series, model_name, lookback=lookback, min_train=min_train
            )
        except Exception as exc:  # one bad coin must not sink the leaderboard
            rows.append({"symbol": symbol, "error": str(exc)})
            continue
        scale = prediction.error_scale or 1e-6
        rows.append(
            {
                "symbol": symbol,
                "last_date": prediction.last_date,
                "last_price": prediction.last_price,
                "predicted_next_price": prediction.predicted_next_price,
                "expected_return": prediction.expected_return,
                "expected_return_pct": prediction.expected_return * 100.0,
                "signal": prediction.signal,
                "confidence": prediction.confidence,
                "error_scale": prediction.error_scale,
                "risk_adjusted_score": prediction.expected_return / scale,
                "volatility_annual": prediction.risk.get("volatility_annual"),
                "max_drawdown": prediction.risk.get("max_drawdown"),
                "sharpe": prediction.risk.get("sharpe"),
                "directional_accuracy": prediction.accuracy.get("directional_accuracy"),
            }
        )

    scored = [r for r in rows if "error" not in r]
    failed = [r for r in rows if "error" in r]
    scored.sort(key=lambda r: r.get("risk_adjusted_score") or float("-inf"), reverse=True)
    ordered = scored + failed
    return ordered[:top] if top else ordered
