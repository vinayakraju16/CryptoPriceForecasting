"""Technical indicators and risk statistics, standard library only.

Implemented here rather than delegated to ``pandas-ta``/``TA-Lib`` so the
feature pipeline has no native build step and stays reproducible.

Conventions: an indicator returns a list the same length as its input, with
``None`` in the warm-up positions where it is not yet defined.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "sma", "ema", "rsi", "macd", "bollinger", "atr",
    "daily_returns", "volatility", "max_drawdown", "sharpe", "value_at_risk", "summary_stats",
]

Number = Optional[float]


def _clean(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values]


def sma(values: Sequence[float], period: int) -> List[Number]:
    """Simple moving average."""
    if period < 1:
        raise ValueError("period must be >= 1")
    vals = _clean(values)
    out: List[Number] = [None] * len(vals)
    running = 0.0
    for i, v in enumerate(vals):
        running += v
        if i >= period:
            running -= vals[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: Sequence[float], period: int) -> List[Number]:
    """Exponential moving average, seeded with the first value."""
    if period < 1:
        raise ValueError("period must be >= 1")
    vals = _clean(values)
    if not vals:
        return []
    alpha = 2.0 / (period + 1)
    out: List[Number] = [None] * len(vals)
    prev = vals[0]
    out[0] = prev
    for i in range(1, len(vals)):
        prev = alpha * vals[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def rsi(values: Sequence[float], period: int = 14) -> List[Number]:
    """Relative Strength Index using Wilder's smoothing."""
    if period < 1:
        raise ValueError("period must be >= 1")
    vals = _clean(values)
    out: List[Number] = [None] * len(vals)
    if len(vals) <= period:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        delta = vals[i] - vals[i - 1]
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1 + avg_gain / avg_loss)
    for i in range(period + 1, len(vals)):
        delta = vals[i] - vals[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0.0)) / period
        out[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1 + avg_gain / avg_loss)
    return out


def macd(
    values: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[List[Number], List[Number], List[Number]]:
    """Return ``(macd_line, signal_line, histogram)``."""
    if fast >= slow:
        raise ValueError("fast period must be shorter than slow period")
    vals = _clean(values)
    fast_ema = ema(vals, fast)
    slow_ema = ema(vals, slow)
    macd_line: List[Number] = [
        (f - s) if f is not None and s is not None else None
        for f, s in zip(fast_ema, slow_ema)
    ]
    defined = [v for v in macd_line if v is not None]
    signal_vals = ema(defined, signal) if defined else []
    signal_line: List[Number] = [None] * len(macd_line)
    cursor = 0
    for i, v in enumerate(macd_line):
        if v is None:
            continue
        signal_line[i] = signal_vals[cursor] if cursor < len(signal_vals) else None
        cursor += 1
    histogram: List[Number] = [
        (m - s) if m is not None and s is not None else None
        for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, histogram


def bollinger(
    values: Sequence[float],
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[List[Number], List[Number], List[Number]]:
    """Return ``(upper, middle, lower)`` Bollinger bands (population std)."""
    if period < 2:
        raise ValueError("period must be >= 2")
    vals = _clean(values)
    mid = sma(vals, period)
    upper: List[Number] = [None] * len(vals)
    lower: List[Number] = [None] * len(vals)
    for i in range(period - 1, len(vals)):
        window = vals[i - period + 1:i + 1]
        mean = mid[i]
        if mean is None:
            continue
        var = sum((v - mean) ** 2 for v in window) / period
        sd = math.sqrt(var)
        upper[i] = mean + num_std * sd
        lower[i] = mean - num_std * sd
    return upper, mid, lower


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> List[Number]:
    """Average True Range (Wilder smoothing) - a volatility proxy in price units."""
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows and closes must have equal length")
    if period < 1:
        raise ValueError("period must be >= 1")
    highs, lows, closes = _clean(highs), _clean(lows), _clean(closes)
    n = len(closes)
    out: List[Number] = [None] * n
    if n == 0:
        return out

    true_ranges: List[float] = [highs[0] - lows[0]]
    for i in range(1, n):
        true_ranges.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    if n <= period:
        return out
    prev = sum(true_ranges[1:period + 1]) / period
    out[period] = prev
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + true_ranges[i]) / period
        out[i] = prev
    return out


def daily_returns(values: Sequence[float]) -> List[float]:
    """Simple close-to-close returns."""
    vals = _clean(values)
    return [
        (cur - prev) / prev
        for prev, cur in zip(vals, vals[1:])
        if prev != 0
    ]


def volatility(values: Sequence[float], periods_per_year: int = 365) -> float:
    """Annualised volatility of close-to-close returns (crypto trades every day)."""
    returns = daily_returns(values)
    if len(returns) < 2:
        return float("nan")
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(var) * math.sqrt(periods_per_year)


def max_drawdown(values: Sequence[float]) -> float:
    """Largest peak-to-trough decline as a negative fraction (0.0 if never down)."""
    vals = _clean(values)
    if not vals:
        return float("nan")
    peak = vals[0]
    worst = 0.0
    for v in vals:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, (v - peak) / peak)
    return worst


def sharpe(
    values: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    """Annualised Sharpe ratio of simple daily returns."""
    returns = daily_returns(values)
    if len(returns) < 2:
        return float("nan")
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return float("nan")
    excess = mean - risk_free_rate / periods_per_year
    return excess / sd * math.sqrt(periods_per_year)


def value_at_risk(values: Sequence[float], confidence: float = 0.95) -> float:
    """Historical VaR of daily returns: the loss at the given confidence level.

    Returns a negative fraction (e.g. -0.08 for an 8% one-day loss) or ``nan``.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    returns = sorted(daily_returns(values))
    if len(returns) < 2:
        return float("nan")
    index = max(0, min(len(returns) - 1, int(math.floor((1 - confidence) * len(returns)))))
    return returns[index]


def summary_stats(values: Sequence[float]) -> dict:
    """Descriptive + risk statistics for a price series (JSON-safe)."""
    vals = _clean(values)
    if not vals:
        return {"n": 0}
    ordered = sorted(vals)
    def _pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = p * (len(ordered) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(ordered) - 1)
        frac = pos - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    returns = daily_returns(vals)
    result = {
        "n": len(vals),
        "last": vals[-1],
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
        "median": _pct(0.5),
        "p05": _pct(0.05),
        "p95": _pct(0.95),
        "total_return": (vals[-1] / vals[0] - 1) if vals[0] else float("nan"),
        "volatility_annual": volatility(vals),
        "max_drawdown": max_drawdown(vals),
        "sharpe": sharpe(vals),
        "var_95": value_at_risk(vals, 0.95),
        "best_day": max(returns) if returns else float("nan"),
        "worst_day": min(returns) if returns else float("nan"),
    }
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in result.items()}
