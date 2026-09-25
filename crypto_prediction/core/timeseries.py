"""Leak-free normalisation, windowing and chronological splits (stdlib only).

The original notebooks fitted ``MinMaxScaler`` on the *whole* series and then
shuffled rows before splitting. Both are bugs: fitting the scaler on the whole
series leaks the test window into training, and shuffling destroys the temporal
order a forecaster must respect. This module provides the safe replacements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

__all__ = ["MinMax", "make_windows", "sequential_split"]


@dataclass
class MinMax:
    """Min/max scaler fitted on a *training* window only."""

    feature_range: Tuple[float, float] = (0.0, 1.0)
    data_min_: float = 0.0
    data_max_: float = 1.0
    fitted_: bool = False

    def fit(self, values: Sequence[float]) -> "MinMax":
        clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        if not clean:
            raise ValueError("MinMax.fit received no finite values")
        self.data_min_ = min(clean)
        self.data_max_ = max(clean)
        self.fitted_ = True
        return self

    @property
    def _scale(self) -> float:
        span = self.data_max_ - self.data_min_
        return span if span > 0 else 1.0

    def transform(self, values: Sequence[float]) -> List[float]:
        if not self.fitted_:
            raise RuntimeError("MinMax.transform called before fit")
        lo, hi = self.feature_range
        return [lo + (float(v) - self.data_min_) / self._scale * (hi - lo) for v in values]

    def inverse_transform(self, values: Sequence[float]) -> List[float]:
        if not self.fitted_:
            raise RuntimeError("MinMax.inverse_transform called before fit")
        lo, hi = self.feature_range
        span = hi - lo
        if span == 0:
            raise RuntimeError("degenerate feature_range with zero span")
        return [self.data_min_ + (float(v) - lo) / span * self._scale for v in values]

    def fit_transform(self, values: Sequence[float]) -> List[float]:
        return self.fit(values).transform(values)


def make_windows(
    values: Sequence[float],
    lookback: int,
    horizon: int = 1,
) -> Tuple[List[List[float]], List[float]]:
    """Supervised windows: ``x = values[i:i+lookback]``, ``y = values[i+lookback+horizon-1]``.

    The target is always a *future* value, so the label can never be part of
    its own input window.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    xs: List[List[float]] = []
    ys: List[float] = []
    last_start = len(values) - lookback - horizon + 1
    for i in range(max(0, last_start)):
        xs.append([float(v) for v in values[i:i + lookback]])
        ys.append(float(values[i + lookback + horizon - 1]))
    return xs, ys


def sequential_split(
    n: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[slice, slice, slice]:
    """Chronological train/val/test slices - never shuffled."""
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    if not 0 <= val_frac < 1 - train_frac:
        raise ValueError("val_frac leaves no room for a test split")
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)
