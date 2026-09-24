"""Dataset discovery, loading and validation for the bundled coin OHLCV CSVs.

Standard-library only, so the data path works on any Python 3.9+ host -
including ones without numpy/pandas installed.

The bundled files look like::

    SNo,Name,Symbol,Date,High,Low,Open,Close,Volume,Marketcap
    1,Bitcoin,BTC,2013-04-29 23:59:59,147.48,134.0,134.44,144.53,0.0,1603768864.5

Columns are matched case-insensitively and the file is read with ``utf-8-sig``
so the UTF-8 BOM the CSVs ship with does not corrupt the first header name.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

__all__ = [
    "DataError",
    "Bar",
    "Series",
    "ValidationReport",
    "discover_datasets",
    "symbol_from_path",
    "load_series",
    "validate_series",
]

_DATE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y")
_DATE_PREFIX = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_NON_FINITE = {"nan", "null", "none", "inf", "-inf", "+inf", ""}


class DataError(ValueError):
    """Raised when a dataset cannot be parsed or fails validation."""


def _parse_date(raw: Optional[str]) -> date:
    text = (raw or "").strip()
    if not text:
        raise DataError("empty date")
    if "." in text:
        text = text.split(".")[0]
    text = text.split("T")[0].strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    match = _DATE_PREFIX.match(text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError as exc:
            # e.g. "2021-01-32" matches the pattern but is not a real date
            raise DataError(f"impossible date: {raw!r} ({exc})") from exc
    raise DataError(f"unparseable date: {raw!r}")


def _parse_float(raw: Optional[str]) -> float:
    text = (raw or "").strip()
    if text.lower() in _NON_FINITE:
        raise DataError(f"non-finite numeric value: {raw!r}")
    try:
        return float(text)
    except ValueError as exc:  # pragma: no cover - defensive
        raise DataError(f"unparseable number: {raw!r}") from exc


@dataclass(frozen=True)
class Bar:
    """A single OHLCV observation."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Series:
    """A validated, chronologically sorted price series for one coin."""

    symbol: str
    bars: List[Bar] = field(default_factory=list)
    source: Optional[str] = None

    def __len__(self) -> int:
        return len(self.bars)

    def __bool__(self) -> bool:  # an empty series is falsy
        return bool(self.bars)

    @property
    def dates(self) -> List[date]:
        return [b.date for b in self.bars]

    @property
    def closes(self) -> List[float]:
        return [b.close for b in self.bars]

    @property
    def highs(self) -> List[float]:
        return [b.high for b in self.bars]

    @property
    def lows(self) -> List[float]:
        return [b.low for b in self.bars]

    @property
    def volumes(self) -> List[float]:
        return [b.volume for b in self.bars]

    @property
    def start(self) -> Optional[date]:
        return self.bars[0].date if self.bars else None

    @property
    def end(self) -> Optional[date]:
        return self.bars[-1].date if self.bars else None

    def tail(self, n: int) -> "Series":
        """Return the last ``n`` bars as a new series (same symbol/source)."""
        if n <= 0:
            return Series(self.symbol, [], self.source)
        return Series(self.symbol, self.bars[-n:], self.source)

    def window(self, start: Optional[date] = None, end: Optional[date] = None) -> "Series":
        bars = [
            b for b in self.bars
            if (start is None or b.date >= start) and (end is None or b.date <= end)
        ]
        return Series(self.symbol, bars, self.source)

    def daily_returns(self) -> List[float]:
        """Simple close-to-close returns (empty when fewer than two bars)."""
        out: List[float] = []
        for prev, cur in zip(self.bars, self.bars[1:]):
            if prev.close == 0:
                continue
            out.append((cur.close - prev.close) / prev.close)
        return out


@dataclass
class ValidationReport:
    """Outcome of validating a series; warnings are non-fatal, errors are fatal."""

    symbol: str
    bars_in: int = 0
    bars_out: int = 0
    dropped_invalid: int = 0
    dropped_duplicates: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors and self.bars_out > 0

    @property
    def ok(self) -> bool:
        return self.is_valid

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "bars_in": self.bars_in,
            "bars_out": self.bars_out,
            "dropped_invalid": self.dropped_invalid,
            "dropped_duplicates": self.dropped_duplicates,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "is_valid": self.is_valid,
        }


def symbol_from_path(path: str) -> str:
    """``.../coin_Bitcoin.csv`` -> ``Bitcoin``."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith("coin_"):
        stem = stem[len("coin_"):]
    return stem or "UNKNOWN"


def discover_datasets(csv_dir: str) -> Dict[str, str]:
    """Map symbol -> csv path for every ``coin_*.csv`` under ``csv_dir``."""
    if not os.path.isdir(csv_dir):
        raise DataError(f"CSV directory not found: {csv_dir}")
    found: Dict[str, str] = {}
    for name in sorted(os.listdir(csv_dir)):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(csv_dir, name)
        if os.path.isfile(path):
            found[symbol_from_path(path)] = path
    return found


def _normalise_row(row: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        out[key.lstrip("\ufeff").strip().lower()] = value
    return out


def load_series(
    path: str,
    symbol: Optional[str] = None,
    *,
    min_bars: int = 30,
    strict: bool = False,
) -> tuple:
    """Load one CSV into a ``(Series, ValidationReport)`` pair.

    Malformed rows are dropped and counted, never silently trusted.
    ``strict=True`` promotes dropped rows to a validation error.
    """
    if not os.path.isfile(path):
        raise DataError(f"dataset not found: {path}")
    symbol = symbol or symbol_from_path(path)
    load_report = ValidationReport(symbol=symbol)

    raw_bars: List[Bar] = []
    with open(path, "r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DataError(f"{path}: file has no header")
        for lineno, row in enumerate(reader, start=2):
            norm = _normalise_row(row)
            load_report.bars_in += 1
            try:
                bar = Bar(
                    date=_parse_date(norm.get("date")),
                    open=_parse_float(norm.get("open")),
                    high=_parse_float(norm.get("high")),
                    low=_parse_float(norm.get("low")),
                    close=_parse_float(norm.get("close")),
                    volume=_parse_float(norm.get("volume") or "0"),
                )
            except DataError as exc:
                load_report.dropped_invalid += 1
                load_report.warnings.append(f"line {lineno}: skipped ({exc})")
                continue
            if bar.close <= 0 or bar.high <= 0 or bar.low <= 0:
                load_report.dropped_invalid += 1
                load_report.warnings.append(f"line {lineno}: non-positive price ignored")
                continue
            if bar.high < bar.low:
                bar = Bar(bar.date, bar.open, bar.low, bar.high, bar.close, bar.volume)
                load_report.warnings.append(f"line {lineno}: high/low swapped so high >= low")
            raw_bars.append(bar)

    if not raw_bars:
        # A file with a header but no parseable rows is unusable; say so plainly
        # instead of returning an empty series the caller might plot as zero.
        raise DataError(
            f"{path}: no valid rows parsed from {load_report.bars_in} line(s)"
        )

    series, report = validate_series(
        Series(symbol=symbol, bars=raw_bars, source=path), min_bars=min_bars
    )
    report.bars_in = load_report.bars_in
    report.dropped_invalid += load_report.dropped_invalid
    report.warnings = load_report.warnings + report.warnings
    if strict and report.dropped_invalid:
        report.errors.append(f"{report.dropped_invalid} malformed row(s) rejected")
    return series, report


def validate_series(series: Series, *, min_bars: int = 30) -> tuple:
    """Sort, de-duplicate and sanity-check a series.

    Guarantees on the returned series: strictly increasing unique dates, every
    bar has ``high >= low`` and positive prices.
    """
    report = ValidationReport(symbol=series.symbol, bars_in=len(series))

    if not series.bars:
        report.errors.append("series is empty")
        return series, report

    ordered = sorted(series.bars, key=lambda b: b.date)

    deduped: List[Bar] = []
    for bar in ordered:
        if deduped and deduped[-1].date == bar.date:
            report.dropped_duplicates += 1
            deduped[-1] = bar  # keep the latest revision for that day
            continue
        deduped.append(bar)

    if report.dropped_duplicates:
        report.warnings.append(
            f"collapsed {report.dropped_duplicates} duplicate date(s) keeping the latest row"
        )

    cleaned = [b for b in deduped if b.close > 0 and b.high > 0 and b.low > 0]
    report.dropped_invalid += len(deduped) - len(cleaned)
    report.bars_out = len(cleaned)

    if report.bars_out < min_bars:
        report.errors.append(
            f"only {report.bars_out} usable bar(s); need at least {min_bars}"
        )

    gaps = sum(1 for prev, cur in zip(cleaned, cleaned[1:]) if (cur.date - prev.date).days > 7)
    if gaps:
        report.warnings.append(f"{gaps} gap(s) longer than a week detected")

    return Series(symbol=series.symbol, bars=cleaned, source=series.source), report
