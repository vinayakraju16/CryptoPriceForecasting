"""Live market data adapters, with an injectable transport for offline testing.

The bundled CSVs stop at 2021-07-06, so every forecast they produce is a
historical backtest rather than a current view. This module closes that gap: it
fetches recent daily candles from a public REST endpoint and turns them into the
same ``Series`` object the rest of the pipeline already consumes.

Design choices that matter:

* **No hard dependency.** The default transport is ``urllib`` from the standard
  library. Tests inject a fake transport, so the parsing and validation logic is
  verified without touching the network.
* **Explicit on failure.** A provider change, a rate limit or a dead network
  raises ``DataError`` with the provider's message rather than silently
  returning an empty series that would produce a meaningless forecast.
* **Append, do not replace.** ``merge_series`` extends an existing series with
  newer bars, de-duplicating by date and keeping the latest revision, so a
  cached history can be topped up cheaply.

Providers: ``binance`` (public klines, no key) and ``coingecko`` (public market
chart, no key). Both are documented public endpoints; respect their rate limits.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence

from .dataset import Bar, DataError, Series, validate_series

__all__ = [
    "HttpTransport",
    "FetchResult",
    "fetch_binance_daily",
    "fetch_coingecko_daily",
    "PROVIDERS",
    "merge_series",
    "series_from_bars",
    "fetch_and_merge",
]

# A transport takes (url, timeout_seconds) and returns the decoded JSON body.
HttpTransport = Callable[[str, float], object]

BINANCE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
    "?vs_currency=usd&days={limit}&interval=daily"
)

# Provider symbol for each coin we ship history for.
BINANCE_SYMBOLS = {
    "Bitcoin": "BTCUSDT",
    "Ethereum": "ETHUSDT",
    "BinanceCoin": "BNBUSDT",
    "Cardano": "ADAUSDT",
    "ChainLink": "LINKUSDT",
    "Dogecoin": "DOGEUSDT",
    "EOS": "EOSUSDT",
    "Cosmos": "ATOMUSDT",
    "Aave": "AAVEUSDT",
    "CryptocomCoin": "CROUSDT",
}

COINGECKO_IDS = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "BinanceCoin": "binancecoin",
    "Cardano": "cardano",
    "ChainLink": "chainlink",
    "Dogecoin": "dogecoin",
    "EOS": "eos",
    "Cosmos": "cosmos",
    "Aave": "aave",
    "CryptocomCoin": "crypto-com-chain",
}


def default_transport(url: str, timeout: float = 20.0):
    """Standard-library HTTP GET returning decoded JSON."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "crypto-forecast/2.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        raise DataError(f"provider returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DataError(f"could not reach the provider: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise DataError(f"provider returned malformed JSON: {exc}") from exc


@dataclass
class FetchResult:
    series: Series
    provider: str
    fetched: int
    dropped: int
    warnings: List[str]

    def to_dict(self) -> dict:
        return {
            "symbol": self.series.symbol,
            "provider": self.provider,
            "fetched": self.fetched,
            "dropped": self.dropped,
            "bars": len(self.series),
            "start": str(self.series.start),
            "end": str(self.series.end),
            "warnings": self.warnings,
        }


def _ms_to_date(value) -> date:
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).date()


def fetch_binance_daily(
    symbol: str,
    *,
    limit: int = 1000,
    transport: Optional[HttpTransport] = None,
) -> FetchResult:
    """Daily candles from Binance klines (no API key required).

    Response shape: ``[[openTime, open, high, low, close, volume, ...], ...]``
    where every numeric field is a *string*.
    """
    provider_symbol = BINANCE_SYMBOLS.get(symbol, symbol)
    url = BINANCE_URL.format(symbol=provider_symbol, limit=min(int(limit), 1000))
    payload = (transport or default_transport)(url, 20.0)

    if not isinstance(payload, list) or not payload:
        raise DataError(f"Binance returned no candles for {provider_symbol}")

    bars: List[Bar] = []
    dropped = 0
    warnings: List[str] = []
    for row in payload:
        try:
            open_time, open_, high, low, close, volume = row[:6]
            bar = Bar(
                date=_ms_to_date(open_time),
                open=float(open_),
                high=float(high),
                low=float(low),
                close=float(close),
                volume=float(volume),
            )
        except (TypeError, ValueError, IndexError) as exc:
            dropped += 1
            warnings.append(f"skipped malformed candle: {exc}")
            continue
        if bar.close <= 0:
            dropped += 1
            continue
        bars.append(bar)

    series, report = validate_series(Series(symbol=symbol, bars=bars, source=f"binance:{provider_symbol}"), min_bars=2)
    if not report.is_valid:
        raise DataError(f"Binance data for {symbol} failed validation: {'; '.join(report.errors)}")
    warnings.extend(report.warnings)
    return FetchResult(series, "binance", len(bars), dropped, warnings)


def fetch_coingecko_daily(
    symbol: str,
    *,
    limit: int = 365,
    transport: Optional[HttpTransport] = None,
) -> FetchResult:
    """Daily prices from CoinGecko's market chart (no API key required).

    CoinGecko returns ``{"prices": [[ms, price], ...]}`` with no OHLC, so close
    is used for every OHLC field - honest for a close-only forecast, and stated
    plainly rather than faked.
    """
    coin_id = COINGECKO_IDS.get(symbol, symbol.lower())
    url = COINGECKO_URL.format(id=coin_id, limit=min(int(limit), 365))
    payload = (transport or default_transport)(url, 20.0)

    points = payload.get("prices") if isinstance(payload, dict) else None
    if not points:
        raise DataError(f"CoinGecko returned no prices for {coin_id}")

    warnings: List[str] = ["CoinGecko provides close prices only; OHLC fields mirror close"]
    bars: List[Bar] = []
    dropped = 0
    for point in points:
        try:
            stamp, price = point[0], float(point[1])
        except (TypeError, ValueError, IndexError):
            dropped += 1
            continue
        if price <= 0:
            dropped += 1
            continue
        bars.append(Bar(date=_ms_to_date(stamp), open=price, high=price, low=price, close=price, volume=0.0))

    series, report = validate_series(Series(symbol=symbol, bars=bars, source=f"coingecko:{coin_id}"), min_bars=2)
    if not report.is_valid:
        raise DataError(f"CoinGecko data for {symbol} failed validation: {'; '.join(report.errors)}")
    warnings.extend(report.warnings)
    return FetchResult(series, "coingecko", len(bars), dropped, warnings)


PROVIDERS: Dict[str, Callable[..., FetchResult]] = {
    "binance": fetch_binance_daily,
    "coingecko": fetch_coingecko_daily,
}


def series_from_bars(symbol: str, bars: Sequence[Bar], source: str = "supplied") -> Series:
    """Validate a hand-built list of bars into a ``Series``."""
    series, report = validate_series(Series(symbol=symbol, bars=list(bars), source=source), min_bars=2)
    if not report.is_valid:
        raise DataError(f"{symbol}: {'; '.join(report.errors)}")
    return series


def merge_series(base: Series, fresh: Series) -> Series:
    """Extend ``base`` with ``fresh``, newest revision winning per date.

    Returns a sorted, de-duplicated series. ``fresh`` values overwrite ``base``
    values for the same date, which is what you want when a cached bar was
    provisional (today's candle, before the close).
    """
    by_date: Dict[date, Bar] = {bar.date: bar for bar in base.bars}
    for bar in fresh.bars:
        by_date[bar.date] = bar
    merged = [by_date[key] for key in sorted(by_date)]
    series, report = validate_series(
        Series(symbol=base.symbol, bars=merged, source=base.source or fresh.source), min_bars=2
    )
    if not report.is_valid:
        raise DataError(f"merge produced an unusable series: {'; '.join(report.errors)}")
    return series


def fetch_and_merge(
    base: Optional[Series],
    symbol: str,
    *,
    provider: str = "binance",
    limit: int = 1000,
    transport: Optional[HttpTransport] = None,
) -> FetchResult:
    """Fetch recent bars and merge them onto ``base`` (or start fresh)."""
    if provider not in PROVIDERS:
        raise KeyError(f"unknown provider {provider!r}; available: {sorted(PROVIDERS)}")
    fetched = PROVIDERS[provider](symbol, limit=limit, transport=transport)
    merged = merge_series(base, fetched.series) if base and base.bars else fetched.series
    return FetchResult(
        series=merged,
        provider=fetched.provider,
        fetched=fetched.fetched,
        dropped=fetched.dropped,
        warnings=fetched.warnings,
    )
