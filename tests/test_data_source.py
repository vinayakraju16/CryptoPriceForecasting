"""Tests for live data adapters - all offline via an injected transport."""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crypto_prediction.core import Bar, Series  # noqa: E402
from crypto_prediction.core.data_source import (  # noqa: E402
    PROVIDERS,
    fetch_and_merge,
    fetch_binance_daily,
    fetch_coingecko_daily,
    merge_series,
    series_from_bars,
)
from crypto_prediction.core.dataset import DataError  # noqa: E402

DAY_MS = 86400 * 1000


def binance_payload(n=5, start_ms=1600000000000, price=100.0):
    """Mimic Binance klines: strings, 12 columns."""
    rows = []
    for i in range(n):
        close = price + i
        rows.append(
            [
                start_ms + i * DAY_MS,
                str(close - 1),
                str(close + 2),
                str(close - 2),
                str(close),
                str(1000 + i),
                start_ms + i * DAY_MS + DAY_MS - 1,
                "0", 0, "0", "0", "0",
            ]
        )
    return rows


def coingecko_payload(n=5, start_ms=1600000000000, price=100.0):
    return {"prices": [[start_ms + i * DAY_MS, price + i] for i in range(n)]}


class TestBinanceAdapter(unittest.TestCase):
    def test_parses_candles(self):
        result = fetch_binance_daily("Bitcoin", transport=lambda url, t: binance_payload(5))
        self.assertEqual(result.provider, "binance")
        self.assertEqual(len(result.series), 5)
        self.assertEqual(result.series.symbol, "Bitcoin")
        self.assertGreater(result.series.bars[-1].close, result.series.bars[0].close)

    def test_url_uses_provider_symbol(self):
        seen = {}

        def transport(url, timeout):
            seen["url"] = url
            return binance_payload()

        fetch_binance_daily("Dogecoin", transport=transport)
        self.assertIn("DOGEUSDT", seen["url"])

    def test_drops_malformed_rows(self):
        payload = binance_payload(4)
        payload.insert(1, ["bad", "x", "y", "z", "w", "v"])
        result = fetch_binance_daily("Bitcoin", transport=lambda u, t: payload)
        self.assertEqual(result.dropped, 1)
        self.assertEqual(len(result.series), 4)

    def test_empty_response_raises(self):
        with self.assertRaises(DataError):
            fetch_binance_daily("Bitcoin", transport=lambda u, t: [])

    def test_non_positive_price_dropped(self):
        payload = binance_payload(3)
        payload[0][4] = "0"
        result = fetch_binance_daily("Bitcoin", transport=lambda u, t: payload)
        self.assertEqual(len(result.series), 2)

    def test_result_serialises(self):
        result = fetch_binance_daily("Bitcoin", transport=lambda u, t: binance_payload(3))
        payload = result.to_dict()
        self.assertEqual(payload["provider"], "binance")
        self.assertEqual(payload["bars"], 3)


class TestCoinGeckoAdapter(unittest.TestCase):
    def test_parses_prices(self):
        result = fetch_coingecko_daily("Ethereum", transport=lambda u, t: coingecko_payload(6))
        self.assertEqual(result.provider, "coingecko")
        self.assertEqual(len(result.series), 6)
        self.assertTrue(any("close prices only" in w for w in result.warnings))

    def test_ohlc_fields_mirror_close(self):
        result = fetch_coingecko_daily("Ethereum", transport=lambda u, t: coingecko_payload(3))
        bar = result.series.bars[0]
        self.assertEqual(bar.open, bar.close)
        self.assertEqual(bar.high, bar.close)

    def test_missing_prices_raises(self):
        with self.assertRaises(DataError):
            fetch_coingecko_daily("Ethereum", transport=lambda u, t: {"prices": []})

    def test_uses_coin_id(self):
        seen = {}

        def transport(url, timeout):
            seen["url"] = url
            return coingecko_payload()

        fetch_coingecko_daily("CryptocomCoin", transport=transport)
        self.assertIn("crypto-com-chain", seen["url"])


class TestMerge(unittest.TestCase):
    def _series(self, symbol, start_day, n, price=100.0):
        bars = []
        for i in range(n):
            value = price + i
            bars.append(
                Bar(
                    date=start_day + timedelta(days=i),
                    open=value, high=value, low=value, close=value, volume=1.0,
                )
            )
        return Series(symbol=symbol, bars=bars, source="test")

    def test_merge_extends_without_duplicates(self):
        base = self._series("X", date(2021, 1, 1), 5)
        fresh = self._series("X", date(2021, 1, 3), 5)
        merged = merge_series(base, fresh)
        self.assertEqual(len(merged), 7)  # 1..5 plus 6..7
        self.assertEqual(merged.dates, sorted(merged.dates))

    def test_newer_revision_wins(self):
        base = self._series("X", date(2021, 1, 1), 3, price=100.0)
        fresh_bars = [Bar(date(2021, 1, 2), 999, 999, 999, 999.0, 1.0)]
        fresh = Series(symbol="X", bars=fresh_bars, source="fresh")
        merged = merge_series(base, fresh)
        by_date = {bar.date: bar.close for bar in merged.bars}
        self.assertEqual(by_date[date(2021, 1, 2)], 999.0)

    def test_merge_with_empty_fresh_is_a_noop(self):
        base = self._series("X", date(2021, 1, 1), 4)
        merged = merge_series(base, Series(symbol="X", bars=[], source="fresh"))
        self.assertEqual(len(merged), 4)

    def test_fetch_and_merge_extends_base(self):
        base = series_from_bars(
            "Bitcoin",
            [
                Bar(date(2020, 9, 13), 1, 1, 1, 1.0, 0),
                Bar(date(2020, 9, 14), 1, 1, 1, 2.0, 0),
            ],
        )
        result = fetch_and_merge(
            base, "Bitcoin", provider="binance",
            transport=lambda u, t: binance_payload(5, start_ms=1600000000000),
        )
        self.assertGreaterEqual(len(result.series), 5)
        self.assertEqual(result.series.dates, sorted(result.series.dates))

    def test_fetch_and_merge_without_base(self):
        result = fetch_and_merge(
            None, "Bitcoin", provider="binance",
            transport=lambda u, t: binance_payload(4),
        )
        self.assertEqual(len(result.series), 4)

    def test_unknown_provider_raises(self):
        with self.assertRaises(KeyError):
            fetch_and_merge(None, "Bitcoin", provider="not-a-provider")

    def test_providers_registered(self):
        self.assertEqual(set(PROVIDERS), {"binance", "coingecko"})

    def test_default_transport_error_is_wrapped(self):
        """A dead network must surface as DataError, not a urllib traceback."""
        from crypto_prediction.core.data_source import default_transport

        with self.assertRaises(DataError):
            default_transport("http://127.0.0.1:9/nothing", timeout=1.0)


class TestSeriesFromBars(unittest.TestCase):
    def test_validates(self):
        bars = [Bar(date(2021, 1, 1) + timedelta(days=i), 1, 1, 1, 1.0 + i, 0) for i in range(5)]
        series = series_from_bars("X", bars)
        self.assertEqual(len(series), 5)

    def test_rejects_empty(self):
        with self.assertRaises(DataError):
            series_from_bars("X", [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
