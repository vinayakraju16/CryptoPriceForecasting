"""Tests for sentiment scoring and position sizing."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crypto_prediction.core.sentiment import (  # noqa: E402
    aggregate_daily_sentiment,
    combine_with_price,
    score_text,
)
from crypto_prediction.core.signals import position_size  # noqa: E402


class TestSentimentLexicon(unittest.TestCase):
    def test_positive_and_negative(self):
        self.assertGreater(score_text("Bitcoin is bullish, huge gains coming").score, 0.2)
        self.assertLess(score_text("Total scam, the token was rug pulled and dumped").score, -0.2)

    def test_neutral_text_scores_zero(self):
        self.assertEqual(score_text("the cat sat on the mat").score, 0.0)
        self.assertEqual(score_text("").score, 0.0)

    def test_negation_flips_sign(self):
        positive = score_text("this is bullish").score
        negated = score_text("this is not bullish").score
        self.assertGreater(positive, 0)
        self.assertLess(negated, 0)

    def test_intensifier_amplifies(self):
        plain = score_text("bullish")
        strong = score_text("very bullish")
        self.assertGreater(strong.score, plain.score)

    def test_urls_are_stripped(self):
        with_url = score_text("bullish https://example.com/scam-dump-moon")
        self.assertGreater(with_url.score, 0)

    def test_score_is_bounded(self):
        extreme = score_text("moon " * 200)
        self.assertLessEqual(extreme.score, 1.0)
        self.assertGreaterEqual(extreme.score, -1.0)

    def test_labels(self):
        self.assertEqual(score_text("bullish rally moon").label, "positive")
        self.assertEqual(score_text("crash scam rug").label, "negative")
        self.assertEqual(score_text("the cat sat").label, "neutral")


class TestSentimentAggregation(unittest.TestCase):
    def test_daily_buckets(self):
        docs = [
            ("2021-01-01", "bullish moon"),
            ("2021-01-01", "crash scam"),
            ("2021-01-02", "bullish"),
        ]
        readings = aggregate_daily_sentiment(docs)
        self.assertEqual([r.date for r in readings], ["2021-01-01", "2021-01-02"])
        self.assertEqual(readings[0].documents, 2)
        self.assertEqual(readings[1].score, readings[1].score)  # finite

    def test_confidence_weighting_changes_result(self):
        docs = [
            ("2021-01-01", "bullish " * 10),
            ("2021-01-01", "cat"),
        ]
        plain = aggregate_daily_sentiment(docs, weighting="count")[0].score
        weighted = aggregate_daily_sentiment(docs, weighting="confidence")[0].score
        self.assertNotEqual(plain, weighted)

    def test_combine_with_price_needs_enough_data(self):
        self.assertEqual(combine_with_price([1, 2], [0.1, 0.2]), {})
        result = combine_with_price(
            [100, 101, 102, 103, 104, 105],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        )
        self.assertEqual(result, {}, "constant sentiment has no correlation")

    def test_combine_with_price_correlates(self):
        prices = [100, 110, 121, 133, 146, 161]
        sentiment = [0.9, 0.8, 0.9, 0.7, 0.8, 0.9]
        result = combine_with_price(prices, sentiment)
        self.assertIn("correlation", result)
        self.assertGreaterEqual(result["correlation"], -1.0)
        self.assertLessEqual(result["correlation"], 1.0)


class TestPositionSizing(unittest.TestCase):
    def test_positive_edge_allocates(self):
        sizing = position_size(0.05, 0.05, capital=1000.0, risk_per_trade=0.02)
        self.assertGreater(sizing["cash"], 0)
        self.assertLessEqual(sizing["weight"], 0.25)
        self.assertAlmostEqual(sizing["risk_cash"], sizing["cash"] * 0.05, places=6)

    def test_no_edge_stays_flat(self):
        self.assertEqual(position_size(-0.02, 0.05)["cash"], 0.0)
        self.assertEqual(position_size(0.0, 0.05)["cash"], 0.0)

    def test_weight_is_capped(self):
        huge = position_size(1.0, 0.0001, capital=1000.0, risk_per_trade=0.02, max_weight=0.25)
        self.assertAlmostEqual(huge["weight"], 0.25)

    def test_bad_capital_and_zero_error_are_safe(self):
        self.assertEqual(position_size(0.1, 0.05, capital=0.0)["cash"], 0.0)
        sizing = position_size(0.1, 0.0)
        self.assertGreaterEqual(sizing["cash"], 0.0)


class TestRecommend(unittest.TestCase):
    def test_recommend_excludes_losers(self):
        import math
        from datetime import date, timedelta

        from crypto_prediction.core import Bar, Series
        from crypto_prediction.core.signals import recommend

        def make(symbol, drift):
            bars = []
            price = 100.0
            day = date(2020, 1, 1)
            for i in range(400):
                # A smooth trend: 'linear' extrapolates it cleanly, which is what
                # this test is about. A strong high-frequency wobble would
                # dominate the OLS fit and is covered elsewhere.
                price = price * (1 + drift) * (1 + 0.001 * math.sin(i / 7))
                bars.append(Bar(day + timedelta(days=i), price, price * 1.01, price * 0.99, price, 10.0))
            return Series(symbol=symbol, bars=bars, source="synthetic")

        datasets = {"Up": make("Up", 0.008), "Down": make("Down", -0.008)}
        # 'linear' extrapolates the trend, so it can produce a non-zero edge on
        # synthetic trending data; 'naive' always predicts zero change by design.
        proposals = recommend(datasets, "linear", lookback=20, min_train=100, min_confidence=0.0)
        symbols = [p["symbol"] for p in proposals]
        self.assertIn("Up", symbols)
        self.assertNotIn("Down", symbols)
        for proposal in proposals:
            self.assertIn("allocation", proposal)


if __name__ == "__main__":
    unittest.main(verbosity=2)
