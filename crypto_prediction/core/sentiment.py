"""Sentiment scoring without a heavy dependency chain.

The original repository scored Reddit text with BERT embeddings plus a small
PyTorch classifier, and the notebook that produced the model had *live Reddit
API credentials committed in plaintext*. Two problems:

1. The credentials are a leak. They now come from the environment only; see
   ``SECURITY.md`` and ``scripts/fetch_reddit_sentiment.py``.
2. Requiring torch + transformers to get a sentiment number makes the feature
   unusable in CI and on small hosts. This module provides a lexicon-based
   scorer that runs on the standard library, so the pipeline is complete
   end-to-end without the heavy stack. The BERT path stays available in the
   notebooks for higher accuracy.

The lexicon is deliberately small and finance-flavoured - crypto chatter is not
general English, and words like "moon", "rug" and "dump" carry the signal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

__all__ = [
    "score_text",
    "score_documents",
    "aggregate_daily_sentiment",
    "SentimentResult",
    "SentimentReading",
]

_TOKEN_RE = re.compile(r"[a-z']+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NEGATORS = {"not", "no", "never", "cannot", "can't", "don't", "doesn't", "isn't", "won't", "without"}

# Scores are in [-3, 3]; magnitude encodes intensity.
_POSITIVE: Dict[str, float] = {
    "bullish": 3.0, "bull": 2.5, "moon": 2.5, "mooning": 2.5, "rally": 2.0,
    "surge": 2.0, "soar": 2.5, "pump": 1.5, "gain": 1.5, "gains": 1.5,
    "profit": 2.0, "profitability": 2.0, "adoption": 2.0, "upgrade": 1.5,
    "breakout": 2.0, "support": 1.0, "buy": 1.5, "accumulate": 1.5,
    "hodl": 1.5, "hold": 1.0, "long": 1.0, "green": 1.0, "ath": 2.5,
    "optimistic": 2.0, "bullrun": 2.5, "upside": 1.5, "recover": 1.0,
    "recovery": 1.0, "partnership": 1.5, "launch": 1.0, "approval": 2.0,
    "good": 1.0, "great": 1.5, "excellent": 2.0, "strong": 1.5, "win": 1.5,
    "record": 1.5, "high": 1.0, "boost": 1.5, "confident": 1.5, "promising": 1.5,
}

_NEGATIVE: Dict[str, float] = {
    "bearish": -3.0, "bear": -2.5, "crash": -3.0, "dump": -2.5, "dumping": -2.5,
    "rug": -3.0, "rugged": -3.0, "scam": -3.0, "fraud": -3.0, "hack": -3.0,
    "hacked": -3.0, "exploit": -3.0, "stolen": -3.0, "loss": -2.0, "losses": -2.0,
    "plunge": -2.5, "plummet": -2.5, "tank": -2.0, "drop": -1.5, "fall": -1.5,
    "decline": -1.5, "sell": -1.5, "short": -1.0, "red": -1.0, "panic": -2.5,
    "fear": -2.0, "lawsuit": -2.5, "sec": -1.0, "ban": -2.5, "banned": -2.5,
    "regulation": -1.0, "delay": -1.5, "delisted": -2.5, "bankrupt": -3.0,
    "insolvent": -3.0, "liquidation": -2.5, "liquidated": -2.5, "risk": -1.0,
    "risky": -1.5, "volatile": -0.5, "bubble": -1.5, "overvalued": -1.5,
    "bad": -1.0, "terrible": -2.0, "awful": -2.0, "weak": -1.5, "dead": -2.5,
    "ponzi": -3.0, "warning": -1.5, "worry": -1.5, "worried": -1.5, "unsustainable": -2.0,
    "down": -1.0, "low": -1.0, "dip": -0.5, "correction": -1.0, "capitulation": -2.5,
}

_INTENSIFIERS = {"very": 1.5, "extremely": 1.8, "massively": 1.8, "insanely": 1.8, "super": 1.5, "so": 1.3}


@dataclass
class SentimentResult:
    """Score for a single document."""

    score: float          # normalised to [-1, 1]
    raw: float            # unnormalised lexicon sum
    matched: int          # tokens that hit the lexicon
    tokens: int
    label: str

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "raw": self.raw,
            "matched": self.matched,
            "tokens": self.tokens,
            "label": self.label,
        }


@dataclass
class SentimentReading:
    """One dated sentiment observation."""

    date: str
    score: float
    documents: int
    positive: int = 0
    negative: int = 0
    neutral: int = 0

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "score": self.score,
            "documents": self.documents,
            "positive": self.positive,
            "negative": self.negative,
            "neutral": self.neutral,
        }


def _label(score: float) -> str:
    if score >= 0.15:
        return "positive"
    if score <= -0.15:
        return "negative"
    return "neutral"


def score_text(text: str, *, normalize: bool = True) -> SentimentResult:
    """Lexicon sentiment for one document.

    Handles negation within a short window ("not bullish" flips sign) and
    intensifiers ("very bullish"), which a plain bag-of-words score misses.
    """
    if not text:
        return SentimentResult(0.0, 0.0, 0, 0, "neutral")

    cleaned = _URL_RE.sub(" ", str(text).lower())
    tokens = _TOKEN_RE.findall(cleaned)

    raw = 0.0
    matched = 0
    for i, token in enumerate(tokens):
        value = _POSITIVE.get(token, _NEGATIVE.get(token, 0.0))
        if value == 0.0:
            continue
        matched += 1
        if i > 0 and tokens[i - 1] in _NEGATORS:
            value = -value * 0.8
        if i > 0 and tokens[i - 1] in _INTENSIFIERS:
            value *= _INTENSIFIERS[tokens[i - 1]]
        raw += value

    score = raw
    if normalize and matched:
        # Soft saturation: many hits should not linearly dominate.
        score = math.tanh(raw / (2.0 * math.sqrt(matched)))
    score = max(-1.0, min(1.0, score))

    return SentimentResult(score, raw, matched, len(tokens), _label(score))


def score_documents(documents: Sequence[str]) -> List[SentimentResult]:
    return [score_text(doc) for doc in documents]


def aggregate_daily_sentiment(
    dated_documents: Iterable[tuple],
    *,
    weighting: str = "count",
) -> List[SentimentReading]:
    """Average sentiment per calendar day.

    ``dated_documents`` yields ``(date, text)`` pairs; ``date`` may be a
    ``datetime.date``/``datetime`` or an ISO string. ``weighting="count"`` is a
    plain mean; ``weighting="confidence"`` weights documents by how many lexicon
    hits they produced, so empty noise does not dilute a strong signal.
    """
    buckets: Dict[str, List[SentimentResult]] = {}
    for raw_date, text in dated_documents:
        key = str(raw_date)[:10]
        buckets.setdefault(key, []).append(score_text(text))

    readings: List[SentimentReading] = []
    for day in sorted(buckets):
        results = buckets[day]
        if weighting == "confidence" and any(r.matched for r in results):
            total_weight = sum(max(r.matched, 1) for r in results)
            mean = sum(r.score * max(r.matched, 1) for r in results) / total_weight
        else:
            mean = sum(r.score for r in results) / len(results)
        readings.append(
            SentimentReading(
                date=day,
                score=mean,
                documents=len(results),
                positive=sum(1 for r in results if r.label == "positive"),
                negative=sum(1 for r in results if r.label == "negative"),
                neutral=sum(1 for r in results if r.label == "neutral"),
            )
        )
    return readings


def combine_with_price(prices: Sequence[float], sentiment: Sequence[float]) -> Dict[str, float]:
    """Correlation between sentiment and next-day return.

    Returns ``{}`` when the inputs cannot support a meaningful correlation, so
    callers never report a number from three data points as if it were signal.
    """
    n = min(len(prices), len(sentiment))
    if n < 4:
        return {}
    prices = [float(p) for p in prices[-n:]]
    sentiment = [float(s) for s in sentiment[-n:]]

    returns = [prices[i] / prices[i - 1] - 1.0 for i in range(1, n) if prices[i - 1]]
    aligned = sentiment[: len(returns)]
    if len(returns) < 4:
        return {}

    mean_r = sum(returns) / len(returns)
    mean_s = sum(aligned) / len(aligned)
    cov = sum((r - mean_r) * (s - mean_s) for r, s in zip(returns, aligned)) / len(returns)
    var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    var_s = sum((s - mean_s) ** 2 for s in aligned) / len(aligned)
    if var_r <= 0 or var_s <= 0:
        return {}
    return {
        "correlation": cov / math.sqrt(var_r * var_s),
        "mean_return_after_positive_sentiment": (
            sum(r for r, s in zip(returns, aligned) if s > 0.15)
            / max(1, sum(1 for s in aligned if s > 0.15))
        ),
        "mean_return_after_negative_sentiment": (
            sum(r for r, s in zip(returns, aligned) if s < -0.15)
            / max(1, sum(1 for s in aligned if s < -0.15))
        ),
        "n": len(returns),
    }
