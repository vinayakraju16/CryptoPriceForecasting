"""Fetch Reddit posts safely and score their sentiment.

The original notebook had live Reddit API credentials hardcoded:

    reddit = praw.Reddit(client_id="h1oT9ECrvYgR8oipbCwXRg",
                         client_secret="wq3QJIj0mcSB9TAIspKMG7pXTbhpLQ", ...)

Those values are now revoked-by-policy and must never be committed. This script
reads them from the environment instead and refuses to run without them.

Usage::

    export REDDIT_CLIENT_ID=...   REDDIT_CLIENT_SECRET=...   REDDIT_USER_AGENT="crypto-forecast/2.0"
    python scripts/fetch_reddit_sentiment.py --subreddit CryptoCurrency --limit 100 \
        --out data/reddit_sentiment.csv

Requires ``praw`` (``pip install praw``); the scoring itself is the stdlib
lexicon in ``crypto_prediction.core.sentiment``, so no torch/transformers needed.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_prediction.core.sentiment import aggregate_daily_sentiment, score_text  # noqa: E402

REQUIRED_ENV = ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT")


def build_client():
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise SystemExit(
            "error: missing environment variable(s): "
            + ", ".join(missing)
            + "\nSet them in your shell or a local .env file (see .env.example). "
            "Never commit credentials."
        )
    try:
        import praw
    except ImportError:
        raise SystemExit("error: praw is not installed. Run: pip install praw")

    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )


def fetch(subreddit: str, limit: int, time_filter: str = "day", listing: str = "top"):
    reddit = build_client()
    source = getattr(reddit.subreddit(subreddit), listing)
    posts = []
    for post in source(time_filter=time_filter, limit=limit):
        created = datetime.fromtimestamp(float(post.created_utc), tz=timezone.utc)
        text = f"{post.title} {post.selftext or ''}".strip()
        result = score_text(text)
        posts.append(
            {
                "id": post.id,
                "subreddit": subreddit,
                "created_utc": created.isoformat(timespec="seconds"),
                "date": created.date().isoformat(),
                "score": post.score,
                "num_comments": post.num_comments,
                "sentiment": round(result.score, 4),
                "sentiment_label": result.label,
                "matched_terms": result.matched,
                "title": post.title.replace("\n", " ")[:300],
            }
        )
    return posts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and score Reddit sentiment (credentials from env).")
    parser.add_argument("--subreddit", default="CryptoCurrency")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--time-filter", default="day", choices=["hour", "day", "week", "month", "year", "all"])
    parser.add_argument("--listing", default="top", choices=["top", "new", "hot", "rising"])
    parser.add_argument("--out", default="data/reddit_sentiment.csv")
    parser.add_argument("--daily-out", default=None, help="also write per-day aggregated sentiment")
    args = parser.parse_args(argv)

    posts = fetch(args.subreddit, args.limit, args.time_filter, args.listing)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(posts[0].keys()) if posts else ["id"])
        writer.writeheader()
        writer.writerows(posts)
    print(f"wrote {args.out} ({len(posts)} posts)")

    if args.daily_out:
        readings = aggregate_daily_sentiment((p["date"], p["title"]) for p in posts)
        with open(args.daily_out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["date", "score", "documents", "positive", "negative", "neutral"])
            writer.writeheader()
            for reading in readings:
                writer.writerow(reading.to_dict())
        print(f"wrote {args.daily_out} ({len(readings)} day(s))")

    if posts:
        mean = sum(p["sentiment"] for p in posts) / len(posts)
        print(f"mean sentiment: {mean:+.3f} over {len(posts)} post(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
