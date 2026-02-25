#!/usr/bin/env python3
"""
Ingest Reddit data for July 2025 via PRAW API.

Writes to:
  - reddit_posts: raw posts/comments with ticker mentions
  - reddit_daily_mentions: daily aggregates with sentiment

Usage:
  python scripts/ingest_reddit_july_2025.py --write-db
  python scripts/ingest_reddit_july_2025.py --symbols "AAPL,NVDA,TSLA" --write-db
  python scripts/ingest_reddit_july_2025.py --start 2025-07-15 --end 2025-07-20 --write-db

Environment variables required:
  REDDIT_CLIENT_ID
  REDDIT_CLIENT_SECRET
  REDDIT_USER_AGENT (optional, defaults to "AWET/1.0")
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg
import praw
from praw.models import Submission

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]

# Hard-coded date limits for July 2025
JULY_2025_START = datetime(2025, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
JULY_2025_END = datetime(2025, 8, 1, 0, 0, 0, tzinfo=timezone.utc)

# Default subreddits to search
DEFAULT_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options"]

# Sentiment lexicon (simple placeholder - swap for FinBERT later)
POSITIVE_WORDS = {
    "buy", "long", "bullish", "moon", "rocket", "calls", "pump", "growth",
    "undervalued", "breakout", "upside", "profit", "gains", "rally", "surge",
    "strong", "beat", "exceeded", "outperform", "winner", "diamond", "hands",
}
NEGATIVE_WORDS = {
    "sell", "short", "bearish", "crash", "dump", "puts", "overvalued", "drop",
    "downside", "loss", "tank", "weak", "miss", "missed", "underperform",
    "loser", "bag", "holder", "fear", "panic", "recession", "correction",
}


@dataclass
class RedditPost:
    """Represents a Reddit post/comment with ticker mention."""
    post_id: str
    ticker: str
    created_utc: datetime
    subreddit: str
    title: str
    body: str
    score: int
    num_comments: int
    post_type: str  # "submission" or "comment"
    source: str  # "praw_api"
    permalink: str = ""
    author: str = ""
    raw_json: dict = field(default_factory=dict)


@dataclass
class DailyAggregate:
    """Daily aggregated mentions for a ticker."""
    day: datetime
    ticker: str
    mentions_count: int = 0
    sentiment_sum: float = 0.0
    weighted_sentiment_sum: float = 0.0
    weight_sum: float = 0.0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0

    @property
    def sentiment_mean(self) -> float:
        if self.mentions_count == 0:
            return 0.0
        return self.sentiment_sum / self.mentions_count

    @property
    def sentiment_weighted(self) -> float:
        if self.weight_sum == 0:
            return 0.0
        return self.weighted_sentiment_sum / self.weight_sum

    @property
    def positive_ratio(self) -> float:
        if self.mentions_count == 0:
            return 0.0
        return self.positive_count / self.mentions_count

    @property
    def negative_ratio(self) -> float:
        if self.mentions_count == 0:
            return 0.0
        return self.negative_count / self.mentions_count


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Scoring (Placeholder - swap for FinBERT later)
# ─────────────────────────────────────────────────────────────────────────────

class SentimentScorer:
    """
    Simple lexicon-based sentiment scorer.
    
    Replace this class with FinBERT for production:
    
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        class FinBERTScorer(SentimentScorer):
            def __init__(self):
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            
            def score(self, text: str) -> tuple[float, str]:
                # Implement FinBERT inference
                ...
    """
    
    def __init__(self):
        self.positive_words = POSITIVE_WORDS
        self.negative_words = NEGATIVE_WORDS
    
    def score(self, text: str) -> tuple[float, str]:
        """
        Score text sentiment.
        
        Returns:
            (sentiment_score, label) where:
            - sentiment_score: float in [-1, 1]
            - label: "positive", "negative", or "neutral"
        """
        if not text:
            return 0.0, "neutral"
        
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        pos_count = len(words & self.positive_words)
        neg_count = len(words & self.negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0, "neutral"
        
        # Score in [-1, 1]
        score = (pos_count - neg_count) / total
        
        if score > 0.1:
            return score, "positive"
        elif score < -0.1:
            return score, "negative"
        else:
            return score, "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# Ticker Detection
# ─────────────────────────────────────────────────────────────────────────────

def load_universe_symbols() -> list[str]:
    """Load ticker symbols from universe.csv."""
    universe_path = REPO_ROOT / "config" / "universe.csv"
    if not universe_path.exists():
        print(f"⚠️  Warning: {universe_path} not found")
        return []
    
    symbols = []
    with open(universe_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker") or row.get("symbol") or row.get("Ticker")
            if ticker:
                symbols.append(ticker.upper().strip())
    
    return symbols


def find_tickers_in_text(text: str, valid_tickers: set[str]) -> list[str]:
    """
    Find stock tickers mentioned in text.
    
    Looks for:
    - $TICKER format (e.g., $AAPL)
    - Standalone uppercase tickers (e.g., AAPL)
    """
    if not text:
        return []
    
    found = set()
    
    # Match $TICKER pattern
    cashtag_pattern = r'\$([A-Z]{1,5})\b'
    for match in re.finditer(cashtag_pattern, text):
        ticker = match.group(1)
        if ticker in valid_tickers:
            found.add(ticker)
    
    # Match standalone tickers (less reliable, but useful)
    # Only match if surrounded by word boundaries and text is relatively short
    for ticker in valid_tickers:
        if len(ticker) >= 2:  # Skip single-letter tickers
            pattern = rf'\b{re.escape(ticker)}\b'
            if re.search(pattern, text):
                found.add(ticker)
    
    return list(found)


# ─────────────────────────────────────────────────────────────────────────────
# Reddit API Client
# ─────────────────────────────────────────────────────────────────────────────

def create_reddit_client() -> praw.Reddit:
    """Create authenticated Reddit client."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "AWET/1.0 (by /u/awet_bot)")
    
    if not client_id or not client_secret:
        raise ValueError(
            "Missing Reddit API credentials. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
        )
    
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def fetch_subreddit_posts(
    reddit: praw.Reddit,
    subreddit_name: str,
    start_date: datetime,
    end_date: datetime,
    valid_tickers: set[str],
    max_posts: int = 1000,
    scorer: SentimentScorer | None = None,
) -> tuple[list[RedditPost], dict[str, dict[str, DailyAggregate]]]:
    """
    Fetch posts from a subreddit that mention valid tickers.
    
    Returns:
        (posts, daily_aggregates) where daily_aggregates is {ticker: {date_str: DailyAggregate}}
    """
    scorer = scorer or SentimentScorer()
    posts: list[RedditPost] = []
    daily_aggs: dict[str, dict[str, DailyAggregate]] = defaultdict(dict)
    
    subreddit = reddit.subreddit(subreddit_name)
    
    # PRAW's search is limited - we'll use new/hot/top with time filter
    # Note: Reddit API has limitations on historical access
    print(f"  Fetching from r/{subreddit_name}...")
    
    fetched = 0
    in_range = 0
    
    # Try multiple listing types
    for listing_type in ["new", "hot", "top"]:
        if fetched >= max_posts:
            break
            
        try:
            if listing_type == "top":
                submissions = subreddit.top(time_filter="month", limit=min(max_posts, 1000))
            elif listing_type == "hot":
                submissions = subreddit.hot(limit=min(max_posts, 1000))
            else:
                submissions = subreddit.new(limit=min(max_posts, 1000))
            
            for submission in submissions:
                fetched += 1
                if fetched > max_posts:
                    break
                
                # Convert timestamp
                created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                
                # Check date range
                if created < start_date or created >= end_date:
                    continue
                
                in_range += 1
                
                # Combine title and selftext for analysis
                full_text = f"{submission.title} {submission.selftext or ''}"
                
                # Find tickers
                tickers = find_tickers_in_text(full_text, valid_tickers)
                if not tickers:
                    continue
                
                # Score sentiment
                sentiment_score, sentiment_label = scorer.score(full_text)
                
                # Create post for each ticker mentioned
                for ticker in tickers:
                    post = RedditPost(
                        post_id=submission.id,
                        ticker=ticker,
                        created_utc=created,
                        subreddit=subreddit_name,
                        title=submission.title[:500],
                        body=(submission.selftext or "")[:2000],
                        score=submission.score,
                        num_comments=submission.num_comments,
                        post_type="submission",
                        source="praw_api",
                        permalink=f"https://reddit.com{submission.permalink}",
                        author=str(submission.author) if submission.author else "[deleted]",
                        raw_json={
                            "id": submission.id,
                            "score": submission.score,
                            "upvote_ratio": getattr(submission, "upvote_ratio", 0),
                            "sentiment_score": sentiment_score,
                            "sentiment_label": sentiment_label,
                        },
                    )
                    posts.append(post)
                    
                    # Update daily aggregate
                    day_key = created.strftime("%Y-%m-%d")
                    if day_key not in daily_aggs[ticker]:
                        daily_aggs[ticker][day_key] = DailyAggregate(
                            day=datetime.strptime(day_key, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                            ticker=ticker,
                        )
                    
                    agg = daily_aggs[ticker][day_key]
                    agg.mentions_count += 1
                    agg.sentiment_sum += sentiment_score
                    
                    # Weight by score (engagement)
                    weight = max(1, abs(submission.score))
                    agg.weighted_sentiment_sum += sentiment_score * weight
                    agg.weight_sum += weight
                    
                    if sentiment_label == "positive":
                        agg.positive_count += 1
                    elif sentiment_label == "negative":
                        agg.negative_count += 1
                    else:
                        agg.neutral_count += 1
                        
        except Exception as e:
            print(f"    ⚠️  Error fetching {listing_type}: {e}")
            continue
    
    print(f"    Fetched {fetched} posts, {in_range} in date range, {len(posts)} with ticker mentions")
    
    return posts, dict(daily_aggs)


# ─────────────────────────────────────────────────────────────────────────────
# Database Operations
# ─────────────────────────────────────────────────────────────────────────────

def _build_dsn() -> str:
    return (
        f"postgresql://{os.getenv('POSTGRES_USER', 'awet')}:{os.getenv('POSTGRES_PASSWORD', 'awet')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5433')}"
        f"/{os.getenv('POSTGRES_DB', 'awet')}"
    )


async def write_posts_to_db(conn: asyncpg.Connection, posts: list[RedditPost]) -> int:
    """Write posts to reddit_posts table with UPSERT."""
    if not posts:
        return 0
    
    inserted = 0
    for post in posts:
        try:
            await conn.execute(
                """
                INSERT INTO reddit_posts (
                    post_id, ticker, created_utc, subreddit, title, body,
                    score, num_comments, post_type, source, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
                ON CONFLICT (post_id, ticker) DO UPDATE SET
                    score = EXCLUDED.score,
                    num_comments = EXCLUDED.num_comments,
                    created_at = NOW()
                """,
                post.post_id,
                post.ticker,
                post.created_utc,
                post.subreddit,
                post.title,
                post.body,
                post.score,
                post.num_comments,
                post.post_type,
                post.source,
            )
            inserted += 1
        except Exception as e:
            print(f"    ⚠️  Error inserting post {post.post_id}: {e}")
    
    return inserted


async def write_daily_aggregates_to_db(
    conn: asyncpg.Connection,
    daily_aggs: dict[str, dict[str, DailyAggregate]],
) -> int:
    """Write daily aggregates to reddit_daily_mentions with UPSERT."""
    inserted = 0
    
    for ticker, days in daily_aggs.items():
        for day_key, agg in days.items():
            try:
                await conn.execute(
                    """
                    INSERT INTO reddit_daily_mentions (
                        day, ticker, mentions_count,
                        sentiment_mean, sentiment_weighted,
                        positive_ratio, negative_ratio,
                        created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    ON CONFLICT (day, ticker) DO UPDATE SET
                        mentions_count = reddit_daily_mentions.mentions_count + EXCLUDED.mentions_count,
                        sentiment_mean = (
                            reddit_daily_mentions.sentiment_mean * reddit_daily_mentions.mentions_count +
                            EXCLUDED.sentiment_mean * EXCLUDED.mentions_count
                        ) / NULLIF(reddit_daily_mentions.mentions_count + EXCLUDED.mentions_count, 0),
                        sentiment_weighted = EXCLUDED.sentiment_weighted,
                        positive_ratio = EXCLUDED.positive_ratio,
                        negative_ratio = EXCLUDED.negative_ratio,
                        created_at = NOW()
                    """,
                    agg.day.date(),
                    agg.ticker,
                    agg.mentions_count,
                    round(agg.sentiment_mean, 6),
                    round(agg.sentiment_weighted, 6),
                    round(agg.positive_ratio, 6),
                    round(agg.negative_ratio, 6),
                )
                inserted += 1
            except Exception as e:
                print(f"    ⚠️  Error inserting daily agg {ticker}/{day_key}: {e}")
    
    return inserted


async def update_features_tft_reddit(
    conn: asyncpg.Connection,
    start_date: datetime,
    end_date: datetime,
) -> int:
    """
    Update features_tft with Reddit data from reddit_daily_mentions.
    
    Joins on (ticker, ts::date) and updates reddit_* columns.
    """
    result = await conn.execute(
        """
        UPDATE features_tft f
        SET
            reddit_mentions_count = COALESCE(r.mentions_count, 0),
            reddit_sentiment_mean = COALESCE(r.sentiment_mean, 0),
            reddit_sentiment_weighted = COALESCE(r.sentiment_weighted, 0),
            reddit_positive_ratio = COALESCE(r.positive_ratio, 0),
            reddit_negative_ratio = COALESCE(r.negative_ratio, 0)
        FROM reddit_daily_mentions r
        WHERE f.ticker = r.ticker
          AND f.ts::date = r.day
          AND f.ts >= $1
          AND f.ts < $2
        """,
        start_date,
        end_date,
    )
    
    # Extract count from result string like "UPDATE 123"
    count = 0
    if result:
        parts = result.split()
        if len(parts) >= 2:
            try:
                count = int(parts[1])
            except ValueError:
                pass
    
    return count


async def verify_ingestion(
    conn: asyncpg.Connection,
    start_date: datetime,
    end_date: datetime,
) -> None:
    """Print verification stats."""
    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Count posts
    posts_count = await conn.fetchval(
        """
        SELECT COUNT(*) FROM reddit_posts
        WHERE created_utc >= $1 AND created_utc < $2
        """,
        start_date,
        end_date,
    )
    print(f"  reddit_posts (July 2025):        {posts_count:,}")
    
    # Count daily aggregates
    daily_count = await conn.fetchval(
        """
        SELECT COUNT(*) FROM reddit_daily_mentions
        WHERE day >= $1::date AND day < $2::date
        """,
        start_date,
        end_date,
    )
    print(f"  reddit_daily_mentions (July 2025): {daily_count:,}")
    
    # Count features with Reddit data
    features_count = await conn.fetchval(
        """
        SELECT COUNT(*) FROM features_tft
        WHERE ts >= $1 AND ts < $2 AND reddit_mentions_count > 0
        """,
        start_date,
        end_date,
    )
    print(f"  features_tft with Reddit data:    {features_count:,}")
    
    # Sample rows
    print()
    print("Sample reddit_daily_mentions rows:")
    rows = await conn.fetch(
        """
        SELECT day, ticker, mentions_count, 
               ROUND(sentiment_mean::numeric, 3) as sent_mean,
               ROUND(positive_ratio::numeric, 3) as pos_ratio
        FROM reddit_daily_mentions
        WHERE day >= $1::date AND day < $2::date
        ORDER BY mentions_count DESC
        LIMIT 10
        """,
        start_date,
        end_date,
    )
    for row in rows:
        print(f"    {row['day']} {row['ticker']}: {row['mentions_count']} mentions, "
              f"sent={row['sent_mean']}, pos_ratio={row['pos_ratio']}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest Reddit data for July 2025 via PRAW API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbols (default: load from universe.csv)",
    )
    parser.add_argument(
        "--start", type=str, default="2025-07-01",
        help="Start date YYYY-MM-DD (default: 2025-07-01, clamped to July 2025)",
    )
    parser.add_argument(
        "--end", type=str, default="2025-07-31",
        help="End date YYYY-MM-DD (default: 2025-07-31, clamped to July 2025)",
    )
    parser.add_argument(
        "--subreddits", type=str, default=None,
        help=f"Comma-separated subreddits (default: {','.join(DEFAULT_SUBREDDITS)})",
    )
    parser.add_argument(
        "--max-posts-per-sub", type=int, default=1000,
        help="Max posts to fetch per subreddit (default: 1000)",
    )
    parser.add_argument(
        "--write-db", action="store_true",
        help="Write to database (default: dry-run)",
    )
    parser.add_argument(
        "--update-features", action="store_true",
        help="Also update features_tft with Reddit data",
    )
    
    args = parser.parse_args()
    
    # Parse dates
    try:
        start_date = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        # Add a day to end date to make it inclusive
        end_date = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        end_date = end_date.replace(hour=23, minute=59, second=59)
    except ValueError as e:
        print(f"❌ Invalid date format: {e}")
        return 1
    
    # Clamp to July 2025
    start_date = max(start_date, JULY_2025_START)
    end_date = min(end_date, JULY_2025_END)
    
    if start_date >= end_date:
        print(f"❌ Invalid date range: {start_date} to {end_date}")
        return 1
    
    # Load symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = load_universe_symbols()
    
    if not symbols:
        print("❌ No symbols to search for. Provide --symbols or create config/universe.csv")
        return 1
    
    valid_tickers = set(symbols)
    
    # Parse subreddits
    if args.subreddits:
        subreddits = [s.strip() for s in args.subreddits.split(",")]
    else:
        subreddits = DEFAULT_SUBREDDITS
    
    print()
    print("=" * 60)
    print("REDDIT INGESTION - JULY 2025")
    print("=" * 60)
    print(f"  Date range:    {start_date.date()} to {end_date.date()}")
    print(f"  Symbols:       {len(symbols)} ({', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''})")
    print(f"  Subreddits:    {', '.join(subreddits)}")
    print(f"  Max posts/sub: {args.max_posts_per_sub}")
    print(f"  Write to DB:   {'Yes' if args.write_db else 'No (dry-run)'}")
    print()
    
    # Check Reddit credentials
    try:
        reddit = create_reddit_client()
        # Test connection
        reddit.user.me()  # Will be None for read-only, but shouldn't error
        print("✅ Reddit API connection OK")
    except Exception as e:
        print(f"❌ Reddit API error: {e}")
        print("   Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables")
        return 1
    
    # Fetch posts
    scorer = SentimentScorer()
    all_posts: list[RedditPost] = []
    all_daily_aggs: dict[str, dict[str, DailyAggregate]] = defaultdict(dict)
    
    print()
    print("Fetching posts...")
    for subreddit_name in subreddits:
        posts, daily_aggs = fetch_subreddit_posts(
            reddit=reddit,
            subreddit_name=subreddit_name,
            start_date=start_date,
            end_date=end_date,
            valid_tickers=valid_tickers,
            max_posts=args.max_posts_per_sub,
            scorer=scorer,
        )
        all_posts.extend(posts)
        
        # Merge daily aggregates
        for ticker, days in daily_aggs.items():
            for day_key, agg in days.items():
                if day_key not in all_daily_aggs[ticker]:
                    all_daily_aggs[ticker][day_key] = agg
                else:
                    # Merge aggregates
                    existing = all_daily_aggs[ticker][day_key]
                    existing.mentions_count += agg.mentions_count
                    existing.sentiment_sum += agg.sentiment_sum
                    existing.weighted_sentiment_sum += agg.weighted_sentiment_sum
                    existing.weight_sum += agg.weight_sum
                    existing.positive_count += agg.positive_count
                    existing.negative_count += agg.negative_count
                    existing.neutral_count += agg.neutral_count
    
    print()
    print(f"Total: {len(all_posts)} posts, {sum(len(d) for d in all_daily_aggs.values())} daily aggregates")
    
    # Write to database
    if args.write_db:
        print()
        print("Writing to database...")
        
        dsn = _build_dsn()
        conn = await asyncpg.connect(dsn=dsn)
        
        try:
            posts_inserted = await write_posts_to_db(conn, all_posts)
            print(f"  Inserted/updated {posts_inserted} rows in reddit_posts")
            
            daily_inserted = await write_daily_aggregates_to_db(conn, all_daily_aggs)
            print(f"  Inserted/updated {daily_inserted} rows in reddit_daily_mentions")
            
            if args.update_features:
                features_updated = await update_features_tft_reddit(conn, start_date, end_date)
                print(f"  Updated {features_updated} rows in features_tft")
            
            # Verify
            await verify_ingestion(conn, start_date, end_date)
            
        finally:
            await conn.close()
    else:
        print()
        print("Dry-run complete. Use --write-db to persist data.")
        
        # Show sample
        if all_posts:
            print()
            print("Sample posts:")
            for post in all_posts[:5]:
                print(f"  {post.created_utc.date()} {post.ticker} r/{post.subreddit}: {post.title[:60]}...")
    
    print()
    print("✅ Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
