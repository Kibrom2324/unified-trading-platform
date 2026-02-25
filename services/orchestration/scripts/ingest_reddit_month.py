#!/usr/bin/env python3
"""
Ingest Reddit data for a specific month into PostgreSQL.

Sources (in priority order):
1. Local monthly dump files (RS_YYYY-MM.zst and RC_YYYY-MM.zst)
2. Reddit API via PRAW (use --use-api; limited historical access)

Writes to:
- reddit_posts: Raw posts/comments with ticker mentions
- reddit_daily_mentions: Daily aggregates with sentiment

Usage:
    # ──────────────────────────────────────────────────────────────────────
    # REDDIT API MODE (--use-api)
    # ──────────────────────────────────────────────────────────────────────
    # First, set environment variables:
    #   export REDDIT_CLIENT_ID=your_client_id
    #   export REDDIT_CLIENT_SECRET=your_client_secret
    #   export REDDIT_USERNAME=your_username
    #   export REDDIT_PASSWORD=your_password
    #   export REDDIT_USER_AGENT="AWET/1.0 by u/your_username"
    #
    # Then run:
    python scripts/ingest_reddit_month.py --month 2025-07 --use-api --write-db --max-items 50000
    
    # With custom subreddits:
    python scripts/ingest_reddit_month.py --month 2025-07 --use-api --write-db \\
        --subreddits "wallstreetbets,stocks,investing" --max-items 10000
    
    # Dry-run (no DB writes):
    python scripts/ingest_reddit_month.py --month 2025-07 --use-api --dry-run --max-items 1000
    
    # ──────────────────────────────────────────────────────────────────────
    # LOCAL DUMP MODE (default, preferred for historical data)
    # ──────────────────────────────────────────────────────────────────────
    python scripts/ingest_reddit_month.py --month 2025-07 --write-db
    
    # With explicit paths:
    python scripts/ingest_reddit_month.py --month 2025-07 \\
        --submissions-zst /home/kironix/train/reddit/submissions/RS_2025-07.zst \\
        --comments-zst /home/kironix/train/reddit/comments/RC_2025-07.zst \\
        --write-db
    
    # Full ingest with feature update:
    python scripts/ingest_reddit_month.py --month 2025-07 --write-db --update-features

Environment variables for --use-api:
    REDDIT_CLIENT_ID      - Reddit app client ID (required)
    REDDIT_CLIENT_SECRET  - Reddit app client secret (required)
    REDDIT_USERNAME       - Reddit username for script auth (required)
    REDDIT_PASSWORD       - Reddit password for script auth (required)
    REDDIT_USER_AGENT     - User agent string (default: AWET/1.0)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import asyncpg
import structlog

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
REDDIT_DIR = Path("/home/kironix/train/reddit")
UNIVERSE_FILE = REPO_ROOT / "config" / "universe.csv"

# Default tickers if universe.csv not found
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOG", "META", "AMZN"]

# Company name to ticker mapping (for matching "Apple" -> AAPL, etc.)
COMPANY_NAME_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "google": "GOOG",
    "alphabet": "GOOG",
    "meta": "META",
    "facebook": "META",
    "amazon": "AMZN",
}

# Subreddits to search when using API
DEFAULT_API_SUBREDDITS = [
    "wallstreetbets", "stocks", "investing", "options",
    "SecurityAnalysis", "StockMarket"
]

# Database settings
DB_BATCH_SIZE = 1000
REDDIT_POSTS_TABLE = "reddit_posts"
REDDIT_DAILY_TABLE = "reddit_daily_mentions"

# Progress logging interval
LOG_INTERVAL = 100000

# Ticker extraction pattern
TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b")
EXCLUDED_WORDS = {
    "I", "A", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN", "IS", "IT",
    "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
    "AM", "AN", "AS", "DD", "EV", "FA", "TA", "AI", "CEO", "CFO", "CTO",
    "IPO", "ETF", "GDP", "SEC", "FBI", "CIA", "USA", "UK", "EU", "NYC",
    "LA", "SF", "TX", "CA", "NY", "FL", "PM", "AM", "EST", "PST", "UTC",
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HAS",
    "HER", "WAS", "ONE", "OUR", "OUT", "HIS", "HAS", "HAD", "GET", "GOT",
    "NOW", "NEW", "OLD", "BIG", "TOP", "LOW", "ATH", "ATL", "IMO", "TBH",
    "LOL", "OMG", "WTF", "FYI", "BTW", "YOLO", "FOMO", "HODL", "TLDR",
    "EDIT", "POST", "LINK", "PUTS", "CALL", "LONG", "SHORT", "HOLD",
    "SELL", "MOON", "CASH", "GAIN", "LOSS", "PLAY", "MOVE", "WEEK",
    "TODAY", "STOCK", "SHARE", "PRICE", "VALUE", "MONEY", "TRADE",
}

logger = structlog.get_logger("ingest_reddit_month")


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RedditPost:
    """Parsed Reddit post/comment."""
    post_id: str
    ticker: str
    created_utc: datetime
    subreddit: str
    title: str
    body: str
    score: int
    num_comments: int
    post_type: str  # "submissions" or "comments"
    source: str  # "reddit_dump" or "praw_api"
    sentiment: float = 0.0  # -1 to 1


@dataclass
class DailyAggregate:
    """Daily ticker mention aggregate."""
    day: datetime
    ticker: str
    mentions_count: int = 0
    sentiment_sum: float = 0.0
    weighted_sentiment_sum: float = 0.0
    weight_sum: float = 0.0
    positive_count: int = 0
    negative_count: int = 0

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


@dataclass
class IngestionStats:
    """Statistics for ingestion run."""
    total_posts: int = 0
    posts_with_tickers: int = 0
    posts_in_range: int = 0
    ticker_counts: Counter = field(default_factory=Counter)
    daily_counts: Counter = field(default_factory=Counter)
    rows_inserted_posts: int = 0
    rows_inserted_daily: int = 0
    errors: int = 0
    source: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _build_dsn() -> str:
    return (
        f"postgresql://{_env('POSTGRES_USER', 'awet')}:{_env('POSTGRES_PASSWORD', 'awet')}"
        f"@{_env('POSTGRES_HOST', 'localhost')}:{_env('POSTGRES_PORT', '5433')}"
        f"/{_env('POSTGRES_DB', 'awet')}"
    )


def parse_month(month_str: str) -> tuple[datetime, datetime]:
    """Parse YYYY-MM and return (start, end) datetimes in UTC."""
    try:
        year, month = map(int, month_str.split("-"))
    except ValueError:
        raise ValueError(f"Invalid month format: {month_str}. Use YYYY-MM")
    
    start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    # End is first day of next month
    if month == 12:
        end = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    return start, end


def load_universe() -> set[str]:
    """Load ticker symbols from universe.csv."""
    if not UNIVERSE_FILE.exists():
        return set(DEFAULT_TICKERS)
    
    symbols = set()
    with open(UNIVERSE_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker") or row.get("symbol") or row.get("Ticker")
            if ticker:
                symbols.add(ticker.upper().strip())
    
    return symbols if symbols else set(DEFAULT_TICKERS)


def extract_tickers(text: str, valid_tickers: set[str]) -> list[str]:
    """Extract valid tickers from text (ticker symbols only)."""
    if not text:
        return []
    
    matches = TICKER_PATTERN.findall(text)
    found = set()
    
    for match in matches:
        ticker = match[0] or match[1]
        if ticker and ticker not in EXCLUDED_WORDS and ticker in valid_tickers:
            found.add(ticker)
    
    return list(found)


def extract_tickers_and_companies(text: str, valid_tickers: set[str]) -> list[str]:
    """
    Extract valid tickers from text, including company name matching.
    
    Matches both:
    - Ticker symbols: $AAPL, AAPL
    - Company names: Apple, Microsoft, Tesla (case-insensitive)
    """
    if not text:
        return []
    
    found = set()
    
    # Match ticker symbols
    matches = TICKER_PATTERN.findall(text)
    for match in matches:
        ticker = match[0] or match[1]
        if ticker and ticker not in EXCLUDED_WORDS and ticker in valid_tickers:
            found.add(ticker)
    
    # Match company names (case-insensitive word boundary match)
    text_lower = text.lower()
    for company_name, ticker in COMPANY_NAME_MAP.items():
        if ticker in valid_tickers:
            # Word boundary match
            pattern = rf"\b{re.escape(company_name)}\b"
            if re.search(pattern, text_lower):
                found.add(ticker)
    
    return list(found)


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Analysis
# ─────────────────────────────────────────────────────────────────────────────

class SentimentScorer:
    """Sentiment scorer using VADER if available, else simple lexicon."""
    
    def __init__(self):
        self.vader = None
        if HAS_VADER:
            self.vader = SentimentIntensityAnalyzer()
    
    def score(self, text: str) -> tuple[float, str]:
        """
        Score text sentiment.
        
        Returns:
            (score, label) where score is in [-1, 1]
        """
        if not text:
            return 0.0, "neutral"
        
        if self.vader:
            scores = self.vader.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                return compound, "positive"
            elif compound <= -0.05:
                return compound, "negative"
            else:
                return compound, "neutral"
        
        # Fallback: simple word-based scoring
        positive_words = {
            "buy", "long", "bullish", "moon", "rocket", "calls", "growth",
            "undervalued", "breakout", "upside", "profit", "gains", "rally",
        }
        negative_words = {
            "sell", "short", "bearish", "crash", "dump", "puts", "overvalued",
            "downside", "loss", "tank", "weak", "miss", "fear", "panic",
        }
        
        words = set(re.findall(r'\b\w+\b', text.lower()))
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        total = pos + neg
        
        if total == 0:
            return 0.0, "neutral"
        
        score = (pos - neg) / total
        label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
        return score, label


# ─────────────────────────────────────────────────────────────────────────────
# Local Dump File Processing
# ─────────────────────────────────────────────────────────────────────────────

def find_dump_files(month_str: str, strict: bool = True) -> tuple[Path | None, Path | None]:
    """
    Find dump files for a given month.
    
    Args:
        month_str: Month in YYYY-MM format (e.g., "2025-07")
        strict: If True, raise error if expected files are missing
    
    Returns:
        (submissions_path, comments_path) - either may be None if not found
    
    Raises:
        FileNotFoundError: If strict=True and expected files don't exist
    """
    subs_dir = REDDIT_DIR / "submissions"
    comments_dir = REDDIT_DIR / "comments"
    
    # Expected paths (canonical naming)
    expected_subs = subs_dir / f"RS_{month_str}.zst"
    expected_comments = comments_dir / f"RC_{month_str}.zst"
    
    # Try various naming patterns for submissions
    patterns = [
        f"RS_{month_str}.zst",
        f"RS_{month_str}",
        f"submissions_{month_str}.zst",
    ]
    
    subs_file = None
    for pattern in patterns:
        candidate = subs_dir / pattern
        if candidate.exists():
            subs_file = candidate
            break
    
    # Try various naming patterns for comments
    patterns = [
        f"RC_{month_str}.zst",
        f"RC_{month_str}",
        f"comments_{month_str}.zst",
    ]
    
    comments_file = None
    for pattern in patterns:
        candidate = comments_dir / pattern
        if candidate.exists():
            comments_file = candidate
            break
    
    # Strict mode: fail fast if files missing
    if strict and not subs_file and not comments_file:
        missing = []
        if not subs_file:
            missing.append(f"  Submissions: {expected_subs}")
        if not comments_file:
            missing.append(f"  Comments:    {expected_comments}")
        
        raise FileNotFoundError(
            f"\n❌ Missing Reddit dump files for {month_str}:\n" +
            "\n".join(missing) + "\n\n" +
            f"Expected files in: {REDDIT_DIR}\n" +
            "Download from: https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10\n" +
            "Or use --use-api flag (limited historical access)"
        )
    
    return subs_file, comments_file


def stream_zst_file(
    file_path: Path,
    start_ts: float,
    end_ts: float,
    valid_tickers: set[str],
    scorer: SentimentScorer,
    max_items: int = 0,
) -> Iterator[tuple[RedditPost, list[str]]]:
    """
    Stream posts from a .zst file, filtering by date and yielding ticker matches.
    
    Yields:
        (post, tickers) for each post with ticker mentions in date range
    """
    if not HAS_ZSTD:
        raise RuntimeError("zstandard not installed: pip install zstandard")
    
    post_type = "comments" if "RC_" in file_path.name.upper() else "submissions"
    count = 0
    
    with open(file_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            
            for line in text_stream:
                if max_items > 0 and count >= max_items:
                    break
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Check timestamp
                created_utc = data.get("created_utc", 0)
                if not created_utc:
                    continue
                
                if created_utc < start_ts or created_utc >= end_ts:
                    continue
                
                # Extract text
                title = data.get("title", "") or ""
                body = data.get("selftext", "") or data.get("body", "") or ""
                text = f"{title} {body}"
                
                # Find tickers
                tickers = extract_tickers(text, valid_tickers)
                if not tickers:
                    continue
                
                count += 1
                
                # Score sentiment
                sentiment, _ = scorer.score(text)
                
                # Create post for each ticker
                post_id = data.get("id") or data.get("name") or ""
                if not post_id:
                    continue
                
                dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                
                post = RedditPost(
                    post_id=post_id,
                    ticker="",  # Will be set per ticker
                    created_utc=dt,
                    subreddit=data.get("subreddit", "unknown"),
                    title=title[:500],
                    body=body[:2000],
                    score=int(data.get("score", 0) or 0),
                    num_comments=int(data.get("num_comments", 0) or 0),
                    post_type=post_type,
                    source="reddit_dump",
                    sentiment=sentiment,
                )
                
                yield post, tickers


def process_dump_files(
    submissions_path: Path | None,
    comments_path: Path | None,
    start: datetime,
    end: datetime,
    valid_tickers: set[str],
    max_items: int = 0,
) -> tuple[list[RedditPost], dict[str, dict[str, DailyAggregate]], IngestionStats]:
    """
    Process dump files and return posts + daily aggregates.
    """
    scorer = SentimentScorer()
    stats = IngestionStats(source="reddit_dump")
    posts: list[RedditPost] = []
    daily_aggs: dict[str, dict[str, DailyAggregate]] = defaultdict(dict)
    
    start_ts = start.timestamp()
    end_ts = end.timestamp()
    
    files_to_process = []
    if submissions_path and submissions_path.exists():
        files_to_process.append(submissions_path)
    if comments_path and comments_path.exists():
        files_to_process.append(comments_path)
    
    for file_path in files_to_process:
        print(f"  Processing {file_path.name}...")
        file_count = 0
        
        for post, tickers in stream_zst_file(
            file_path, start_ts, end_ts, valid_tickers, scorer, max_items
        ):
            stats.posts_in_range += 1
            file_count += 1
            
            if file_count % LOG_INTERVAL == 0:
                print(f"    Processed {file_count:,} posts with tickers...")
            
            for ticker in tickers:
                stats.ticker_counts[ticker] += 1
                stats.posts_with_tickers += 1
                
                # Create ticker-specific post
                ticker_post = RedditPost(
                    post_id=post.post_id,
                    ticker=ticker,
                    created_utc=post.created_utc,
                    subreddit=post.subreddit,
                    title=post.title,
                    body=post.body,
                    score=post.score,
                    num_comments=post.num_comments,
                    post_type=post.post_type,
                    source=post.source,
                    sentiment=post.sentiment,
                )
                posts.append(ticker_post)
                
                # Update daily aggregate
                day_key = post.created_utc.strftime("%Y-%m-%d")
                stats.daily_counts[(day_key, ticker)] += 1
                
                if day_key not in daily_aggs[ticker]:
                    daily_aggs[ticker][day_key] = DailyAggregate(
                        day=datetime.strptime(day_key, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                        ticker=ticker,
                    )
                
                agg = daily_aggs[ticker][day_key]
                agg.mentions_count += 1
                agg.sentiment_sum += post.sentiment
                
                weight = max(1, abs(post.score))
                agg.weighted_sentiment_sum += post.sentiment * weight
                agg.weight_sum += weight
                
                if post.sentiment > 0.05:
                    agg.positive_count += 1
                elif post.sentiment < -0.05:
                    agg.negative_count += 1
        
        print(f"    Found {file_count:,} posts with ticker mentions in date range")
    
    return posts, dict(daily_aggs), stats


# ─────────────────────────────────────────────────────────────────────────────
# Reddit API Mode
# ─────────────────────────────────────────────────────────────────────────────

def _api_request_with_retry(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    """
    Execute a PRAW API call with exponential backoff retry on rate limits.
    
    Handles:
    - 429 Too Many Requests
    - Connection errors
    - General exceptions with retry
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            # Check if it's a rate limit or retriable error
            is_rate_limit = "429" in error_str or "rate" in error_str or "too many" in error_str
            is_connection = "connection" in error_str or "timeout" in error_str
            
            if attempt < max_retries and (is_rate_limit or is_connection):
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"      ⏳ Rate limited or connection error, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                break
    
    raise last_exception


def fetch_from_api(
    start: datetime,
    end: datetime,
    valid_tickers: set[str],
    subreddits: list[str],
    max_items: int = 0,
) -> tuple[list[RedditPost], dict[str, dict[str, DailyAggregate]], IngestionStats]:
    """
    Fetch submissions AND comments from Reddit API via PRAW.
    
    Features:
    - Script authentication (username/password)
    - Rate limiting with exponential backoff
    - Retry on 429 and connection errors
    - Both submissions and comments
    - Company name matching (Apple -> AAPL)
    
    WARNING: Reddit API has limited historical access. For complete data,
    use local Pushshift dump files instead.
    """
    try:
        import praw
        from prawcore.exceptions import ResponseException, RequestException
    except ImportError:
        raise RuntimeError("praw not installed: pip install praw")
    
    # Load credentials from environment
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "AWET/1.0")
    username = os.getenv("REDDIT_USERNAME")
    password = os.getenv("REDDIT_PASSWORD")
    
    # Validate required credentials
    missing = []
    if not client_id:
        missing.append("REDDIT_CLIENT_ID")
    if not client_secret:
        missing.append("REDDIT_CLIENT_SECRET")
    if not username:
        missing.append("REDDIT_USERNAME")
    if not password:
        missing.append("REDDIT_PASSWORD")
    
    if missing:
        raise RuntimeError(
            f"Missing Reddit API credentials: {', '.join(missing)}\\n"
            "Set these environment variables before running with --use-api"
        )
    
    print()
    print("⚠️  NOTICE: Using Reddit API mode")
    print("   Reddit API has LIMITED historical access and may not return")
    print(f"   complete data for {start.strftime('%Y-%m')}.")
    print("   For complete historical coverage, use local Pushshift dumps.")
    print()
    print(f"   Subreddits: {', '.join(subreddits)}")
    print(f"   Date range: {start.date()} to {end.date()}")
    print()
    
    # Create Reddit client with script authentication
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password,
    )
    
    # Verify authentication
    try:
        me = reddit.user.me()
        print(f"   ✅ Authenticated as: u/{me.name}")
    except Exception as e:
        raise RuntimeError(f"Reddit authentication failed: {e}")
    
    print()
    
    scorer = SentimentScorer()
    stats = IngestionStats(source="praw_api")
    posts: list[RedditPost] = []
    daily_aggs: dict[str, dict[str, DailyAggregate]] = defaultdict(dict)
    
    start_ts = start.timestamp()
    end_ts = end.timestamp()
    
    # Calculate items per subreddit
    if max_items > 0:
        items_per_sub = max(100, max_items // len(subreddits))
    else:
        items_per_sub = 1000  # Default limit per subreddit
    
    total_items_collected = 0
    
    for subreddit_name in subreddits:
        if max_items > 0 and total_items_collected >= max_items:
            print(f"  ⏹️  Reached max_items limit ({max_items}), stopping...")
            break
        
        print(f"  📥 Fetching from r/{subreddit_name}...")
        sub_submissions = 0
        sub_comments = 0
        
        try:
            subreddit = reddit.subreddit(subreddit_name)
            
            # ─────────────────────────────────────────────────────────────
            # Fetch SUBMISSIONS
            # ─────────────────────────────────────────────────────────────
            print(f"     Submissions (new, hot, top)...")
            
            for listing_name, listing_func in [
                ("new", lambda: subreddit.new(limit=min(items_per_sub, 1000))),
                ("hot", lambda: subreddit.hot(limit=min(items_per_sub, 500))),
                ("top", lambda: subreddit.top(time_filter="month", limit=min(items_per_sub, 500))),
            ]:
                if max_items > 0 and total_items_collected >= max_items:
                    break
                
                try:
                    submissions = _api_request_with_retry(listing_func)
                    
                    for submission in submissions:
                        if max_items > 0 and total_items_collected >= max_items:
                            break
                        
                        stats.total_posts += 1
                        created = submission.created_utc
                        
                        # Filter by date range
                        if created < start_ts or created >= end_ts:
                            continue
                        
                        stats.posts_in_range += 1
                        
                        title = submission.title or ""
                        body = submission.selftext or ""
                        text = f"{title} {body}"
                        
                        # Match tickers AND company names
                        tickers = extract_tickers_and_companies(text, valid_tickers)
                        if not tickers:
                            continue
                        
                        sub_submissions += 1
                        total_items_collected += 1
                        sentiment, _ = scorer.score(text)
                        dt = datetime.fromtimestamp(created, tz=timezone.utc)
                        
                        for ticker in tickers:
                            stats.ticker_counts[ticker] += 1
                            stats.posts_with_tickers += 1
                            
                            post = RedditPost(
                                post_id=submission.id,
                                ticker=ticker,
                                created_utc=dt,
                                subreddit=subreddit_name,
                                title=title[:500],
                                body=body[:2000],
                                score=submission.score,
                                num_comments=submission.num_comments,
                                post_type="submissions",
                                source="praw_api",
                                sentiment=sentiment,
                            )
                            posts.append(post)
                            
                            # Update daily aggregate
                            _update_daily_aggregate(daily_aggs, stats, ticker, dt, sentiment, submission.score)
                        
                        # Rate limiting: small delay between requests
                        time.sleep(0.05)
                
                except Exception as e:
                    print(f"       ⚠️  Error fetching {listing_name}: {e}")
                    stats.errors += 1
            
            # ─────────────────────────────────────────────────────────────
            # Fetch COMMENTS (from top submissions)
            # ─────────────────────────────────────────────────────────────
            print(f"     Comments (from top posts)...")
            
            try:
                # Get top submissions to fetch their comments
                top_subs = _api_request_with_retry(
                    lambda: list(subreddit.top(time_filter="month", limit=50))
                )
                
                for submission in top_subs:
                    if max_items > 0 and total_items_collected >= max_items:
                        break
                    
                    # Skip if submission is outside date range
                    if submission.created_utc < start_ts or submission.created_utc >= end_ts:
                        continue
                    
                    try:
                        # Replace "more comments" with actual comments (limited)
                        submission.comments.replace_more(limit=0)
                        
                        for comment in submission.comments.list()[:100]:  # Limit comments per post
                            if max_items > 0 and total_items_collected >= max_items:
                                break
                            
                            stats.total_posts += 1
                            created = comment.created_utc
                            
                            if created < start_ts or created >= end_ts:
                                continue
                            
                            stats.posts_in_range += 1
                            
                            body = comment.body or ""
                            
                            tickers = extract_tickers_and_companies(body, valid_tickers)
                            if not tickers:
                                continue
                            
                            sub_comments += 1
                            total_items_collected += 1
                            sentiment, _ = scorer.score(body)
                            dt = datetime.fromtimestamp(created, tz=timezone.utc)
                            
                            for ticker in tickers:
                                stats.ticker_counts[ticker] += 1
                                stats.posts_with_tickers += 1
                                
                                post = RedditPost(
                                    post_id=comment.id,
                                    ticker=ticker,
                                    created_utc=dt,
                                    subreddit=subreddit_name,
                                    title="",  # Comments don't have titles
                                    body=body[:2000],
                                    score=comment.score,
                                    num_comments=0,
                                    post_type="comments",
                                    source="praw_api",
                                    sentiment=sentiment,
                                )
                                posts.append(post)
                                
                                _update_daily_aggregate(daily_aggs, stats, ticker, dt, sentiment, comment.score)
                        
                        # Rate limiting between posts
                        time.sleep(0.1)
                    
                    except Exception as e:
                        # Skip individual comment errors
                        continue
            
            except Exception as e:
                print(f"       ⚠️  Error fetching comments: {e}")
                stats.errors += 1
            
            print(f"     ✅ Found {sub_submissions} submissions, {sub_comments} comments with tickers")
        
        except Exception as e:
            print(f"     ❌ Error accessing r/{subreddit_name}: {e}")
            stats.errors += 1
        
        # Rate limiting between subreddits
        time.sleep(0.5)
    
    return posts, dict(daily_aggs), stats


def _update_daily_aggregate(
    daily_aggs: dict[str, dict[str, DailyAggregate]],
    stats: IngestionStats,
    ticker: str,
    dt: datetime,
    sentiment: float,
    score: int,
) -> None:
    """Helper to update daily aggregates."""
    day_key = dt.strftime("%Y-%m-%d")
    stats.daily_counts[(day_key, ticker)] += 1
    
    if day_key not in daily_aggs[ticker]:
        daily_aggs[ticker][day_key] = DailyAggregate(
            day=datetime.strptime(day_key, "%Y-%m-%d").replace(tzinfo=timezone.utc),
            ticker=ticker,
        )
    
    agg = daily_aggs[ticker][day_key]
    agg.mentions_count += 1
    agg.sentiment_sum += sentiment
    
    weight = max(1, abs(score))
    agg.weighted_sentiment_sum += sentiment * weight
    agg.weight_sum += weight
    
    if sentiment > 0.05:
        agg.positive_count += 1
    elif sentiment < -0.05:
        agg.negative_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Database Operations
# ─────────────────────────────────────────────────────────────────────────────

async def write_posts_to_db(
    conn: asyncpg.Connection,
    posts: list[RedditPost],
) -> int:
    """Write posts to reddit_posts with UPSERT."""
    if not posts:
        return 0
    
    inserted = 0
    batch = []
    
    for post in posts:
        batch.append((
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
        ))
        
        if len(batch) >= DB_BATCH_SIZE:
            await conn.executemany(
                f"""
                INSERT INTO {REDDIT_POSTS_TABLE} (
                    post_id, ticker, created_utc, subreddit,
                    title, body, score, num_comments, post_type, source
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (post_id, ticker) DO UPDATE SET
                    score = EXCLUDED.score,
                    num_comments = EXCLUDED.num_comments,
                    source = EXCLUDED.source
                """,
                batch,
            )
            inserted += len(batch)
            batch = []
    
    if batch:
        await conn.executemany(
            f"""
            INSERT INTO {REDDIT_POSTS_TABLE} (
                post_id, ticker, created_utc, subreddit,
                title, body, score, num_comments, post_type, source
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (post_id, ticker) DO UPDATE SET
                score = EXCLUDED.score,
                num_comments = EXCLUDED.num_comments,
                source = EXCLUDED.source
            """,
            batch,
        )
        inserted += len(batch)
    
    return inserted


async def write_daily_aggregates_to_db(
    conn: asyncpg.Connection,
    daily_aggs: dict[str, dict[str, DailyAggregate]],
) -> int:
    """Write daily aggregates to reddit_daily_mentions with UPSERT."""
    inserted = 0
    
    for ticker, days in daily_aggs.items():
        for day_key, agg in days.items():
            await conn.execute(
                f"""
                INSERT INTO {REDDIT_DAILY_TABLE} (
                    day, ticker, mentions_count,
                    sentiment_mean, sentiment_weighted,
                    positive_ratio, negative_ratio
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (day, ticker) DO UPDATE SET
                    mentions_count = EXCLUDED.mentions_count,
                    sentiment_mean = EXCLUDED.sentiment_mean,
                    sentiment_weighted = EXCLUDED.sentiment_weighted,
                    positive_ratio = EXCLUDED.positive_ratio,
                    negative_ratio = EXCLUDED.negative_ratio
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
    
    return inserted


async def update_features_tft_from_reddit(
    conn: asyncpg.Connection,
    start: datetime,
    end: datetime,
) -> int:
    """
    Update features_tft with Reddit data from reddit_daily_mentions.
    
    This backfills reddit_* columns for existing feature rows.
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
          AND r.day >= $1::date
          AND r.day < $2::date
        """,
        start,
        end,
    )
    
    # Parse UPDATE count
    count = 0
    if result:
        parts = result.split()
        if len(parts) >= 2:
            try:
                count = int(parts[1])
            except ValueError:
                pass
    
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

async def print_verification(
    conn: asyncpg.Connection,
    start: datetime,
    end: datetime,
    stats: IngestionStats,
) -> None:
    """Print verification stats."""
    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Days in month
    total_days = (end - start).days
    print(f"  Month period:      {start.date()} to {end.date()} ({total_days} days)")
    print(f"  Data source:       {stats.source}")
    
    # Count in DB
    posts_count = await conn.fetchval(
        f"""
        SELECT COUNT(*) FROM {REDDIT_POSTS_TABLE}
        WHERE created_utc >= $1 AND created_utc < $2
        """,
        start, end,
    )
    print(f"  Posts in DB:       {posts_count:,}")
    
    daily_count = await conn.fetchval(
        f"""
        SELECT COUNT(*) FROM {REDDIT_DAILY_TABLE}
        WHERE day >= $1::date AND day < $2::date
        """,
        start, end,
    )
    print(f"  Daily aggregates:  {daily_count:,}")
    
    # Coverage by ticker
    print()
    print("  Coverage by ticker (days with mentions / total days):")
    
    ticker_coverage = await conn.fetch(
        f"""
        SELECT ticker, COUNT(DISTINCT day) as days_with_data,
               SUM(mentions_count) as total_mentions
        FROM {REDDIT_DAILY_TABLE}
        WHERE day >= $1::date AND day < $2::date
        GROUP BY ticker
        ORDER BY total_mentions DESC
        LIMIT 15
        """,
        start, end,
    )
    
    if ticker_coverage:
        for row in ticker_coverage:
            pct = 100.0 * row["days_with_data"] / total_days
            print(f"    {row['ticker']:6s}: {row['days_with_data']:2d}/{total_days} days ({pct:5.1f}%), "
                  f"{row['total_mentions']:,} mentions")
    else:
        print("    ⚠️  No data found!")
    
    # Sample rows
    print()
    print("  Sample reddit_daily_mentions rows:")
    samples = await conn.fetch(
        f"""
        SELECT day, ticker, mentions_count,
               ROUND(sentiment_mean::numeric, 3) as sent,
               ROUND(positive_ratio::numeric, 3) as pos
        FROM {REDDIT_DAILY_TABLE}
        WHERE day >= $1::date AND day < $2::date
          AND mentions_count > 0
        ORDER BY mentions_count DESC
        LIMIT 10
        """,
        start, end,
    )
    
    for row in samples:
        print(f"    {row['day']} {row['ticker']:6s}: {row['mentions_count']:5d} mentions, "
              f"sent={row['sent']}, pos={row['pos']}")
    
    # Features coverage
    features_count = await conn.fetchval(
        """
        SELECT COUNT(*) FROM features_tft
        WHERE ts >= $1 AND ts < $2
          AND reddit_mentions_count > 0
        """,
        start, end,
    )
    total_features = await conn.fetchval(
        """
        SELECT COUNT(*) FROM features_tft
        WHERE ts >= $1 AND ts < $2
        """,
        start, end,
    )
    
    print()
    if total_features and total_features > 0:
        pct = 100.0 * features_count / total_features
        print(f"  features_tft with Reddit: {features_count:,} / {total_features:,} ({pct:.2f}%)")
    else:
        print("  features_tft: No feature rows in date range")
    
    if features_count == 0 and daily_count > 0:
        print()
        print("  💡 TIP: Run 'python scripts/feature_engineer.py' to populate features_tft")
        print("         or use --update-features flag")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest Reddit data for a specific month",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--month", required=True,
        help="Month to ingest in YYYY-MM format (e.g., 2025-07)",
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help=f"Comma-separated tickers (default: from universe.csv or {','.join(DEFAULT_TICKERS)})",
    )
    parser.add_argument(
        "--submissions-zst", type=Path, default=None,
        help="Path to submissions .zst file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--comments-zst", type=Path, default=None,
        help="Path to comments .zst file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--use-api", action="store_true",
        help="Use Reddit API as data source (requires credentials)",
    )
    parser.add_argument(
        "--write-db", action="store_true",
        help="Write to database (default: dry-run)",
    )
    parser.add_argument(
        "--update-features", action="store_true",
        help="Also update features_tft with Reddit data",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be ingested without writing",
    )
    parser.add_argument(
        "--max-items", type=int, default=0,
        help="Maximum items to process (0 = unlimited)",
    )
    parser.add_argument(
        "--subreddits", type=str, default=None,
        help=f"Comma-separated subreddits for API mode (default: {','.join(DEFAULT_API_SUBREDDITS)})",
    )
    
    args = parser.parse_args()
    
    # Parse month
    try:
        start, end = parse_month(args.month)
    except ValueError as e:
        print(f"❌ {e}")
        return 1
    
    # Load tickers
    if args.tickers:
        valid_tickers = set(t.strip().upper() for t in args.tickers.split(","))
    else:
        valid_tickers = load_universe()
    
    print()
    print("=" * 60)
    print(f"REDDIT INGESTION - {args.month}")
    print("=" * 60)
    print(f"  Date range:    {start.date()} to {end.date()}")
    print(f"  Tickers:       {len(valid_tickers)} ({', '.join(sorted(valid_tickers)[:10])}{'...' if len(valid_tickers) > 10 else ''})")
    print(f"  Write to DB:   {'Yes' if args.write_db else 'No (dry-run)'}")
    print(f"  Max items:     {'Unlimited' if args.max_items == 0 else args.max_items:,}")
    
    # Determine data source
    if args.use_api:
        source = "api"
        # Parse subreddits
        if args.subreddits:
            subreddits = [s.strip() for s in args.subreddits.split(",")]
        else:
            subreddits = DEFAULT_API_SUBREDDITS
        print(f"  Data source:   Reddit API (PRAW)")
        print(f"  Subreddits:    {', '.join(subreddits)}")
    else:
        # Look for dump files
        subs_file = args.submissions_zst
        comments_file = args.comments_zst
        
        if not subs_file and not comments_file:
            try:
                subs_file, comments_file = find_dump_files(args.month, strict=True)
            except FileNotFoundError as e:
                print(str(e))
                return 1
        
        # Validate explicit paths exist
        if subs_file and not subs_file.exists():
            print(f"\n❌ Submissions file not found: {subs_file}")
            return 1
        if comments_file and not comments_file.exists():
            print(f"\n❌ Comments file not found: {comments_file}")
            return 1
        
        source = "dump"
        print(f"  Data source:   Local dumps ({args.month})")
        if subs_file:
            print(f"    Submissions: {subs_file}")
        if comments_file:
            print(f"    Comments:    {comments_file}")
    
    print()
    
    # Fetch data
    if source == "api":
        posts, daily_aggs, stats = fetch_from_api(
            start, end, valid_tickers, subreddits, args.max_items
        )
    else:
        posts, daily_aggs, stats = process_dump_files(
            subs_file, comments_file, start, end, valid_tickers, args.max_items
        )
    
    # Summary
    print()
    print("Processing complete:")
    print(f"  Posts with tickers: {len(posts):,}")
    print(f"  Daily aggregates:   {sum(len(d) for d in daily_aggs.values()):,}")
    print(f"  Top tickers:")
    for ticker, count in stats.ticker_counts.most_common(10):
        print(f"    {ticker}: {count:,}")
    
    # Write to DB
    if args.write_db and not args.dry_run:
        print()
        print("Writing to database...")
        
        conn = await asyncpg.connect(_build_dsn())
        
        try:
            stats.rows_inserted_posts = await write_posts_to_db(conn, posts)
            print(f"  Inserted/updated {stats.rows_inserted_posts:,} rows in {REDDIT_POSTS_TABLE}")
            
            stats.rows_inserted_daily = await write_daily_aggregates_to_db(conn, daily_aggs)
            print(f"  Inserted/updated {stats.rows_inserted_daily:,} rows in {REDDIT_DAILY_TABLE}")
            
            if args.update_features:
                features_updated = await update_features_tft_from_reddit(conn, start, end)
                print(f"  Updated {features_updated:,} rows in features_tft")
            
            await print_verification(conn, start, end, stats)
            
        finally:
            await conn.close()
    else:
        print()
        if args.dry_run:
            print("Dry-run complete. Use --write-db to persist data.")
        else:
            print("Use --write-db to persist data to database.")
    
    print()
    print("✅ Done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
        sys.exit(1)
