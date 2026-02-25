#!/usr/bin/env python3
"""
Verify Reddit data alignment with market data and write to DB.

Checks:
1. Reddit timestamps align with trading days (ignores weekends)
2. Ticker mentions in Reddit match universe symbols
3. Coverage report per symbol
4. Optional DB ingestion for reddit_posts + reddit_daily_mentions
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import structlog

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

logger = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
REDDIT_DIR = Path("/home/kironix/train/reddit")
UNIVERSE_FILE = PROJECT_ROOT / "config" / "universe.csv"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://awet:awet@localhost:5433/awet")
REDDIT_POSTS_TABLE = "reddit_posts"
REDDIT_DAILY_TABLE = "reddit_daily_mentions"
DB_BATCH_SIZE = 1000
PRAW_CACHE_FILE = PROJECT_ROOT / ".tmp" / "reddit_praw_cache.jsonl"

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


def extract_tickers(text: str, valid_tickers: set[str] | None = None) -> list[str]:
    if not text:
        return []
    matches = TICKER_PATTERN.findall(text)
    tickers = set()
    for match in matches:
        ticker = match[0] or match[1]
        if ticker and ticker not in EXCLUDED_WORDS:
            if valid_tickers is None or ticker in valid_tickers:
                tickers.add(ticker)
    return list(tickers)


def load_universe(file_path: Path) -> set[str]:
    if not file_path.exists():
        return set()
    symbols = set()
    with open(file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbols.add(row["ticker"])
    return symbols


def is_trading_day(dt: datetime) -> bool:
    return dt.weekday() < 5


def parse_reddit_file(
    file_path: Path,
    valid_tickers: set[str] | None = None,
    max_samples: int = 5,
    max_posts: int = 0,
) -> dict:
    if not HAS_ZSTD:
        return {"error": "zstandard not installed: pip install zstandard"}

    stats = {
        "file": file_path.name,
        "total_posts": 0,
        "posts_with_tickers": 0,
        "trading_day_posts": 0,
        "weekend_posts": 0,
        "ticker_counts": Counter(),
        "subreddit_counts": Counter(),
        "dates": set(),
        "daily_counts": Counter(),
        "ticker_daily_counts": Counter(),
        "sample_posts": [],
        "errors": 0,
    }

    try:
        with open(file_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                import io
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")

                for line in text_stream:
                    if max_posts and stats["total_posts"] >= max_posts:
                        break
                    try:
                        post = json.loads(line)
                        stats["total_posts"] += 1

                        created_utc = post.get("created_utc", 0)
                        dt = None
                        if created_utc:
                            dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                            stats["dates"].add(dt.date())
                            stats["daily_counts"][dt.date()] += 1
                            if is_trading_day(dt):
                                stats["trading_day_posts"] += 1
                            else:
                                stats["weekend_posts"] += 1

                        subreddit = post.get("subreddit", "unknown")
                        stats["subreddit_counts"][subreddit] += 1

                        title = post.get("title", "")
                        body = post.get("selftext", "") or post.get("body", "")
                        text = f"{title} {body}"

                        tickers = extract_tickers(text, valid_tickers)
                        if tickers:
                            stats["posts_with_tickers"] += 1
                            for ticker in tickers:
                                stats["ticker_counts"][ticker] += 1
                                if dt:
                                    stats["ticker_daily_counts"][(dt.date(), ticker)] += 1

                            if len(stats["sample_posts"]) < max_samples:
                                snippet = text[:150].replace("\n", " ").strip()
                                if len(text) > 150:
                                    snippet += "..."
                                stats["sample_posts"].append({
                                    "ts": dt.isoformat() if dt else "unknown",
                                    "tickers": tickers,
                                    "subreddit": subreddit,
                                    "snippet": snippet,
                                })

                    except json.JSONDecodeError:
                        stats["errors"] += 1
                        continue
    except Exception as e:
        stats["error"] = str(e)

    stats["date_range"] = (
        min(stats["dates"]) if stats["dates"] else None,
        max(stats["dates"]) if stats["dates"] else None,
    )
    stats["unique_dates"] = len(stats["dates"])
    del stats["dates"]
    return stats


async def ingest_reddit_file(
    file_path: Path,
    conn,
    valid_tickers: set[str] | None = None,
    max_samples: int = 5,
    max_posts: int = 0,
) -> dict:
    if not HAS_ZSTD:
        return {"error": "zstandard not installed: pip install zstandard"}

    stats = {
        "file": file_path.name,
        "total_posts": 0,
        "posts_with_tickers": 0,
        "trading_day_posts": 0,
        "weekend_posts": 0,
        "ticker_counts": Counter(),
        "subreddit_counts": Counter(),
        "dates": set(),
        "daily_counts": Counter(),
        "ticker_daily_counts": Counter(),
        "sample_posts": [],
        "errors": 0,
        "rows_inserted": 0,
    }

    batch = []
    post_type = "comments" if "RC_" in file_path.name else "submissions"

    try:
        with open(file_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                import io
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")

                for line in text_stream:
                    if max_posts and stats["total_posts"] >= max_posts:
                        break
                    try:
                        post = json.loads(line)
                        stats["total_posts"] += 1

                        created_utc = post.get("created_utc", 0)
                        dt = None
                        if created_utc:
                            dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                            stats["dates"].add(dt.date())
                            stats["daily_counts"][dt.date()] += 1
                            if is_trading_day(dt):
                                stats["trading_day_posts"] += 1
                            else:
                                stats["weekend_posts"] += 1

                        subreddit = post.get("subreddit", "unknown")
                        stats["subreddit_counts"][subreddit] += 1

                        title = post.get("title", "")
                        body = post.get("selftext", "") or post.get("body", "")
                        text = f"{title} {body}"

                        tickers = extract_tickers(text, valid_tickers)
                        if tickers:
                            stats["posts_with_tickers"] += 1
                            for ticker in tickers:
                                stats["ticker_counts"][ticker] += 1
                                if dt:
                                    stats["ticker_daily_counts"][(dt.date(), ticker)] += 1

                                post_id = post.get("id") or post.get("name")
                                if not post_id:
                                    continue

                                batch.append((
                                    post_id,
                                    ticker,
                                    dt,
                                    subreddit,
                                    title[:500],
                                    body[:2000],
                                    int(post.get("score", 0) or 0),
                                    int(post.get("num_comments", 0) or 0),
                                    post_type,
                                    "reddit_dump",
                                ))

                            if len(stats["sample_posts"]) < max_samples:
                                snippet = text[:150].replace("\n", " ").strip()
                                if len(text) > 150:
                                    snippet += "..."
                                stats["sample_posts"].append({
                                    "ts": dt.isoformat() if dt else "unknown",
                                    "tickers": tickers,
                                    "subreddit": subreddit,
                                    "snippet": snippet,
                                })

                        if len(batch) >= DB_BATCH_SIZE:
                            await conn.executemany(
                                f"""
                                INSERT INTO {REDDIT_POSTS_TABLE} (
                                    post_id, ticker, created_utc, subreddit,
                                    title, body, score, num_comments, post_type, source
                                ) VALUES (
                                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
                                )
                                ON CONFLICT (post_id, ticker) DO NOTHING
                                """,
                                batch,
                            )
                            stats["rows_inserted"] += len(batch)
                            batch = []

                    except json.JSONDecodeError:
                        stats["errors"] += 1
                        continue
    except Exception as e:
        stats["error"] = str(e)

    if batch:
        await conn.executemany(
            f"""
            INSERT INTO {REDDIT_POSTS_TABLE} (
                post_id, ticker, created_utc, subreddit,
                title, body, score, num_comments, post_type, source
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
            )
            ON CONFLICT (post_id, ticker) DO NOTHING
            """,
            batch,
        )
        stats["rows_inserted"] += len(batch)

    stats["date_range"] = (
        min(stats["dates"]) if stats["dates"] else None,
        max(stats["dates"]) if stats["dates"] else None,
    )
    stats["unique_dates"] = len(stats["dates"])
    del stats["dates"]
    return stats


def ingest_praw_posts(
    subreddits: list[str],
    days: int,
    limit: int,
    valid_tickers: set[str] | None = None,
) -> list[dict]:
    try:
        import praw
    except Exception as e:
        raise RuntimeError(f"praw not installed: {e}")

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "awet-reddit-ingest")

    if not client_id or not client_secret:
        raise RuntimeError("Missing REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET in environment")

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    cutoff = datetime.now(tz=timezone.utc).timestamp() - days * 86400
    results = []

    PRAW_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRAW_CACHE_FILE, "a", encoding="utf-8") as cache_f:
        for sub in subreddits:
            subreddit = reddit.subreddit(sub)
            for post in subreddit.new(limit=limit):
                if post.created_utc < cutoff:
                    continue
                title = post.title or ""
                body = post.selftext or ""
                text = f"{title} {body}"
                tickers = extract_tickers(text, valid_tickers)
                if not tickers:
                    continue
                entry = {
                    "post_id": post.id,
                    "subreddit": sub,
                    "title": title,
                    "body": body,
                    "score": int(post.score or 0),
                    "num_comments": int(post.num_comments or 0),
                    "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                    "tickers": tickers,
                }
                results.append(entry)
                cache_f.write(json.dumps({
                    **entry,
                    "created_utc": entry["created_utc"].isoformat(),
                }) + "\n")

    return results


def _find_reddit_files(base_dir: Path, month: str, is_submissions: bool) -> list[Path]:
    if not base_dir.exists():
        return []
    # Check subdirectories first (submissions/ and comments/)
    subdir_name = "submissions" if is_submissions else "comments"
    subdir = base_dir / subdir_name
    if subdir.exists():
        files = list(subdir.glob(f"*{month}*.zst"))
    else:
        files = list(base_dir.glob(f"*{month}*.zst"))
    if is_submissions:
        return [p for p in files if "RS_" in p.name]
    return [p for p in files if "RC_" in p.name]


async def _print_db_join_coverage(conn) -> None:
    try:
        total_posts = await conn.fetchval(f"SELECT COUNT(*) FROM {REDDIT_POSTS_TABLE}")
        distinct_tickers = await conn.fetchval(f"SELECT COUNT(DISTINCT ticker) FROM {REDDIT_POSTS_TABLE}")
        daily_rows = await conn.fetchval(f"SELECT COUNT(*) FROM {REDDIT_DAILY_TABLE}")
        features_rows = await conn.fetchval("SELECT COUNT(*) FROM features_tft")
        join_rows = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM features_tft f
            JOIN reddit_daily_mentions r
              ON f.day = r.day AND f.ticker = r.ticker
            """
        )

        print(f"DB totals: {total_posts:,} reddit_posts rows, {distinct_tickers} tickers")
        print(f"DB totals: {daily_rows:,} reddit_daily_mentions rows")
        if features_rows:
            print(f"Join coverage: {join_rows:,} / {features_rows:,} feature rows ({join_rows / features_rows * 100:.1f}%)")

        coverage_rows = await conn.fetch(
            """
            SELECT ticker,
                   COUNT(*) AS total_rows,
                   SUM(CASE WHEN reddit_mentions > 0 THEN 1 ELSE 0 END) AS with_mentions
            FROM features_tft
            GROUP BY ticker
            ORDER BY with_mentions DESC, total_rows DESC
            LIMIT 10
            """
        )
        if coverage_rows:
            print("\nTop tickers by reddit coverage (features_tft):")
            for row in coverage_rows:
                pct = (row["with_mentions"] / row["total_rows"] * 100) if row["total_rows"] else 0
                print(f"  {row['ticker']}: {row['with_mentions']}/{row['total_rows']} ({pct:.1f}%)")
    except Exception as e:
        print(f"⚠️  DB join coverage check failed: {e}")


async def _print_reddit_mentions_report(conn, universe: set[str] | None = None) -> None:
    table_exists = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'reddit_mentions'
        )
        """
    )
    if not table_exists:
        print("❌ reddit_mentions table not found. Run ingest script first.")
        return

    day_range = await conn.fetchrow("SELECT MIN(ts)::date AS min_day, MAX(ts)::date AS max_day FROM market_raw_day")
    if not day_range or not day_range["min_day"]:
        day_range = await conn.fetchrow("SELECT MIN(ts)::date AS min_day, MAX(ts)::date AS max_day FROM market_raw_minute")

    min_day = day_range["min_day"] if day_range else None
    max_day = day_range["max_day"] if day_range else None

    if not min_day or not max_day:
        print("⚠️  No market data range found. Load market data first.")
        return

    print(f"Market data range: {min_day} to {max_day}")

    params = [min_day, max_day]
    ticker_filter = ""
    if universe:
        ticker_filter = "AND ticker = ANY($3)"
        params.append(sorted(universe))

    rows = await conn.fetch(
        f"""
        SELECT ticker, COUNT(*) AS mentions
        FROM reddit_mentions
        WHERE ts::date >= $1 AND ts::date <= $2
        {ticker_filter}
        GROUP BY ticker
        ORDER BY mentions DESC
        """,
        *params,
    )

    total_tickers = len(universe) if universe else await conn.fetchval("SELECT COUNT(DISTINCT ticker) FROM reddit_mentions")
    tickers_with_mentions = len(rows)

    print(f"Reddit mentions tickers: {tickers_with_mentions} / {total_tickers}")
    if rows:
        print("Top 20 tickers by reddit_mentions:")
        for row in rows[:20]:
            print(f"  {row['ticker']}: {row['mentions']:,}")

    overlap = await conn.fetchval(
        """
        SELECT COUNT(DISTINCT m.ticker)
        FROM market_raw_day m
        JOIN reddit_mentions r
          ON m.ticker = r.ticker AND m.ts::date = r.ts::date
        WHERE m.ts::date >= $1 AND m.ts::date <= $2
        """,
        min_day,
        max_day,
    )

    status = "PASS" if tickers_with_mentions > 0 and overlap > 0 else "FAIL"
    print(f"Overlap coverage with market data: {overlap} tickers")
    print(f"Result: {status}")


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Reddit data alignment with market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--month", type=str, default="2024-07")
    parser.add_argument("--universe", type=Path, default=UNIVERSE_FILE)
    parser.add_argument("--submissions-dir", type=Path, default=REDDIT_DIR)
    parser.add_argument("--comments-dir", type=Path, default=REDDIT_DIR)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-posts", type=int, default=0)
    parser.add_argument("--use-praw", action="store_true")
    parser.add_argument("--subreddits", type=str, default="wallstreetbets,stocks,investing")
    parser.add_argument("--praw-days", type=int, default=7)
    parser.add_argument("--praw-limit", type=int, default=200)
    parser.add_argument("--db-only", action="store_true", help="Skip file parsing and report from DB only")

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("REDDIT DATA ALIGNMENT VERIFICATION")
    print("=" * 70)

    if not args.use_praw and not HAS_ZSTD:
        print("❌ ERROR: zstandard library not installed")
        return 1

    universe = set() if args.no_filter else load_universe(args.universe)
    if universe:
        print(f"Universe loaded: {len(universe)} symbols from {args.universe}")
    else:
        print("No universe file found, extracting ALL ticker mentions")
    print()

    submissions_files: list[Path] = []
    comments_files: list[Path] = []
    if not args.use_praw:
        submissions_dir = args.submissions_dir
        comments_dir = args.comments_dir
        if not submissions_dir.exists() and submissions_dir.name == "submissions":
            submissions_dir = REDDIT_DIR
        if not comments_dir.exists() and comments_dir.name == "comments":
            comments_dir = REDDIT_DIR
        submissions_files = _find_reddit_files(submissions_dir, args.month, True)
        comments_files = _find_reddit_files(comments_dir, args.month, False)
        print(f"Submissions files: {len(submissions_files)}")
        print(f"Comments files: {len(comments_files)}")
        if not submissions_files and not comments_files:
            print(f"❌ No Reddit files found for {args.month}")
            return 1
    else:
        print("Using Reddit API (PRAW) for ingestion")

    db_conn = None
    if args.write_db or args.db_only:
        import asyncpg
        db_conn = await asyncpg.connect(DATABASE_URL)
        if args.write_db:
            await db_conn.execute("ALTER TABLE features_tft ADD COLUMN IF NOT EXISTS reddit_mentions INTEGER DEFAULT 0")
            await db_conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {REDDIT_POSTS_TABLE} (
                    post_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    created_utc TIMESTAMPTZ,
                    subreddit TEXT,
                    title TEXT,
                    body TEXT,
                    score INT DEFAULT 0,
                    num_comments INT DEFAULT 0,
                    post_type TEXT,
                    source TEXT,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (post_id, ticker)
                )
                """
            )
            await db_conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {REDDIT_DAILY_TABLE} (
                    day date NOT NULL,
                    ticker text NOT NULL,
                    mentions integer NOT NULL,
                    created_at timestamptz DEFAULT NOW(),
                    PRIMARY KEY (day, ticker)
                )
                """
            )

    if args.db_only:
        if not db_conn:
            print("❌ DB connection not available")
            return 1
        await _print_reddit_mentions_report(db_conn, universe if universe else None)
        if db_conn:
            await db_conn.close()
        return 0

    all_stats = {
        "total_posts": 0,
        "posts_with_tickers": 0,
        "trading_day_posts": 0,
        "weekend_posts": 0,
        "ticker_counts": Counter(),
        "subreddit_counts": Counter(),
        "ticker_daily_counts": Counter(),
        "rows_inserted": 0,
    }

    print("\nAnalyzing files...")

    if args.use_praw:
        try:
            subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]
            praw_posts = ingest_praw_posts(subreddits, args.praw_days, args.praw_limit, universe if universe else None)

            for post in praw_posts:
                all_stats["total_posts"] += 1
                dt = post["created_utc"]
                if dt:
                    all_stats.setdefault("daily_counts", Counter())[dt.date()] += 1
                    if is_trading_day(dt):
                        all_stats["trading_day_posts"] += 1
                    else:
                        all_stats["weekend_posts"] += 1
                all_stats["posts_with_tickers"] += 1
                all_stats["subreddit_counts"][post["subreddit"]] += 1
                for ticker in post["tickers"]:
                    all_stats["ticker_counts"][ticker] += 1
                    if dt:
                        all_stats["ticker_daily_counts"][(dt.date(), ticker)] += 1

                if len(all_stats.setdefault("sample_posts", [])) < args.max_samples:
                    snippet = (post["title"] + " " + post["body"])[:150].replace("\n", " ").strip()
                    all_stats["sample_posts"].append({
                        "ts": dt.isoformat() if dt else "unknown",
                        "tickers": post["tickers"],
                        "subreddit": post["subreddit"],
                        "snippet": snippet,
                    })

            if args.write_db and db_conn and praw_posts:
                batch = []
                for post in praw_posts:
                    for ticker in post["tickers"]:
                        batch.append((
                            post["post_id"],
                            ticker,
                            post["created_utc"],
                            post["subreddit"],
                            post["title"][:500],
                            post["body"][:2000],
                            int(post["score"] or 0),
                            int(post["num_comments"] or 0),
                            "praw",
                            "reddit_api",
                        ))
                if batch:
                    await db_conn.executemany(
                        f"""
                        INSERT INTO {REDDIT_POSTS_TABLE} (
                            post_id, ticker, created_utc, subreddit,
                            title, body, score, num_comments, post_type, source
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
                        )
                        ON CONFLICT (post_id, ticker) DO NOTHING
                        """,
                        batch,
                    )
                    all_stats["rows_inserted"] += len(batch)
        except Exception as e:
            print(f"❌ PRAW ingestion failed: {e}")
            if db_conn:
                await db_conn.close()
            return 1
    else:
        for file_path in submissions_files + comments_files:
            print(f"  Processing: {file_path.name}...", end=" ", flush=True)

            if args.write_db:
                stats = await ingest_reddit_file(file_path, db_conn, universe if universe else None, args.max_samples, args.max_posts)
            else:
                stats = parse_reddit_file(file_path, universe if universe else None, args.max_samples, args.max_posts)

            if "error" in stats:
                print(f"ERROR: {stats['error']}")
                continue

            all_stats["total_posts"] += stats["total_posts"]
            all_stats["posts_with_tickers"] += stats["posts_with_tickers"]
            all_stats["trading_day_posts"] += stats["trading_day_posts"]
            all_stats["weekend_posts"] += stats["weekend_posts"]
            all_stats["ticker_counts"].update(stats["ticker_counts"])
            all_stats["subreddit_counts"].update(stats["subreddit_counts"])
            all_stats.setdefault("daily_counts", Counter()).update(stats.get("daily_counts", {}))
            all_stats.setdefault("sample_posts", []).extend(stats.get("sample_posts", []))
            all_stats["ticker_daily_counts"].update(stats.get("ticker_daily_counts", {}))
            all_stats["rows_inserted"] += stats.get("rows_inserted", 0)

            print(f"{stats['total_posts']:,} posts, {stats['posts_with_tickers']:,} with tickers")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70 + "\n")
    print(f"Total posts analyzed:     {all_stats['total_posts']:,}")
    print(f"Posts with tickers:       {all_stats['posts_with_tickers']:,} ({all_stats['posts_with_tickers']/max(1,all_stats['total_posts'])*100:.1f}%)")
    print(f"Trading day posts:        {all_stats['trading_day_posts']:,}")
    print(f"Weekend posts:            {all_stats['weekend_posts']:,}")
    if args.write_db:
        print(f"Rows inserted into DB:    {all_stats['rows_inserted']:,}")

    daily_counts = all_stats.get("daily_counts", {})
    if daily_counts:
        min_day = min(daily_counts.keys())
        max_day = max(daily_counts.keys())
        print(f"Date range covered:       {min_day} to {max_day}")

    print("\nTop 20 Ticker Mentions:")
    top_tickers = all_stats["ticker_counts"].most_common(20)
    if top_tickers:
        print(f"  {'Rank':<6}{'Ticker':<8}{'Mentions':>12}")
        print("  " + "-" * 30)
        for i, (ticker, count) in enumerate(top_tickers, 1):
            print(f"  {i:<6}{ticker:<8}{count:>12,}")
    else:
        print("  No ticker mentions found")

    sample_posts = all_stats.get("sample_posts", [])[:args.max_samples]
    if sample_posts:
        print("\nSample Posts (ts | ticker | snippet):")
        for post in sample_posts:
            tickers_str = ",".join(post["tickers"])
            print(f"  {post['ts']} | {tickers_str} | {post['snippet']}")

    pct_trading = all_stats["trading_day_posts"] / max(1, all_stats["total_posts"]) * 100
    if pct_trading >= 70:
        print("\n✅ ALIGNMENT: Good - Most posts are on trading days")
    elif pct_trading >= 50:
        print("\n⚠️  ALIGNMENT: Moderate - Mix of trading/weekend posts")
    else:
        print("\n❌ ALIGNMENT: Poor - Too many weekend posts")

    if args.write_db and db_conn:
        rows = [(day, ticker, count) for (day, ticker), count in all_stats["ticker_daily_counts"].items()]
        if rows:
            await db_conn.executemany(
                f"""
                INSERT INTO {REDDIT_DAILY_TABLE} (day, ticker, mentions_count)
                VALUES ($1, $2, $3)
                ON CONFLICT (day, ticker) DO UPDATE
                SET mentions_count = EXCLUDED.mentions_count
                """,
                rows,
            )
            print(f"\n✅ Reddit daily mentions written to DB ({REDDIT_DAILY_TABLE})")
        await _print_db_join_coverage(db_conn)

    if db_conn:
        await db_conn.close()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
        sys.exit(1)

