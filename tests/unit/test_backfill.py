"""Unit tests for Polygon backfill loader."""

from __future__ import annotations

import gzip
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.backfill.polygon_loader import PolygonBar, PolygonCSVLoader


@pytest.fixture
def sample_csv_data() -> str:
    return """ticker,volume,open,close,high,low,window_start,transactions
AAPL,17742,117.44,117.26,117.54,117.06,1751376600000000000,104
AAPL,1962,117.3,117.27,117.63,117.27,1751376660000000000,86
MSFT,3767,117.28,117.305,117.91,117.28,1751376720000000000,98
GOOG,594,117.11,117.11,117.11,117.11,1751376780000000000,18
"""


@pytest.fixture
def temp_data_dir(sample_csv_data: str) -> Path:
    """Create temp directory with test CSV data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the directory structure: data_dir/July/2025-07-01.csv.gz
        minute_dir = Path(tmpdir) / "July"
        minute_dir.mkdir(parents=True)
        csv_path = minute_dir / "2025-07-01.csv.gz"
        with gzip.open(csv_path, "wt", encoding="utf-8") as f:
            f.write(sample_csv_data)
        yield Path(tmpdir)


class TestPolygonBar:
    def test_idempotency_key_stable(self) -> None:
        """Idempotency key should be deterministic."""
        bar = PolygonBar(
            ticker="AAPL",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            vwap=100.2,
            transactions=50,
            timestamp=datetime(2025, 7, 1, 9, 30, 0, tzinfo=timezone.utc),
        )
        key1 = bar.idempotency_key()
        key2 = bar.idempotency_key()
        assert key1 == key2
        assert len(key1) == 32

    def test_idempotency_key_differs_by_symbol(self) -> None:
        """Different symbols should have different keys."""
        ts = datetime(2025, 7, 1, 9, 30, 0, tzinfo=timezone.utc)
        bar1 = PolygonBar("AAPL", 100, 101, 99, 100.5, 1000, 100.2, 50, ts)
        bar2 = PolygonBar("MSFT", 100, 101, 99, 100.5, 1000, 100.2, 50, ts)
        assert bar1.idempotency_key() != bar2.idempotency_key()


class TestPolygonCSVLoader:
    def test_discover_files(self, temp_data_dir: Path) -> None:
        """Should discover .csv.gz files."""
        loader = PolygonCSVLoader(str(temp_data_dir))
        files = loader.discover_files()
        assert len(files) == 1
        assert files[0].name == "2025-07-01.csv.gz"

    def test_load_file(self, temp_data_dir: Path) -> None:
        """Should parse CSV correctly."""
        loader = PolygonCSVLoader(str(temp_data_dir))
        files = loader.discover_files()
        bars = list(loader.load_file(files[0]))
        assert len(bars) == 4
        assert bars[0].ticker == "AAPL"
        assert bars[0].close == 117.26
        assert bars[0].volume == 17742.0

    def test_symbol_filter(self, temp_data_dir: Path) -> None:
        """Should filter by symbol."""
        loader = PolygonCSVLoader(
            str(temp_data_dir),
            symbols=["AAPL"],
        )
        bars = list(loader.load_all())
        assert len(bars) == 2
        assert all(b.ticker == "AAPL" for b in bars)

    def test_timestamp_parsing(self, temp_data_dir: Path) -> None:
        """Should parse nanosecond timestamps correctly."""
        loader = PolygonCSVLoader(str(temp_data_dir))
        bars = list(loader.load_all())
        ts = bars[0].timestamp
        assert ts.tzinfo == timezone.utc
        assert ts.year == 2025
        assert ts.month == 7

    def test_batched_loading(self, temp_data_dir: Path) -> None:
        """Should batch records correctly."""
        loader = PolygonCSVLoader(str(temp_data_dir))
        batches = list(loader.load_batched(batch_size=2))
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2

    def test_date_filter(self, temp_data_dir: Path) -> None:
        """Should filter by date range."""
        loader = PolygonCSVLoader(
            str(temp_data_dir),
            start_date=datetime(2025, 7, 2, tzinfo=timezone.utc),
        )
        files = loader.discover_files()
        assert len(files) == 0
