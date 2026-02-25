"""Trading signal filters — market hours, liquidity, spread, earnings blackout.

These filters reject signals that are unlikely to execute cleanly:
  - Market hours: only trade 9:30 AM – 3:45 PM ET
  - Minimum liquidity: 10-day avg volume >= threshold
  - Maximum spread: reject if bid-ask > threshold
  - Earnings blackout: reject signals N days before/after earnings
"""
from __future__ import annotations

import os
from datetime import datetime, time, timezone, timedelta
from typing import Optional

import structlog

logger = structlog.get_logger("signal_filters")

# Configurable thresholds
MIN_AVG_VOLUME = int(os.getenv("FILTER_MIN_AVG_VOLUME", "500000"))
MAX_SPREAD_PCT = float(os.getenv("FILTER_MAX_SPREAD_PCT", "0.001"))  # 0.1%
MARKET_OPEN = time(9, 30)    # ET
MARKET_CLOSE = time(15, 45)  # ET (15 min before close)
EARNINGS_BLACKOUT_DAYS = int(os.getenv("FILTER_EARNINGS_BLACKOUT_DAYS", "2"))


class SignalFilter:
    """Pre-execution signal quality filters.

    Usage:
        f = SignalFilter()
        ok, reason = f.check_all(symbol, price, avg_volume, spread, ts)
    """

    def __init__(
        self,
        min_avg_volume: int = MIN_AVG_VOLUME,
        max_spread_pct: float = MAX_SPREAD_PCT,
        earnings_blackout_days: int = EARNINGS_BLACKOUT_DAYS,
    ):
        self.min_avg_volume = min_avg_volume
        self.max_spread_pct = max_spread_pct
        self.earnings_blackout_days = earnings_blackout_days
        # Earnings calendar — populated externally
        self._earnings_dates: dict[str, list[datetime]] = {}

    def set_earnings_dates(self, symbol: str, dates: list[datetime]) -> None:
        """Set known earnings dates for a symbol."""
        self._earnings_dates[symbol] = sorted(dates)

    def check_market_hours(self, ts: datetime) -> tuple[bool, str]:
        """Check if timestamp falls within tradeable market hours (ET)."""
        # Convert to ET (UTC-5 / UTC-4 depending on DST)
        # Simplified: assume UTC-5
        et_hour = (ts.hour - 5) % 24
        et_time = time(et_hour, ts.minute)

        if et_time < MARKET_OPEN or et_time > MARKET_CLOSE:
            return False, f"outside_market_hours: {et_time} not in {MARKET_OPEN}-{MARKET_CLOSE}"
        # Skip weekends
        if ts.weekday() >= 5:
            return False, "weekend"
        return True, "ok"

    def check_liquidity(self, symbol: str, avg_volume_10d: float) -> tuple[bool, str]:
        """Check 10-day average volume meets minimum."""
        if avg_volume_10d < self.min_avg_volume:
            return False, f"low_liquidity: {symbol} avg_vol={avg_volume_10d:.0f} < {self.min_avg_volume}"
        return True, "ok"

    def check_spread(self, symbol: str, bid: float, ask: float) -> tuple[bool, str]:
        """Check bid-ask spread is within bounds."""
        if bid <= 0 or ask <= 0:
            return True, "ok"  # No quote data available — pass through
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid if mid > 0 else 0
        if spread_pct > self.max_spread_pct:
            return False, f"wide_spread: {symbol} spread={spread_pct:.4%} > {self.max_spread_pct:.4%}"
        return True, "ok"

    def check_earnings_blackout(
        self, symbol: str, ts: datetime
    ) -> tuple[bool, str]:
        """Check if we're within earnings blackout window."""
        dates = self._earnings_dates.get(symbol, [])
        for ed in dates:
            delta = abs((ts - ed).days)
            if delta <= self.earnings_blackout_days:
                return False, f"earnings_blackout: {symbol} earnings on {ed.date()}, delta={delta}d"
        return True, "ok"

    def check_all(
        self,
        symbol: str,
        ts: datetime,
        avg_volume_10d: float = float("inf"),
        bid: float = 0.0,
        ask: float = 0.0,
    ) -> tuple[bool, str]:
        """Run all filters. Returns (pass, reason)."""
        checks = [
            self.check_market_hours(ts),
            self.check_liquidity(symbol, avg_volume_10d),
            self.check_spread(symbol, bid, ask),
            self.check_earnings_blackout(symbol, ts),
        ]
        for passed, reason in checks:
            if not passed:
                logger.debug("signal_filtered", symbol=symbol, reason=reason)
                return False, reason
        return True, "all_filters_passed"
