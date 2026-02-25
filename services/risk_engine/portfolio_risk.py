"""Portfolio-level risk models — ported from Lean Algorithm.Framework/Risk.

Implements:
  1. MaximumDrawdownPercentPortfolio — portfolio-wide drawdown kill switch
  2. TrailingStopRiskModel — per-position trailing stop loss
  3. MaximumSectorExposureModel — sector exposure caps

Source: Lean-master/Algorithm.Framework/Risk/
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog

logger = structlog.get_logger("portfolio_risk")


@dataclass
class PositionState:
    """Tracks a single open position for risk calculations."""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    highest_price: float  # for trailing stop
    sector: str = "unknown"
    entry_ts: Optional[datetime] = None

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_entry_price) * self.qty

    @property
    def market_value(self) -> float:
        return abs(self.current_price * self.qty)


class MaximumDrawdownPortfolio:
    """Portfolio-wide drawdown limiter.

    Source: Lean MaximumDrawdownPercentPortfolio.cs
    If portfolio drawdown exceeds threshold, triggers liquidation signal.
    """

    def __init__(self, max_drawdown_pct: float = 0.10):
        self.max_drawdown_pct = max_drawdown_pct
        self._peak_value: float = 0.0
        self._triggered = False

    def update(self, portfolio_value: float) -> tuple[bool, str]:
        """Update with current portfolio value.

        Returns (should_liquidate, reason).
        """
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

        if self._peak_value <= 0:
            return False, "ok"

        drawdown = (self._peak_value - portfolio_value) / self._peak_value

        if drawdown >= self.max_drawdown_pct:
            self._triggered = True
            reason = (
                f"portfolio_drawdown_kill: {drawdown:.2%} >= "
                f"{self.max_drawdown_pct:.2%} (peak={self._peak_value:.2f}, "
                f"current={portfolio_value:.2f})"
            )
            logger.critical("portfolio_drawdown_breached", drawdown=drawdown,
                          peak=self._peak_value, current=portfolio_value)
            return True, reason

        return False, "ok"

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    def reset(self) -> None:
        self._triggered = False
        self._peak_value = 0.0


class TrailingStopRisk:
    """Per-position trailing stop loss.

    Source: Lean TrailingStopRiskManagementModel.cs
    Tracks the highest price since entry; triggers exit when price drops
    by trailing_pct from the high.
    """

    def __init__(self, trailing_pct: float = 0.05):
        self.trailing_pct = trailing_pct

    def check(self, position: PositionState) -> tuple[bool, str]:
        """Check if position should be stopped out.

        Returns (should_exit, reason).
        """
        if position.qty == 0:
            return False, "no_position"

        # Update high watermark
        if position.current_price > position.highest_price:
            position.highest_price = position.current_price

        if position.highest_price <= 0:
            return False, "ok"

        drop = (position.highest_price - position.current_price) / position.highest_price

        if drop >= self.trailing_pct:
            reason = (
                f"trailing_stop: {position.symbol} dropped {drop:.2%} from "
                f"high {position.highest_price:.2f} → {position.current_price:.2f}"
            )
            logger.info("trailing_stop_triggered", symbol=position.symbol,
                       drop_pct=drop, high=position.highest_price,
                       current=position.current_price)
            return True, reason

        return False, "ok"


class MaximumSectorExposure:
    """Sector exposure cap.

    Source: Lean MaximumSectorExposureRiskManagementModel.cs
    No single sector can exceed max_sector_pct of total portfolio value.
    """

    def __init__(self, max_sector_pct: float = 0.30):
        self.max_sector_pct = max_sector_pct
        # Sector assignments
        self._sector_map: dict[str, str] = {
            "AAPL": "technology", "MSFT": "technology", "GOOG": "technology",
            "GOOGL": "technology", "NVDA": "technology", "META": "technology",
            "AMZN": "consumer_discretionary", "TSLA": "consumer_discretionary",
            "JPM": "financials", "BAC": "financials", "GS": "financials",
            "XOM": "energy", "CVX": "energy",
            "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
            "SPY": "index", "QQQ": "index",
        }

    def set_sector(self, symbol: str, sector: str) -> None:
        self._sector_map[symbol] = sector

    def get_sector(self, symbol: str) -> str:
        return self._sector_map.get(symbol, "unknown")

    def check_exposure(
        self,
        positions: list[PositionState],
        portfolio_value: float,
    ) -> list[tuple[str, str]]:
        """Check all sector exposures.

        Returns list of (sector, reason) for sectors exceeding the cap.
        """
        if portfolio_value <= 0:
            return []

        sector_exposure: dict[str, float] = {}
        for pos in positions:
            sector = self.get_sector(pos.symbol)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + pos.market_value

        violations = []
        for sector, exposure in sector_exposure.items():
            pct = exposure / portfolio_value
            if pct > self.max_sector_pct:
                reason = (
                    f"sector_exposure_breach: {sector} = {pct:.1%} "
                    f"(${exposure:.0f} / ${portfolio_value:.0f}) > "
                    f"{self.max_sector_pct:.0%} cap"
                )
                logger.warning("sector_exposure_exceeded",
                             sector=sector, pct=pct, exposure=exposure)
                violations.append((sector, reason))

        return violations

    def can_add_position(
        self,
        symbol: str,
        notional: float,
        positions: list[PositionState],
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """Check if adding a new position would breach sector cap."""
        if portfolio_value <= 0:
            return False, "no_portfolio_value"

        sector = self.get_sector(symbol)
        current_sector_exposure = sum(
            p.market_value for p in positions
            if self.get_sector(p.symbol) == sector
        )
        new_exposure = current_sector_exposure + notional
        pct = new_exposure / portfolio_value

        if pct > self.max_sector_pct:
            return False, (
                f"sector_cap_would_breach: {sector} would be {pct:.1%} "
                f"> {self.max_sector_pct:.0%}"
            )

        return True, "ok"


class UnifiedPortfolioRisk:
    """Combines all portfolio risk models into a single checker.

    Aggregates:
      - AWET risk engine (CVaR, position limits, daily loss, kill switch)
      - Lean MaximumDrawdownPercentPortfolio
      - Lean TrailingStopRiskManagementModel
      - Lean MaximumSectorExposureRiskManagementModel
    """

    def __init__(
        self,
        max_portfolio_drawdown: float = 0.10,
        trailing_stop_pct: float = 0.05,
        max_sector_pct: float = 0.30,
    ):
        self.drawdown = MaximumDrawdownPortfolio(max_portfolio_drawdown)
        self.trailing_stop = TrailingStopRisk(trailing_stop_pct)
        self.sector_exposure = MaximumSectorExposure(max_sector_pct)

    def check_portfolio(
        self,
        portfolio_value: float,
        positions: list[PositionState],
    ) -> list[str]:
        """Run all portfolio-level risk checks.

        Returns list of risk violation reasons (empty = all clear).
        """
        violations = []

        # Portfolio drawdown
        should_liquidate, reason = self.drawdown.update(portfolio_value)
        if should_liquidate:
            violations.append(reason)

        # Per-position trailing stops
        for pos in positions:
            should_exit, reason = self.trailing_stop.check(pos)
            if should_exit:
                violations.append(reason)

        # Sector exposure
        sector_violations = self.sector_exposure.check_exposure(
            positions, portfolio_value
        )
        for sector, reason in sector_violations:
            violations.append(reason)

        return violations

    def can_open_new(
        self,
        symbol: str,
        notional: float,
        positions: list[PositionState],
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """Check if a new position is allowed by all risk models."""
        if self.drawdown.is_triggered:
            return False, "portfolio_drawdown_kill_active"

        ok, reason = self.sector_exposure.can_add_position(
            symbol, notional, positions, portfolio_value
        )
        if not ok:
            return False, reason

        return True, "ok"
