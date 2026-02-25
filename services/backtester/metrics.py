"""Quant-grade performance metrics — Sharpe, Sortino, Calmar, Alpha, Beta.

Provides calculations that match Lean Report output:
  - Annualized Sharpe Ratio
  - Annualized Sortino Ratio
  - Calmar Ratio (annual return / max drawdown)
  - Alpha and Beta vs benchmark (SPY)
  - Information Ratio
  - Profit Factor
  - Max Drawdown (duration and depth)

Source: Lean-master/Report/
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np


@dataclass
class PerformanceReport:
    """Full performance metrics report."""
    # Returns
    total_return: float = 0.0
    cagr: float = 0.0
    annual_return: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    annual_volatility: float = 0.0
    downside_deviation: float = 0.0

    # Benchmark-relative
    alpha: float = 0.0
    beta: float = 0.0

    # Trade stats
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_period_bars: float = 0.0

    # Exposure
    avg_exposure: float = 0.0    # fraction of capital deployed


def compute_metrics(
    strategy_returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.04,
    trading_days_per_year: int = 252,
    bars_per_day: int = 1,
) -> PerformanceReport:
    """Compute full quant metrics from daily/bar returns.

    Args:
        strategy_returns: Array of period returns (e.g., daily or per-bar)
        benchmark_returns: Optional benchmark returns for alpha/beta
        risk_free_rate: Annual risk-free rate (default 4%)
        trading_days_per_year: 252 for daily
        bars_per_day: Number of bars per trading day (1=daily, 390=minute)

    Returns:
        PerformanceReport with all metrics
    """
    report = PerformanceReport()
    n = len(strategy_returns)
    if n == 0:
        return report

    bars_per_year = trading_days_per_year * bars_per_day

    # Total return
    cumulative = np.cumprod(1 + strategy_returns)
    report.total_return = float(cumulative[-1] - 1)

    # CAGR
    years = n / bars_per_year
    if years > 0 and cumulative[-1] > 0:
        report.cagr = float(cumulative[-1] ** (1 / years) - 1)
    report.annual_return = report.cagr

    # Volatility (annualized)
    vol = float(np.std(strategy_returns)) * np.sqrt(bars_per_year)
    report.annual_volatility = vol

    # Sharpe ratio (annualized)
    mean_ret = float(np.mean(strategy_returns))
    std_ret = float(np.std(strategy_returns))
    rf_per_bar = risk_free_rate / bars_per_year
    if std_ret > 1e-10:
        report.sharpe_ratio = round(
            (mean_ret - rf_per_bar) / std_ret * np.sqrt(bars_per_year), 4
        )

    # Sortino (downside deviation)
    downside = strategy_returns[strategy_returns < 0]
    if len(downside) > 0:
        downside_std = float(np.std(downside))
        report.downside_deviation = downside_std
        if downside_std > 1e-10:
            report.sortino_ratio = round(
                (mean_ret - rf_per_bar) / downside_std * np.sqrt(bars_per_year), 4
            )

    # Max drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    report.max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Max drawdown duration
    in_drawdown = drawdowns > 0
    if np.any(in_drawdown):
        dd_starts = np.where(np.diff(np.concatenate(([False], in_drawdown)).astype(int)) == 1)[0]
        dd_ends = np.where(np.diff(np.concatenate((in_drawdown, [False])).astype(int)) == -1)[0]
        if len(dd_starts) > 0 and len(dd_ends) > 0:
            durations = dd_ends[:len(dd_starts)] - dd_starts[:len(dd_ends)]
            report.max_drawdown_duration_days = int(np.max(durations)) if len(durations) > 0 else 0

    # Calmar ratio
    if report.max_drawdown > 1e-10:
        report.calmar_ratio = round(report.cagr / report.max_drawdown, 4)

    # Alpha and Beta vs benchmark
    if benchmark_returns is not None and len(benchmark_returns) == n:
        bm = benchmark_returns
        cov = np.cov(strategy_returns, bm)
        var_bm = float(np.var(bm))
        if var_bm > 1e-10:
            report.beta = round(float(cov[0, 1] / var_bm), 4)
        report.alpha = round(
            float(np.mean(strategy_returns) - report.beta * np.mean(bm)) * bars_per_year, 4
        )

        # Information ratio
        tracking_error = float(np.std(strategy_returns - bm)) * np.sqrt(bars_per_year)
        if tracking_error > 1e-10:
            excess = float(np.mean(strategy_returns - bm)) * bars_per_year
            report.information_ratio = round(excess / tracking_error, 4)

    # Trade-level stats (treating each return as a "trade")
    report.total_trades = n
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    total_trades_nonzero = len(wins) + len(losses)
    report.win_rate = round(len(wins) / max(total_trades_nonzero, 1), 4)

    if len(wins) > 0:
        report.avg_win = round(float(np.mean(wins)), 6)
        report.largest_win = round(float(np.max(wins)), 6)
    if len(losses) > 0:
        report.avg_loss = round(float(np.mean(losses)), 6)
        report.largest_loss = round(float(np.min(losses)), 6)

    # Profit factor
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0
    if gross_loss > 1e-10:
        report.profit_factor = round(gross_profit / gross_loss, 4)

    return report
