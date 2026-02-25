"""Walk-forward cross-validation for TFT model training.

Implements rolling-window train/test splits:
  - Train on months [i, i+train_months)
  - Test on months [i+train_months, i+train_months+oos_months)
  - Step forward by oos_months

Each fold produces an independent model + OOS metrics.
Only models with OOS Sharpe >= threshold are eligible for promotion.

Stores results in TimescaleDB `walk_forward_results` table.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import structlog

logger = structlog.get_logger("walk_forward")

# Try optional imports
try:
    import asyncpg
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


@dataclass
class FoldSpec:
    """Specification for a single walk-forward fold."""
    fold_number: int
    in_sample_start: datetime
    in_sample_end: datetime
    oos_start: datetime
    oos_end: datetime


@dataclass
class FoldResult:
    """Result from training and evaluating one fold."""
    fold_number: int
    in_sample_start: datetime
    in_sample_end: datetime
    oos_start: datetime
    oos_end: datetime
    model_id: str
    onnx_path: str
    # In-sample metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_samples: int = 0
    # OOS metrics
    oos_total_return: float = 0.0
    oos_sharpe: float = 0.0
    oos_sortino: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_win_rate: float = 0.0
    oos_calmar: float = 0.0
    oos_samples: int = 0
    promoted: bool = False

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_samples": self.train_samples,
            "oos_total_return": self.oos_total_return,
            "oos_sharpe": self.oos_sharpe,
            "oos_sortino": self.oos_sortino,
            "oos_max_drawdown": self.oos_max_drawdown,
            "oos_win_rate": self.oos_win_rate,
            "oos_calmar": self.oos_calmar,
            "oos_samples": self.oos_samples,
        }


def generate_folds(
    data_start: datetime,
    data_end: datetime,
    train_months: int = 12,
    oos_months: int = 1,
    embargo_bars: int = 0,
    bar_duration_minutes: int = 1,
) -> list[FoldSpec]:
    """Generate walk-forward fold specifications.

    Args:
        data_start: Earliest available data timestamp
        data_end: Latest available data timestamp
        train_months: Number of months for in-sample window
        oos_months: Number of months for out-of-sample window
        embargo_bars: Number of bars to skip between IS end and OOS start
                      (prevents serial correlation leakage). Recommended:
                      lookback_window + max(horizons).
        bar_duration_minutes: Duration of one bar in minutes (1 for minute, 1440 for daily)

    Returns:
        List of FoldSpec objects
    """
    embargo_gap = timedelta(minutes=embargo_bars * bar_duration_minutes)
    folds = []
    fold_num = 0
    current_start = data_start

    while True:
        is_end = current_start + timedelta(days=train_months * 30)
        oos_start = is_end + embargo_gap
        oos_end = oos_start + timedelta(days=oos_months * 30)

        if oos_end > data_end:
            break

        folds.append(FoldSpec(
            fold_number=fold_num,
            in_sample_start=current_start,
            in_sample_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
        ))
        fold_num += 1
        current_start += timedelta(days=oos_months * 30)

    logger.info("folds_generated", total_folds=len(folds),
                embargo_bars=embargo_bars,
                data_start=data_start.isoformat(), data_end=data_end.isoformat())
    return folds


def compute_oos_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    risk_free_rate: float = 0.0,
    bars_per_day: int = 1,
) -> dict[str, float]:
    """Compute out-of-sample quant metrics from predicted vs actual returns.

    Args:
        predictions: Model predicted returns (N,)
        actuals: Actual realized returns (N,)
        risk_free_rate: Annualized risk-free rate
        bars_per_day: Number of bars per trading day (1=daily, 390=minute)

    Returns:
        Dict with sharpe, sortino, max_drawdown, win_rate, total_return, calmar
    """
    if len(predictions) == 0 or len(actuals) == 0:
        return {
            "oos_sharpe": 0.0, "oos_sortino": 0.0, "oos_max_drawdown": 0.0,
            "oos_win_rate": 0.0, "oos_total_return": 0.0, "oos_calmar": 0.0,
            "oos_samples": 0,
        }

    # Strategy returns: go long when predicted positive, short when negative
    signal = np.sign(predictions)
    strategy_returns = signal * actuals

    n = len(strategy_returns)
    mean_ret = float(np.mean(strategy_returns))
    std_ret = float(np.std(strategy_returns))

    # Annualize using bars_per_day: sqrt(252 trading days * bars_per_day)
    bars_per_year = 252 * bars_per_day
    annualization_factor = np.sqrt(bars_per_year)

    # Sharpe (annualized)
    rf_per_bar = risk_free_rate / bars_per_year
    if std_ret > 1e-10:
        sharpe = (mean_ret - rf_per_bar) / std_ret * annualization_factor
    else:
        sharpe = 0.0

    # Sortino (downside deviation only)
    downside = strategy_returns[strategy_returns < 0]
    if len(downside) > 0:
        downside_std = float(np.std(downside))
        if downside_std > 1e-10:
            sortino = (mean_ret - rf_per_bar) / downside_std * annualization_factor
        else:
            sortino = 0.0
    else:
        sortino = float("inf") if mean_ret > 0 else 0.0

    # Max drawdown
    cumulative = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Total return
    total_return = float(np.sum(strategy_returns))

    # Win rate
    wins = int(np.sum(strategy_returns > 0))
    total_trades = int(np.sum(signal != 0))
    win_rate = wins / max(total_trades, 1)

    # Calmar ratio (annual return / max drawdown)
    annual_return = mean_ret * bars_per_year
    calmar = annual_return / max(max_drawdown, 1e-10)

    return {
        "oos_sharpe": round(sharpe, 4),
        "oos_sortino": round(sortino, 4),
        "oos_max_drawdown": round(max_drawdown, 6),
        "oos_win_rate": round(win_rate, 4),
        "oos_total_return": round(total_return, 6),
        "oos_calmar": round(calmar, 4),
        "oos_samples": n,
    }


class WalkForwardValidator:
    """Orchestrates walk-forward training + OOS evaluation.

    Usage:
        validator = WalkForwardValidator(config)
        results = await validator.run(symbol_data)
        best = validator.select_best(results)
    """

    def __init__(
        self,
        train_months: int = 12,
        oos_months: int = 1,
        min_folds: int = 6,
        min_oos_sharpe: float = 0.5,
        min_win_rate: float = 0.45,
        embargo_bars: int = 0,
        bars_per_day: int = 1,
    ) -> None:
        self.train_months = train_months
        self.oos_months = oos_months
        self.min_folds = min_folds
        self.min_oos_sharpe = min_oos_sharpe
        self.min_win_rate = min_win_rate
        self.embargo_bars = embargo_bars
        self.bars_per_day = bars_per_day
        self.run_id = str(uuid.uuid4())

    def generate_folds(
        self, data_start: datetime, data_end: datetime
    ) -> list[FoldSpec]:
        bar_duration_minutes = 1440 // self.bars_per_day if self.bars_per_day > 0 else 1
        return generate_folds(
            data_start, data_end, self.train_months, self.oos_months,
            embargo_bars=self.embargo_bars,
            bar_duration_minutes=bar_duration_minutes,
        )

    def evaluate_fold_oos(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> dict[str, float]:
        """Compute OOS metrics for a single fold."""
        return compute_oos_metrics(predictions, actuals, bars_per_day=self.bars_per_day)

    def select_best(
        self, results: list[FoldResult]
    ) -> Optional[FoldResult]:
        """Select the model to deploy based on walk-forward validation.

        Strategy: Use median OOS Sharpe and win rate across all folds as
        quality gates. If gates pass, deploy the LATEST fold (highest
        fold_number) — not the best-performing fold — to avoid selection
        bias.

        Returns None if:
          - Too few folds completed
          - Median OOS Sharpe < threshold
          - Median win rate < threshold
        """
        if len(results) < self.min_folds:
            logger.warning(
                "insufficient_folds",
                completed=len(results),
                required=self.min_folds,
            )
            return None

        sharpes = [r.oos_sharpe for r in results]
        win_rates = [r.oos_win_rate for r in results]
        median_sharpe = float(np.median(sharpes))
        median_win_rate = float(np.median(win_rates))

        logger.info(
            "walk_forward_summary",
            folds=len(results),
            median_sharpe=median_sharpe,
            median_win_rate=median_win_rate,
            min_sharpe=min(sharpes),
            max_sharpe=max(sharpes),
        )

        if median_sharpe < self.min_oos_sharpe:
            logger.warning(
                "model_rejected_low_sharpe",
                median_sharpe=median_sharpe,
                threshold=self.min_oos_sharpe,
            )
            return None

        if median_win_rate < self.min_win_rate:
            logger.warning(
                "model_rejected_low_win_rate",
                median_win_rate=median_win_rate,
                threshold=self.min_win_rate,
            )
            return None

        # Deploy the LATEST fold (most recent data), not the best OOS Sharpe.
        # The walk-forward test validates the methodology; individual fold
        # performance is not a selection criterion.
        latest = max(results, key=lambda r: r.fold_number)
        latest.promoted = True
        logger.info(
            "model_selected_for_promotion",
            model_id=latest.model_id,
            oos_sharpe=latest.oos_sharpe,
            oos_win_rate=latest.oos_win_rate,
            fold=latest.fold_number,
            best_fold_sharpe=max(sharpes),
        )
        return latest

    async def persist_results(
        self,
        results: list[FoldResult],
        db_dsn: str,
    ) -> None:
        """Store walk-forward results in TimescaleDB."""
        if not PG_AVAILABLE:
            logger.warning("asyncpg_not_available_skipping_persist")
            return

        pool = await asyncpg.create_pool(dsn=db_dsn, min_size=1, max_size=3)
        try:
            async with pool.acquire() as conn:
                for r in results:
                    await conn.execute(
                        """
                        INSERT INTO walk_forward_results
                            (run_id, fold, model_id,
                             in_sample_start, in_sample_end,
                             oos_start, oos_end,
                             params, metrics, promoted)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (run_id, fold) DO UPDATE SET
                            metrics = EXCLUDED.metrics,
                            promoted = EXCLUDED.promoted
                        """,
                        uuid.UUID(self.run_id),
                        r.fold_number,
                        r.model_id,
                        r.in_sample_start,
                        r.in_sample_end,
                        r.oos_start,
                        r.oos_end,
                        json.dumps({
                            "train_months": self.train_months,
                            "oos_months": self.oos_months,
                        }),
                        json.dumps(r.metrics_dict()),
                        r.promoted,
                    )
            logger.info("walk_forward_results_persisted", run_id=self.run_id, folds=len(results))
        finally:
            await pool.close()

    def log_to_mlflow(self, results: list[FoldResult]) -> None:
        """Log all fold results to MLflow."""
        if not MLFLOW_AVAILABLE:
            logger.warning("mlflow_not_available_skipping_log")
            return

        with mlflow.start_run(run_name=f"walk_forward_{self.run_id[:8]}"):
            mlflow.log_param("train_months", self.train_months)
            mlflow.log_param("oos_months", self.oos_months)
            mlflow.log_param("total_folds", len(results))

            for r in results:
                with mlflow.start_run(
                    run_name=f"fold_{r.fold_number}", nested=True
                ):
                    mlflow.log_params({
                        "fold": r.fold_number,
                        "is_start": r.in_sample_start.isoformat(),
                        "is_end": r.in_sample_end.isoformat(),
                        "oos_start": r.oos_start.isoformat(),
                        "oos_end": r.oos_end.isoformat(),
                    })
                    mlflow.log_metrics(r.metrics_dict())
                    mlflow.log_metric("promoted", 1.0 if r.promoted else 0.0)

            # Log aggregate metrics
            sharpes = [r.oos_sharpe for r in results]
            mlflow.log_metrics({
                "median_oos_sharpe": float(np.median(sharpes)),
                "mean_oos_sharpe": float(np.mean(sharpes)),
                "min_oos_sharpe": float(np.min(sharpes)),
                "max_oos_sharpe": float(np.max(sharpes)),
                "median_win_rate": float(np.median([r.oos_win_rate for r in results])),
            })
