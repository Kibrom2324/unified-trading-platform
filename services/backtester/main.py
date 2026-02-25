from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import structlog
from fastapi import HTTPException
from pydantic import BaseModel

from src.agents.base_agent import BaseAgent
from src.core.config import load_settings
from src.core.logging import set_correlation_id
from src.models.events_trade_decision import TradeDecisionEvent
from src.monitoring.metrics import EVENTS_PROCESSED
from src.streaming.topics import TRADE_DECISIONS
from src.core.trade_decision import decide_trade

logger = structlog.get_logger("backtester")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _parse_date(value: str, end: bool = False) -> datetime:
    if "T" in value:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(value)
        if end:
            dt = dt + timedelta(days=1) - timedelta(seconds=1)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class BacktestRequest(BaseModel):
    symbols: list[str]
    start: str
    end: str
    initial_cash: float = 10000.0
    max_pos_pct: float = 0.1
    slippage_bps: float = 5.0  # More realistic default: 5 bps
    fee_per_trade: float = 1.0  # $1 per trade
    # Risk controls
    stop_loss_pct: float = 0.02  # 2% stop-loss
    take_profit_pct: float = 0.05  # 5% take-profit
    cooldown_bars: int = 3  # Bars to wait after exit before re-entry
    max_total_exposure_pct: float = 0.5  # Max 50% of portfolio in positions


class TradeSummary(BaseModel):
    symbol: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    holding_seconds: float


class BacktestMetrics(BaseModel):
    total_return: float
    cagr: float
    win_rate: float
    max_drawdown: float
    num_trades: int
    avg_trade_return: float
    sharpe_like: float
    exposure_pct: float
    avg_holding_seconds: float
    final_equity: float


class BacktestResponse(BaseModel):
    run_id: UUID
    metrics: BacktestMetrics
    top_trades: list[TradeSummary]
    equity_curve: list[dict[str, Any]]
    message: str | None = None


@dataclass
class PricePoint:
    ts: datetime
    price: float


@dataclass
class TradeAction:
    ts: datetime
    symbol: str
    decision: str


class BacktesterService:
    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._settings = load_settings()

    async def connect(self) -> None:
        if self._pool is not None:
            return
        dsn = (
            f"postgresql://{_env('POSTGRES_USER', 'awet')}:{_env('POSTGRES_PASSWORD', 'awet')}"
            f"@{_env('POSTGRES_HOST', 'localhost')}:{_env('POSTGRES_PORT', '5433')}"
            f"/{_env('POSTGRES_DB', 'awet')}"
        )
        self._pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
        await self._ensure_backtest_table()

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def _ensure_backtest_table(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    run_id UUID PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    params JSONB NOT NULL,
                    metrics JSONB NOT NULL
                )
                """
            )

    async def ensure_market_data(
        self,
        symbols: list[str],
        start_ts: datetime,
        end_ts: datetime,
    ) -> dict[str, Any]:
        await self.connect()
        assert self._pool is not None
        missing_tables: list[str] = []
        missing_rows: list[str] = []
        details: dict[str, Any] = {}

        async with self._pool.acquire() as conn:
            for table in ("market_raw_day", "market_raw_minute", "predictions_tft", "audit_events"):
                exists = await conn.fetchval("SELECT to_regclass($1)", table)
                if exists is None:
                    missing_tables.append(table)

            decision_count = 0
            if "audit_events" not in missing_tables:
                decision_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM audit_events
                    WHERE event_type = $1 AND symbol = ANY($2) AND ts BETWEEN $3 AND $4
                    """,
                    TRADE_DECISIONS,
                    symbols,
                    start_ts,
                    end_ts,
                )
                details["trade_decisions_rows"] = int(decision_count or 0)

            if "predictions_tft" not in missing_tables:
                pred_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM predictions_tft
                    WHERE ticker = ANY($1) AND ts BETWEEN $2 AND $3
                    """,
                    symbols,
                    start_ts,
                    end_ts,
                )
                details["predictions_tft_rows"] = int(pred_count or 0)
                if pred_count == 0 and decision_count == 0:
                    missing_rows.append("predictions_tft")

            day_count = 0
            minute_count = 0
            for table in ("market_raw_day", "market_raw_minute"):
                if table in missing_tables:
                    continue
                count = await conn.fetchval(
                    f"""
                    SELECT COUNT(*) FROM {table}
                    WHERE ticker = ANY($1) AND ts BETWEEN $2 AND $3
                    """,
                    symbols,
                    start_ts,
                    end_ts,
                )
                details[f"{table}_rows"] = int(count or 0)
                if table == "market_raw_day":
                    day_count = int(count or 0)
                else:
                    minute_count = int(count or 0)
            if day_count == 0 and minute_count == 0:
                missing_rows.append("market_raw_day/market_raw_minute")

        suggested_commands = []
        if any(table in missing_rows for table in ("market_raw_day/market_raw_minute",)):
            suggested_commands.append(
                "python scripts/ingest_market_data.py --symbols AAPL,MSFT --start YYYY-MM-DD --end YYYY-MM-DD"
            )
        if "predictions_tft" in missing_rows:
            suggested_commands.append(
                "python -m src.agents.time_series_prediction  # populate predictions_tft via pipeline"
            )

        ok = not missing_tables and not any(
            table in missing_rows for table in ("market_raw_day/market_raw_minute", "predictions_tft")
        )
        return {
            "ok": ok,
            "missing_tables": missing_tables,
            "missing_rows": missing_rows,
            "details": details,
            "suggested_commands": suggested_commands,
        }

    async def _fetch_trade_decisions(
        self,
        symbols: list[str],
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[dict[str, Any]]:
        assert self._pool is not None
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT payload
                    FROM audit_events
                    WHERE event_type = $1
                      AND symbol = ANY($2)
                      AND ts BETWEEN $3 AND $4
                    ORDER BY ts ASC
                    """,
                    TRADE_DECISIONS,
                    symbols,
                    start_ts,
                    end_ts,
                )
        except asyncpg.UndefinedTableError:
            logger.warning("audit_events_missing_falling_back")
            return []
        payloads: list[dict[str, Any]] = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            payloads.append(dict(payload))
        return payloads

    async def _fetch_predictions(
        self,
        symbols: list[str],
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[dict[str, Any]]:
        assert self._pool is not None
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT ticker, ts, direction, confidence, q10, q50, q90
                    FROM predictions_tft
                    WHERE ticker = ANY($1)
                      AND ts BETWEEN $2 AND $3
                    ORDER BY ts ASC
                    """,
                    symbols,
                    start_ts,
                    end_ts,
                )
        except asyncpg.UndefinedTableError as exc:
            raise ValueError("predictions_tft table missing; run ingestion/prediction first") from exc
        return [dict(row) for row in rows]

    async def _fetch_prices(
        self,
        symbols: list[str],
        start_ts: datetime,
        end_ts: datetime,
    ) -> dict[str, list[PricePoint]]:
        assert self._pool is not None
        data: dict[str, list[PricePoint]] = {}
        async with self._pool.acquire() as conn:
            for symbol in symbols:
                try:
                    rows = await conn.fetch(
                        """
                        SELECT ts, open
                        FROM market_raw_day
                        WHERE ticker = $1 AND ts BETWEEN $2 AND $3
                        ORDER BY ts ASC
                        """,
                        symbol,
                        start_ts,
                        end_ts,
                    )
                except asyncpg.UndefinedTableError as exc:
                    raise ValueError("market_raw_day table missing; run market data load") from exc
                if not rows:
                    try:
                        rows = await conn.fetch(
                            """
                            SELECT ts, open
                            FROM market_raw_minute
                            WHERE ticker = $1 AND ts BETWEEN $2 AND $3
                            ORDER BY ts ASC
                            """,
                            symbol,
                            start_ts,
                            end_ts,
                        )
                    except asyncpg.UndefinedTableError as exc:
                        raise ValueError("market_raw_minute table missing; run market data load") from exc
                points = [PricePoint(ts=row["ts"], price=float(row["open"])) for row in rows]
                data[symbol] = points
                if points:
                    logger.info(
                        "backtest_prices_loaded",
                        symbol=symbol,
                        rows=len(points),
                        min_ts=str(points[0].ts),
                        max_ts=str(points[-1].ts),
                    )
        return data

    async def run_backtest(self, request: BacktestRequest) -> BacktestResponse:
        await self.connect()
        start_ts = _parse_date(request.start)
        end_ts = _parse_date(request.end, end=True)
        symbols = [s.upper() for s in request.symbols]

        if not symbols:
            raise ValueError("No symbols provided")
        if end_ts <= start_ts:
            raise ValueError("End date must be after start date")

        logger.info("backtest_data_load_start", symbols=symbols, start=str(start_ts), end=str(end_ts))
        decision_payloads = await self._fetch_trade_decisions(symbols, start_ts, end_ts)
        use_decisions = len(decision_payloads) > 0

        if use_decisions:
            logger.info("backtest_decisions_loaded", count=len(decision_payloads))
            decisions = [TradeDecisionEvent.model_validate(payload) for payload in decision_payloads]
        else:
            predictions = await self._fetch_predictions(symbols, start_ts, end_ts)
            logger.info("backtest_predictions_loaded", count=len(predictions))
            if not predictions:
                empty_metrics = _empty_metrics(request.initial_cash)
                run_id = uuid4()
                await self._persist_run(run_id, request, empty_metrics)
                return BacktestResponse(
                    run_id=run_id,
                    metrics=empty_metrics,
                    top_trades=[],
                    equity_curve=[],
                    message="No predictions or decisions found for requested range",
                )
            decisions = []
            for row in predictions:
                decision = decide_trade(
                    {
                        "direction": row.get("direction", "neutral"),
                        "confidence": float(row.get("confidence", 0.0)),
                    },
                    self._settings.trader_decision_agent.min_confidence,
                )
                reason = "generated_from_prediction"
                decisions.append(
                    TradeDecisionEvent(
                        idempotency_key=f"backtest:{row['ticker']}:{row['ts'].isoformat()}",
                        symbol=row["ticker"],
                        source="backtester",
                        correlation_id=uuid4(),
                        ts=row["ts"],
                        decision=decision,
                        confidence=float(row.get("confidence", 0.0)),
                        reason=reason,
                    )
                )

        prices = await self._fetch_prices(symbols, start_ts, end_ts + timedelta(days=1))
        if not any(prices.values()):
            logger.warning("backtest_no_prices", symbols=symbols)
            empty_metrics = _empty_metrics(request.initial_cash)
            run_id = uuid4()
            await self._persist_run(run_id, request, empty_metrics)
            return BacktestResponse(
                run_id=run_id,
                metrics=empty_metrics,
                top_trades=[],
                equity_curve=[],
                message="No market data found for requested symbols/time range",
            )
        logger.info("backtest_prices_loaded", symbols=symbols)

        actions_by_symbol: dict[str, dict[datetime, list[TradeAction]]] = {s: {} for s in symbols}
        for symbol in symbols:
            price_points = prices.get(symbol, [])
            if not price_points:
                continue
            ts_list = [p.ts for p in price_points]
            for decision in decisions:
                if decision.symbol != symbol:
                    continue
                if decision.decision not in ("BUY", "SELL"):
                    continue
                idx = _bisect_next(ts_list, decision.ts)
                if idx is None:
                    continue
                fill_ts = ts_list[idx]
                actions_by_symbol[symbol].setdefault(fill_ts, []).append(
                    TradeAction(ts=fill_ts, symbol=symbol, decision=decision.decision)
                )

        cash = request.initial_cash
        positions: dict[str, dict[str, Any]] = {}
        equity_curve: list[dict[str, Any]] = []
        exposure_series: list[float] = []
        trades: list[TradeSummary] = []
        last_prices: dict[str, float] = {}
        cooldowns: dict[str, int] = {}  # symbol -> bars remaining in cooldown
        total_costs: float = 0.0  # Track total transaction costs

        timeline: list[tuple[datetime, str, float]] = []
        for symbol, points in prices.items():
            for point in points:
                timeline.append((point.ts, symbol, point.price))
        timeline.sort(key=lambda x: x[0])

        logger.info("backtest_simulation_start")
        for ts, symbol, price in timeline:
            last_prices[symbol] = price

            # Decrement cooldowns
            if symbol in cooldowns:
                cooldowns[symbol] -= 1
                if cooldowns[symbol] <= 0:
                    del cooldowns[symbol]

            # Check stop-loss and take-profit for existing positions
            if symbol in positions:
                position = positions[symbol]
                entry_price = position["entry_price"]
                current_return = (price - entry_price) / entry_price

                # Stop-loss triggered
                if current_return <= -request.stop_loss_pct:
                    fill_price = price * (1 - request.slippage_bps / 10000)
                    cost = request.fee_per_trade
                    proceeds = position["qty"] * fill_price - cost
                    total_costs += cost + (request.slippage_bps / 10000) * position["qty"] * price
                    cash += proceeds
                    pnl = (fill_price - entry_price) * position["qty"] - cost
                    return_pct = (fill_price - entry_price) / entry_price
                    holding_seconds = (ts - position["entry_ts"]).total_seconds()
                    trades.append(
                        TradeSummary(
                            symbol=symbol,
                            entry_ts=position["entry_ts"],
                            exit_ts=ts,
                            entry_price=entry_price,
                            exit_price=fill_price,
                            pnl=pnl,
                            return_pct=return_pct,
                            holding_seconds=holding_seconds,
                        )
                    )
                    del positions[symbol]
                    cooldowns[symbol] = request.cooldown_bars  # Start cooldown
                    continue

                # Take-profit triggered
                if current_return >= request.take_profit_pct:
                    fill_price = price * (1 - request.slippage_bps / 10000)
                    cost = request.fee_per_trade
                    proceeds = position["qty"] * fill_price - cost
                    total_costs += cost + (request.slippage_bps / 10000) * position["qty"] * price
                    cash += proceeds
                    pnl = (fill_price - entry_price) * position["qty"] - cost
                    return_pct = (fill_price - entry_price) / entry_price
                    holding_seconds = (ts - position["entry_ts"]).total_seconds()
                    trades.append(
                        TradeSummary(
                            symbol=symbol,
                            entry_ts=position["entry_ts"],
                            exit_ts=ts,
                            entry_price=entry_price,
                            exit_price=fill_price,
                            pnl=pnl,
                            return_pct=return_pct,
                            holding_seconds=holding_seconds,
                        )
                    )
                    del positions[symbol]
                    cooldowns[symbol] = request.cooldown_bars  # Start cooldown
                    continue

            # Process trade signals
            for action in actions_by_symbol.get(symbol, {}).get(ts, []):
                # Check cooldown
                if symbol in cooldowns:
                    continue  # Skip trades during cooldown

                if action.decision == "BUY" and symbol not in positions:
                    # Check total exposure limit
                    current_exposure = sum(
                        pos["qty"] * last_prices.get(sym, pos["entry_price"])
                        for sym, pos in positions.items()
                    )
                    current_equity = cash + current_exposure
                    if current_exposure / current_equity >= request.max_total_exposure_pct:
                        continue  # Skip if max exposure reached

                    allocation = cash * request.max_pos_pct
                    cost = request.fee_per_trade
                    fill_price = price * (1 + request.slippage_bps / 10000)
                    total_costs += cost + (request.slippage_bps / 10000) * allocation
                    qty = max(0.0, (allocation - cost) / fill_price)
                    if qty <= 0:
                        continue
                    total_cost = qty * fill_price + cost
                    if total_cost > cash:
                        qty = max(0.0, (cash - cost) / fill_price)
                        if qty <= 0:
                            continue
                        total_cost = qty * fill_price + cost
                    cash -= total_cost
                    positions[symbol] = {
                        "qty": qty,
                        "entry_price": fill_price,
                        "entry_ts": ts,
                    }
                elif action.decision == "SELL" and symbol in positions:
                    position = positions.pop(symbol)
                    fill_price = price * (1 - request.slippage_bps / 10000)
                    cost = request.fee_per_trade
                    total_costs += cost + (request.slippage_bps / 10000) * position["qty"] * price
                    proceeds = position["qty"] * fill_price - cost
                    cash += proceeds
                    pnl = (fill_price - position["entry_price"]) * position["qty"] - cost
                    return_pct = (fill_price - position["entry_price"]) / position["entry_price"]
                    holding_seconds = (ts - position["entry_ts"]).total_seconds()
                    trades.append(
                        TradeSummary(
                            symbol=symbol,
                            entry_ts=position["entry_ts"],
                            exit_ts=ts,
                            entry_price=position["entry_price"],
                            exit_price=fill_price,
                            pnl=pnl,
                            return_pct=return_pct,
                            holding_seconds=holding_seconds,
                        )
                    )

            equity = cash
            exposure_value = 0.0
            for sym, pos in positions.items():
                last_price = last_prices.get(sym)
                if last_price is None:
                    continue
                equity += pos["qty"] * last_price
                exposure_value += abs(pos["qty"] * last_price)
            equity_curve.append({"ts": ts.isoformat(), "equity": float(equity)})
            exposure_series.append(exposure_value / equity if equity > 0 else 0.0)

        for symbol, position in list(positions.items()):
            last_price = last_prices.get(symbol)
            if last_price is None:
                continue
            fill_price = last_price * (1 - request.slippage_bps / 10000)
            proceeds = position["qty"] * fill_price - request.fee_per_trade
            cash += proceeds
            pnl = (fill_price - position["entry_price"]) * position["qty"] - request.fee_per_trade
            return_pct = (fill_price - position["entry_price"]) / position["entry_price"]
            holding_seconds = (timeline[-1][0] - position["entry_ts"]).total_seconds() if timeline else 0.0
            trades.append(
                TradeSummary(
                    symbol=symbol,
                    entry_ts=position["entry_ts"],
                    exit_ts=timeline[-1][0] if timeline else position["entry_ts"],
                    entry_price=position["entry_price"],
                    exit_price=fill_price,
                    pnl=pnl,
                    return_pct=return_pct,
                    holding_seconds=holding_seconds,
                )
            )
            positions.pop(symbol, None)

        final_equity = cash

        metrics = _compute_metrics(
            equity_curve=equity_curve,
            trades=trades,
            initial_cash=request.initial_cash,
            final_equity=final_equity,
            exposure_series=exposure_series,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        logger.info("backtest_metrics_computed", num_trades=metrics.num_trades)

        run_id = uuid4()
        await self._persist_run(run_id, request, metrics)
        logger.info("backtest_persisted", run_id=str(run_id))

        top_trades = _top_trades(trades)
        sample_curve = _sample_equity_curve(equity_curve, max_points=200)

        return BacktestResponse(
            run_id=run_id,
            metrics=metrics,
            top_trades=top_trades,
            equity_curve=sample_curve,
            message=None,
        )

    async def _persist_run(self, run_id: UUID, request: BacktestRequest, metrics: BacktestMetrics) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO backtest_runs (run_id, params, metrics)
                VALUES ($1, $2, $3)
                """,
                run_id,
                json.dumps(request.model_dump(), default=str),
                json.dumps(metrics.model_dump(), default=str),
            )


def _bisect_next(values: list[datetime], target: datetime) -> int | None:
    lo = 0
    hi = len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(values) else None


def _compute_metrics(
    equity_curve: list[dict[str, Any]],
    trades: list[TradeSummary],
    initial_cash: float,
    final_equity: float,
    exposure_series: list[float],
    start_ts: datetime,
    end_ts: datetime,
) -> BacktestMetrics:
    total_return = (final_equity - initial_cash) / initial_cash if initial_cash else 0.0
    days = max(1.0, (end_ts - start_ts).total_seconds() / 86400)
    cagr = (final_equity / initial_cash) ** (365.0 / days) - 1 if initial_cash > 0 else 0.0

    num_trades = len(trades)
    wins = len([t for t in trades if t.pnl > 0])
    win_rate = wins / num_trades if num_trades else 0.0
    avg_trade_return = sum(t.return_pct for t in trades) / num_trades if num_trades else 0.0
    avg_holding = sum(t.holding_seconds for t in trades) / num_trades if num_trades else 0.0

    equity_values = [point["equity"] for point in equity_curve]
    max_drawdown = _max_drawdown(equity_values)
    sharpe_like = _sharpe_like(equity_values)
    exposure_pct = sum(exposure_series) / len(exposure_series) if exposure_series else 0.0

    return BacktestMetrics(
        total_return=float(total_return),
        cagr=float(cagr),
        win_rate=float(win_rate),
        max_drawdown=float(max_drawdown),
        num_trades=num_trades,
        avg_trade_return=float(avg_trade_return),
        sharpe_like=float(sharpe_like),
        exposure_pct=float(exposure_pct),
        avg_holding_seconds=float(avg_holding),
        final_equity=float(final_equity),
    )


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for value in values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak else 0.0
        max_dd = max(max_dd, drawdown)
    return max_dd


def _sharpe_like(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    returns = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        if prev <= 0:
            continue
        returns.append((values[i] - prev) / prev)
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = variance ** 0.5
    if std == 0:
        return 0.0
    return mean / std * (len(returns) ** 0.5)


def _top_trades(trades: list[TradeSummary]) -> list[TradeSummary]:
    if not trades:
        return []
    sorted_trades = sorted(trades, key=lambda t: t.pnl)
    worst = sorted_trades[:5]
    best = sorted_trades[-5:][::-1]
    return best + worst


def _sample_equity_curve(equity_curve: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(equity_curve) <= max_points:
        return equity_curve
    step = max(1, len(equity_curve) // max_points)
    return equity_curve[::step]


def _empty_metrics(initial_cash: float) -> BacktestMetrics:
    return BacktestMetrics(
        total_return=0.0,
        cagr=0.0,
        win_rate=0.0,
        max_drawdown=0.0,
        num_trades=0,
        avg_trade_return=0.0,
        sharpe_like=0.0,
        exposure_pct=0.0,
        avg_holding_seconds=0.0,
        final_equity=float(initial_cash),
    )


class BacktesterAgent(BaseAgent):
    def __init__(self) -> None:
        settings = load_settings()
        super().__init__("backtester", settings.app.http.backtester_port)
        self._service = BacktesterService()
        self.add_protected_route("/backtest", self.backtest, methods=["POST"])

    async def start(self) -> None:
        self.app.add_event_handler("shutdown", self._shutdown)

    async def _shutdown(self) -> None:
        await self._service.close()

    async def backtest(self, request: BacktestRequest) -> BacktestResponse:
        set_correlation_id(str(uuid4()))
        try:
            response = await self._service.run_backtest(request)
            EVENTS_PROCESSED.labels(agent=self.name, event_type="backtest").inc()
            return response
        except Exception as exc:
            logger.error("backtest_failed", error=str(exc))
            raise HTTPException(status_code=400, detail=str(exc))

    async def health(self) -> dict[str, str]:
        return {"status": "ok", "agent": "BacktesterAgent"}


def main() -> None:
    BacktesterAgent().run()


if __name__ == "__main__":
    main()
