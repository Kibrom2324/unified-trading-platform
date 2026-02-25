from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.backtester_agent import BacktestRequest, BacktesterService
from src.core.trade_decision import decide_trade


def main() -> int:
    request = BacktestRequest(
        symbols=["AAPL"],
        start="2025-07-01",
        end="2025-07-08",
        initial_cash=2000.0,
        max_pos_pct=0.1,
        slippage_bps=1.0,
        fee_per_trade=0.0,
    )

    async def _run() -> int:
        service = BacktesterService()
        try:
            response = await service.run_backtest(request)
        finally:
            await service.close()

        metrics = response.metrics
        assert metrics.num_trades >= 0
        assert response.equity_curve
        assert isinstance(metrics.total_return, float)
        assert decide_trade({"direction": "long", "confidence": 0.9}, 0.65) == "BUY"
        assert decide_trade({"direction": "short", "confidence": 0.9}, 0.65) == "SELL"
        assert decide_trade({"direction": "long", "confidence": 0.1}, 0.65) == "HOLD"
        print("OK")
        return 0

    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
