from __future__ import annotations

from src.models.base import BaseEvent


class ExecutionEvent(BaseEvent):
    """
    Execution event produced by ExecutionAgent.
    
    Attributes:
        status: "filled" if approved and order succeeded, "blocked" if rejected/dry-run/error
        filled_qty: Number of shares filled (0 if blocked)
        avg_price: Average fill price (from risk_score in simulation, from Alpaca if real order)
        paper_trade: Always True - this system only supports paper trading
        dry_run: True if blocked due to dry_run mode (safety override)
        side: Order side - "buy" or "sell" (default: "buy")
        alpaca_order_id: Alpaca order ID if order was submitted (None if blocked/simulated)
        alpaca_status: Alpaca order status (new, filled, rejected, etc.)
        error_message: Error message if order failed (None if successful)
    """
    status: str
    filled_qty: int
    avg_price: float
    paper_trade: bool
    dry_run: bool = False
    side: str = "buy"
    alpaca_order_id: str | None = None
    alpaca_status: str | None = None
    error_message: str | None = None
