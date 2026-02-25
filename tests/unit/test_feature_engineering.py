from src.models.events_engineered import MarketEngineeredEvent
from src.models.events_market import MarketRawEvent


def test_feature_engineering() -> None:
    raw = MarketRawEvent(idempotency_key="x", symbol="AAPL", source="test", price=100, volume=10)
    engineered = MarketEngineeredEvent(
        idempotency_key=raw.idempotency_key,
        symbol=raw.symbol,
        source="test",
        correlation_id=raw.correlation_id,
        price=raw.price,
        volume=raw.volume,
        returns_1=0.01,
        returns_5=0.01,
    )
    assert engineered.returns_1 == 0.01
