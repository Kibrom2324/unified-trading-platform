from src.agents.trader_decision_agent import LruDedupSet, _decision_reason
from src.core.trade_decision import DecisionThresholds, decide_trade_v2


def test_long_high_confidence_buy() -> None:
    thresholds = DecisionThresholds(min_confidence=0.65)
    decision = decide_trade_v2({"direction": "long", "confidence": 0.9}, thresholds)
    assert decision == "BUY"
    assert _decision_reason(decision) == "long_signal_confident"


def test_short_high_confidence_sell() -> None:
    thresholds = DecisionThresholds(min_confidence=0.65)
    decision = decide_trade_v2({"direction": "short", "confidence": 0.9}, thresholds)
    assert decision == "SELL"
    assert _decision_reason(decision) == "short_signal_confident"


def test_low_confidence_hold() -> None:
    thresholds = DecisionThresholds(min_confidence=0.65)
    decision = decide_trade_v2({"direction": "long", "confidence": 0.4}, thresholds)
    assert decision == "HOLD"
    assert _decision_reason(decision) == "insufficient_confidence_or_neutral"


def test_idempotency_dedup_drops_duplicate() -> None:
    dedup = LruDedupSet(max_size=3)
    assert dedup.add("k1") is True
    assert dedup.add("k1") is False
    assert len(dedup) == 1
