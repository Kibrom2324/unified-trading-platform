import pytest

from src.core.circuit_breaker import CircuitBreaker


def test_circuit_breaker() -> None:
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

    def fail() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        breaker.call(fail)
    with pytest.raises(RuntimeError):
        breaker.call(fail)

    with pytest.raises(RuntimeError):
        breaker.call(fail)
