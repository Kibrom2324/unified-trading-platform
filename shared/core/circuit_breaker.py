from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_timeout: float = 5.0
    failure_count: int = 0
    state: str = "closed"
    last_failure_time: float = field(default_factory=lambda: 0.0)

    def allow(self) -> bool:
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def call(self, func):
        if not self.allow():
            raise RuntimeError("Circuit breaker open")
        try:
            result = func()
            self.record_success()
            return result
        except Exception:  # noqa: BLE001
            self.record_failure()
            raise
