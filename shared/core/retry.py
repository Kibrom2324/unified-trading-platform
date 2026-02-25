from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    retries: int = 3,
    base_delay: float = 0.2,
    max_delay: float = 5.0,
    jitter: float = 0.2,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return await operation()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            delay = min(max_delay, base_delay * (2**attempt))
            delay = delay + random.uniform(0, jitter)
            await asyncio.sleep(delay)
    if last_exc:
        raise last_exc
    raise RuntimeError("retry_async failed with no exception")
