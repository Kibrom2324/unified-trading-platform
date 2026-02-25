import pytest

from src.core.retry import retry_async


@pytest.mark.asyncio
async def test_retry_async() -> None:
    attempts = {"count": 0}

    async def op() -> int:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("fail")
        return 42

    result = await retry_async(op, retries=3)
    assert result == 42
