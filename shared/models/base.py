from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class BaseEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: UUID = Field(default_factory=uuid4)
    correlation_id: UUID = Field(default_factory=uuid4)
    idempotency_key: str
    symbol: str
    ts: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    schema_version: int = 1
    source: str

    def to_avro_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
