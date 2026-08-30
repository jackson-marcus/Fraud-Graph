"""Publish GraphEdge records onto an in-memory Redis Stream."""

from __future__ import annotations

from typing import Any

from fraudgraph.streams.schemas import STREAM_NAME, StreamEvent


class InMemoryStream:
    """Minimal XADD/XREAD stand-in used by tests and local workers."""

    def __init__(self) -> None:
        self.entries: list[StreamEvent] = []

    def xadd(self, event: StreamEvent) -> str:
        self.entries.append(event)
        return event.event_id

    def xread(self, last_id: str | None = None) -> list[StreamEvent]:
        if last_id is None:
            return list(self.entries)
        seen = False
        pending: list[StreamEvent] = []
        for event in self.entries:
            if seen:
                pending.append(event)
            elif event.event_id == last_id:
                seen = True
        return pending


class StreamProducer:
    def __init__(self, stream: InMemoryStream, stream_name: str = STREAM_NAME) -> None:
        self.stream = stream
        self.stream_name = stream_name

    def publish(self, payload: dict[str, Any], **headers: str) -> StreamEvent:
        event = StreamEvent.create(payload, **headers)
        self.stream.xadd(event)
        return event
