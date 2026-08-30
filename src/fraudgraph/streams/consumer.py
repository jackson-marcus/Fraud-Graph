"""Consumer-group reader for the graph.edges stream."""

from __future__ import annotations

from collections.abc import Callable

from fraudgraph.streams.producer import InMemoryStream
from fraudgraph.streams.schemas import StreamEvent


class StreamConsumer:
    def __init__(self, stream: InMemoryStream, group: str = "workers") -> None:
        self.stream = stream
        self.group = group
        self._last_id: str | None = None
        self.processed: list[str] = []
        self.dead_letters: list[StreamEvent] = []

    def poll(self) -> list[StreamEvent]:
        batch = self.stream.xread(self._last_id)
        if batch:
            self._last_id = batch[-1].event_id
        return batch

    def run_once(self, handler: Callable[[StreamEvent], None]) -> int:
        batch = self.poll()
        for event in batch:
            try:
                handler(event)
                self.processed.append(event.event_id)
            except Exception:
                self.dead_letters.append(event)
        return len(batch)
