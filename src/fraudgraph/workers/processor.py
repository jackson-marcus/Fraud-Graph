"""ring_detector — feature processing sink for GraphEdge events."""

from __future__ import annotations

from fraudgraph.streams.schemas import REQUIRED_FIELDS, StreamEvent


class FeatureProcessor:
    """Turn stream payloads into a running feature snapshot."""

    name = "ring_detector"

    def __init__(self) -> None:
        self.seen = 0
        self.snapshot: dict[str, float] = dict.fromkeys(REQUIRED_FIELDS, 0.0)

    def handle(self, event: StreamEvent) -> dict[str, float]:
        self.seen += 1
        for key, value in event.payload.items():
            if isinstance(value, int | float):
                self.snapshot[key] = self.snapshot.get(key, 0.0) + float(value)
            else:
                self.snapshot[key] = self.snapshot.get(key, 0.0) + 1.0
        return dict(self.snapshot)
