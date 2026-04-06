from __future__ import annotations

from .types import TraceEvent


class TraceRecorder:
    def __init__(self):
        self.events: list[TraceEvent] = []

    def record(self, event_type: str, cycle_index: int, **payload) -> TraceEvent:
        event = TraceEvent(event_type=event_type, cycle_index=cycle_index, payload=payload)
        self.events.append(event)
        return event

    def latest(self, event_type: str | None = None) -> TraceEvent | None:
        if event_type is None:
            return self.events[-1] if self.events else None
        for event in reversed(self.events):
            if event.event_type == event_type:
                return event
        return None
