from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Dict, List


@dataclass
class Event:
    name: str
    timestamp: float
    payload: Dict[str, float | str | int]


@dataclass
class Telemetry:
    events: List[Event] = field(default_factory=list)

    def emit(self, name: str, **payload: float | str | int) -> None:
        self.events.append(Event(name=name, timestamp=time(), payload=payload))

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for event in self.events:
            counts[event.name] = counts.get(event.name, 0) + 1
        return counts
