from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .experts import ExpertPool, ExpertState
from .memory import ExternalMemory


@dataclass
class LifecycleDecision:
    spawned: List[str]
    quiesced: List[str]
    retired: List[str]


class ExpertLifecycleManager:
    """Innovation: dynamic but controlled expert lifecycle.

    Spawn only on statistically repeated unresolved signatures.
    Quiesce low-health experts before retiring them.
    """

    def __init__(self, pool: ExpertPool, memory: ExternalMemory, failure_threshold: int = 3):
        self.pool = pool
        self.memory = memory
        self.failure_threshold = failure_threshold
        self.quiescent_cycles: Dict[str, int] = {}

    def step(self) -> LifecycleDecision:
        spawned: List[str] = []
        quiesced: List[str] = []
        retired: List[str] = []

        for signature, count in self.memory.counters.items():
            if count >= self.failure_threshold and not any(signature in n for n in self.pool.names(active_only=False)):
                ex = self.pool.maybe_create_expert(signature, [f"auto::{signature}"])
                spawned.append(ex.name)

        for name in self.pool.names(active_only=False):
            ex = self.pool.get(name)
            health = ex.health()
            if ex.state == ExpertState.ACTIVE and health < 0.15 and ex.use_count > 8:
                ex.state = ExpertState.QUIESCENT
                self.quiescent_cycles[name] = self.quiescent_cycles.get(name, 0) + 1
                quiesced.append(name)
            elif ex.state == ExpertState.QUIESCENT:
                self.quiescent_cycles[name] = self.quiescent_cycles.get(name, 0) + 1
                if self.quiescent_cycles[name] >= 4 and ex.health() < 0.15:
                    ex.state = ExpertState.RETIRED
                    retired.append(name)

        return LifecycleDecision(spawned=spawned, quiesced=quiesced, retired=retired)
