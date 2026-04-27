from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass
class BudgetState:
    max_latency_ms: int
    max_fuzz_cases: int
    max_train_steps: int
    used_fuzz_cases: int = 0
    used_train_steps: int = 0
    start_time: float = 0.0

    def start(self) -> None:
        self.start_time = perf_counter()

    def elapsed_ms(self) -> float:
        return (perf_counter() - self.start_time) * 1000.0

    def can_fuzz(self) -> bool:
        return self.used_fuzz_cases < self.max_fuzz_cases and self.elapsed_ms() < self.max_latency_ms

    def can_train(self) -> bool:
        return self.used_train_steps < self.max_train_steps and self.elapsed_ms() < self.max_latency_ms


class ResourcePolicy:
    """Runtime budget policy to preserve <=8GB-class constrained deployments."""

    def __init__(self, max_latency_ms: int = 2500, max_fuzz_cases: int = 64, max_train_steps: int = 24):
        self.max_latency_ms = max_latency_ms
        self.max_fuzz_cases = max_fuzz_cases
        self.max_train_steps = max_train_steps

    def new_state(self) -> BudgetState:
        state = BudgetState(
            max_latency_ms=self.max_latency_ms,
            max_fuzz_cases=self.max_fuzz_cases,
            max_train_steps=self.max_train_steps,
        )
        state.start()
        return state
