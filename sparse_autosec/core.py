from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .runtime import SparseRuntime, SparseTensor


@dataclass
class Adapter:
    delta: Dict[int, float] = field(default_factory=dict)

    def apply(self, tensor: SparseTensor) -> float:
        return sum(tensor.values.get(i, 0.0) * weight for i, weight in self.delta.items())

    def update(self, gradient: Dict[int, float], lr: float = 0.001, clip: float = 0.1) -> None:
        for idx, g in gradient.items():
            g = max(-clip, min(clip, g))
            self.delta[idx] = self.delta.get(idx, 0.0) + lr * g


class CoreModel:
    """Stable core coordinator with minimal parameter-efficient adapter."""

    def __init__(self, runtime: SparseRuntime):
        self.runtime = runtime
        self.base_bias = 0.25
        self.adapter = Adapter()

    def encode_task(self, text: str) -> SparseTensor:
        return self.runtime.tokenize_sparse(text)

    def score_task_complexity(self, text: str) -> float:
        vec = self.encode_task(text)
        lexical_complexity = min(2.0, 0.005 * sum(len(t) for t in text.split()))
        return self.base_bias + lexical_complexity + self.adapter.apply(vec)

    def propose_plan_tags(self, findings: List[str]) -> List[str]:
        tags: List[str] = []
        for sig in findings:
            if "eval" in sig or "injection" in sig:
                tags.append("sanitize_input")
            if "crash" in sig or "memory" in sig:
                tags.append("add_bounds_checks")
            if "deserial" in sig:
                tags.append("safe_deserialize")
        if not tags:
            tags.append("general_hardening")
        return sorted(set(tags))
