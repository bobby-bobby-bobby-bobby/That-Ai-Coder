from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .runtime import SparseRuntime, SparseTensor


@dataclass
class Adapter:
    """LoRA-style lightweight adaptation as sparse delta weights."""

    delta: Dict[int, float] = field(default_factory=dict)

    def apply(self, tensor: SparseTensor) -> float:
        return sum(tensor.values.get(i, 0.0) * w for i, w in self.delta.items())

    def update(self, gradient: Dict[int, float], lr: float = 0.05) -> None:
        for idx, g in gradient.items():
            self.delta[idx] = self.delta.get(idx, 0.0) + lr * g


class CoreModel:
    """Small stable core for coordination/reasoning; base weights are fixed."""

    def __init__(self, runtime: SparseRuntime):
        self.runtime = runtime
        self.base_bias = 0.2
        self.adapter = Adapter()

    def encode_task(self, text: str) -> SparseTensor:
        return self.runtime.tokenize_sparse(text)

    def score_task_complexity(self, text: str) -> float:
        vec = self.encode_task(text)
        return self.base_bias + self.adapter.apply(vec) + 0.01 * len(vec.values)
