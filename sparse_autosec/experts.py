from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .runtime import SparseTensor


@dataclass
class ExpertModule:
    name: str
    specialty_keywords: List[str]
    weights: Dict[int, float] = field(default_factory=dict)
    success_count: int = 1
    use_count: int = 1

    def score(self, tensor: SparseTensor) -> float:
        return sum(tensor.values.get(idx, 0.0) * w for idx, w in self.weights.items())

    def confidence(self) -> float:
        return self.success_count / max(self.use_count, 1)

    def train(self, tensor: SparseTensor, target: float, lr: float = 0.02) -> None:
        self.use_count += 1
        pred = self.score(tensor)
        err = target - pred
        for idx, value in tensor.values.items():
            self.weights[idx] = self.weights.get(idx, 0.0) + lr * err * value
        if target > 0:
            self.success_count += 1


class ExpertPool:
    def __init__(self) -> None:
        self.experts: Dict[str, ExpertModule] = {}

    def add_expert(self, expert: ExpertModule) -> None:
        self.experts[expert.name] = expert

    def names(self) -> List[str]:
        return list(self.experts)

    def get(self, name: str) -> ExpertModule:
        return self.experts[name]

    def maybe_create_expert(self, signature: str, evidence: List[str]) -> ExpertModule:
        name = f"expert_{len(self.experts)+1}_{signature[:12].replace(' ', '_')}"
        expert = ExpertModule(name=name, specialty_keywords=evidence)
        self.add_expert(expert)
        return expert
