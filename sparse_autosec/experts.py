from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List

from .runtime import SparseTensor


class ExpertState(str, Enum):
    ACTIVE = "active"
    QUIESCENT = "quiescent"
    RETIRED = "retired"


@dataclass
class ExpertModule:
    name: str
    specialty_keywords: List[str]
    weights: Dict[int, float] = field(default_factory=dict)
    state: ExpertState = ExpertState.ACTIVE
    success_count: int = 1
    fail_count: int = 0
    use_count: int = 1
    recent_scores: List[float] = field(default_factory=list)

    def score(self, tensor: SparseTensor) -> float:
        raw = sum(tensor.values.get(idx, 0.0) * w for idx, w in self.weights.items())
        self.recent_scores.append(raw)
        if len(self.recent_scores) > 32:
            self.recent_scores.pop(0)
        return raw

    def confidence(self) -> float:
        return self.success_count / max(self.use_count, 1)

    def health(self) -> float:
        penalty = self.fail_count / max(1, self.use_count)
        return max(0.01, self.confidence() - penalty)

    def mean_score(self) -> float:
        if not self.recent_scores:
            return 0.0
        return sum(self.recent_scores) / len(self.recent_scores)

    def train(self, tensor: SparseTensor, target: float, lr: float = 0.02) -> None:
        self.use_count += 1
        pred = self.score(tensor)
        err = target - pred
        for idx, value in tensor.values.items():
            self.weights[idx] = self.weights.get(idx, 0.0) + lr * err * value
        if target > 0.0:
            self.success_count += 1
        else:
            self.fail_count += 1

    def tag_failure(self) -> None:
        self.fail_count += 1
        self.use_count += 1


class ExpertPool:
    def __init__(self) -> None:
        self.experts: Dict[str, ExpertModule] = {}

    def add_expert(self, expert: ExpertModule) -> None:
        self.experts[expert.name] = expert

    def names(self, active_only: bool = True) -> List[str]:
        if not active_only:
            return list(self.experts)
        return [name for name, ex in self.experts.items() if ex.state == ExpertState.ACTIVE]

    def get(self, name: str) -> ExpertModule:
        return self.experts[name]

    def values(self) -> Iterable[ExpertModule]:
        return self.experts.values()

    def maybe_create_expert(self, signature: str, evidence: List[str]) -> ExpertModule:
        suffix = signature[:16].replace(" ", "_")
        name = f"expert_{len(self.experts) + 1}_{suffix}"
        expert = ExpertModule(name=name, specialty_keywords=evidence)
        self.add_expert(expert)
        return expert
