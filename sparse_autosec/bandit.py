from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from typing import Dict, Iterable, List, Tuple


@dataclass
class ArmStats:
    pulls: int = 1
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / max(1, self.pulls)


@dataclass
class RoutingBandit:
    arms: Dict[str, ArmStats] = field(default_factory=dict)
    total_pulls: int = 1

    def score(self, expert_names: Iterable[str], prior_scores: Dict[str, float], c: float = 1.2) -> List[Tuple[str, float]]:
        rows: List[Tuple[str, float]] = []
        for name in expert_names:
            stats = self.arms.setdefault(name, ArmStats())
            bonus = c * sqrt(log(max(2, self.total_pulls)) / max(1, stats.pulls))
            rows.append((name, prior_scores.get(name, 0.0) + stats.mean_reward + bonus))
        return sorted(rows, key=lambda x: x[1], reverse=True)

    def update(self, selected: List[str], reward: float) -> None:
        self.total_pulls += 1
        for name in selected:
            stats = self.arms.setdefault(name, ArmStats())
            stats.pulls += 1
            stats.reward_sum += reward
