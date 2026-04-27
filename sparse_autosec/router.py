from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Dict, List, Tuple

from .experts import ExpertPool
from .runtime import SparseRuntime, SparseTensor


@dataclass
class RoutingDecision:
    selected: List[str]
    confidence: float
    raw_scores: Dict[str, float]
    rationale: Dict[str, str]


class SparseRouter:
    """Sparse MoE router with confidence, load balancing, and anti-collapse exploration."""

    def __init__(self, runtime: SparseRuntime, pool: ExpertPool, min_k: int = 2, max_k: int = 5):
        self.runtime = runtime
        self.pool = pool
        self.min_k = min_k
        self.max_k = max_k
        self.total_routes = 1

    def route(self, tensor: SparseTensor, complexity: float = 0.5) -> RoutingDecision:
        score_rows: List[Tuple[str, float]] = []
        rationale: Dict[str, str] = {}

        names = self.pool.names(active_only=True)
        for name in names:
            ex = self.pool.get(name)
            exploit = ex.score(tensor)
            confidence = ex.confidence()
            load_penalty = 0.25 * (ex.use_count / max(ex.success_count + 1, 1))
            exploration = sqrt(2.0 * log(max(2, self.total_routes)) / max(1, ex.use_count))
            score = exploit + 0.8 * confidence + 0.15 * exploration - load_penalty
            score_rows.append((name, score))
            rationale[name] = (
                f"exploit={exploit:.3f}, conf={confidence:.3f}, "
                f"explore={exploration:.3f}, load_penalty={load_penalty:.3f}"
            )

        k = self._choose_k(len(score_rows), complexity)
        selected = self.runtime.topk(score_rows, k)
        conf = sum(max(0.0, s) for _, s in selected) / max(1, k)
        self.total_routes += 1
        return RoutingDecision(
            selected=[name for name, _ in selected],
            confidence=conf,
            raw_scores={name: score for name, score in score_rows},
            rationale=rationale,
        )

    def _choose_k(self, n_experts: int, complexity: float) -> int:
        if n_experts <= self.min_k:
            return n_experts
        scaled = self.min_k + int(round((self.max_k - self.min_k) * min(1.0, complexity / 3.0)))
        return min(max(self.min_k, scaled), min(self.max_k, n_experts))
