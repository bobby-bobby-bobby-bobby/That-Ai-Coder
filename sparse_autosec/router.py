from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .experts import ExpertPool
from .runtime import SparseRuntime, SparseTensor


@dataclass
class RoutingDecision:
    selected: List[str]
    confidence: float
    raw_scores: Dict[str, float]


class SparseRouter:
    """Lightweight router selecting 2-5 experts with anti-collapse load balancing."""

    def __init__(self, runtime: SparseRuntime, pool: ExpertPool, min_k: int = 2, max_k: int = 5):
        self.runtime = runtime
        self.pool = pool
        self.min_k = min_k
        self.max_k = max_k

    def route(self, tensor: SparseTensor) -> RoutingDecision:
        score_rows: List[Tuple[str, float]] = []
        for name in self.pool.names():
            ex = self.pool.get(name)
            load_penalty = 0.2 * (ex.use_count / max(ex.success_count, 1))
            raw = ex.score(tensor) + ex.confidence() - load_penalty
            score_rows.append((name, raw))

        k = min(max(self.min_k, len(score_rows) // 3), self.max_k)
        selected = self.runtime.topk(score_rows, k)
        confidence = sum(max(0.0, s) for _, s in selected) / max(1, k)
        return RoutingDecision(
            selected=[name for name, _ in selected],
            confidence=confidence,
            raw_scores={n: s for n, s in score_rows},
        )
