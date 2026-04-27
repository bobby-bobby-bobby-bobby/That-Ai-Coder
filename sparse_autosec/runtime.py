from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


@dataclass
class SparseTensor:
    """A minimal sparse vector/tensor representation.

    Keys are integer indices, values are floats.
    """

    size: int
    values: Dict[int, float] = field(default_factory=dict)

    def dot(self, other: "SparseTensor") -> float:
        if len(self.values) > len(other.values):
            return other.dot(self)
        return sum(value * other.values.get(idx, 0.0) for idx, value in self.values.items())

    def add_scaled(self, other: "SparseTensor", alpha: float) -> "SparseTensor":
        out = SparseTensor(size=self.size, values=dict(self.values))
        for idx, value in other.values.items():
            out.values[idx] = out.values.get(idx, 0.0) + alpha * value
            if abs(out.values[idx]) < 1e-10:
                del out.values[idx]
        return out


class RuntimeCache:
    """Simple keyed cache for incremental/event-driven inference."""

    def __init__(self, max_items: int = 256):
        self.max_items = max_items
        self._cache: Dict[str, SparseTensor] = {}
        self._order: List[str] = []

    def get(self, key: str) -> SparseTensor | None:
        return self._cache.get(key)

    def set(self, key: str, value: SparseTensor) -> None:
        if key in self._cache:
            return
        if len(self._cache) >= self.max_items:
            oldest = self._order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[key] = value
        self._order.append(key)


class SparseRuntime:
    """Custom lightweight runtime focused on sparse/event-driven computation."""

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.cache = RuntimeCache()

    def tokenize_sparse(self, text: str) -> SparseTensor:
        key = f"tok::{text}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        indices: Dict[int, float] = {}
        for tok in text.lower().split():
            idx = hash(tok) % self.vocab_size
            indices[idx] = indices.get(idx, 0.0) + 1.0
        tensor = SparseTensor(size=self.vocab_size, values=indices)
        self.cache.set(key, tensor)
        return tensor

    def topk(self, scores: Iterable[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
        return sorted(scores, key=lambda it: it[1], reverse=True)[:k]
