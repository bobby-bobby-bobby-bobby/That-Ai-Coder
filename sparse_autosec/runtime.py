from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Tuple


@dataclass
class SparseTensor:
    size: int
    values: Dict[int, float] = field(default_factory=dict)

    def nnz(self) -> int:
        return len(self.values)

    def dot(self, other: "SparseTensor") -> float:
        if self.nnz() > other.nnz():
            return other.dot(self)
        return sum(v * other.values.get(i, 0.0) for i, v in self.values.items())

    def l1_norm(self) -> float:
        return sum(abs(v) for v in self.values.values())

    def scale(self, alpha: float) -> "SparseTensor":
        return SparseTensor(size=self.size, values={i: v * alpha for i, v in self.values.items() if v != 0.0})

    def add_(self, other: "SparseTensor", alpha: float = 1.0) -> None:
        for idx, value in other.values.items():
            self.values[idx] = self.values.get(idx, 0.0) + value * alpha
            if abs(self.values[idx]) < 1e-12:
                del self.values[idx]

    def clipped(self, threshold: float = 1e-6) -> "SparseTensor":
        return SparseTensor(size=self.size, values={i: v for i, v in self.values.items() if abs(v) >= threshold})


class TensorPool:
    """Reusable sparse tensor instances to reduce repeated allocations."""

    def __init__(self, max_items: int = 128):
        self.max_items = max_items
        self.pool: List[SparseTensor] = []

    def acquire(self, size: int) -> SparseTensor:
        if self.pool:
            tensor = self.pool.pop()
            tensor.size = size
            tensor.values.clear()
            return tensor
        return SparseTensor(size=size)

    def release(self, tensor: SparseTensor) -> None:
        if len(self.pool) < self.max_items:
            tensor.values.clear()
            self.pool.append(tensor)


class RuntimeCache:
    def __init__(self, max_items: int = 1024):
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
        self._cache[key] = SparseTensor(size=value.size, values=dict(value.values))
        self._order.append(key)


class SparseRuntime:
    """Event-driven, low-memory sparse runtime for routing/training."""

    def __init__(self, vocab_size: int = 2048, cache_items: int = 1024):
        self.vocab_size = vocab_size
        self.pool = TensorPool()
        self.cache = RuntimeCache(max_items=cache_items)

    def tokenize_sparse(self, text: str) -> SparseTensor:
        key = f"tok::{text}"
        cached = self.cache.get(key)
        if cached is not None:
            return SparseTensor(size=cached.size, values=dict(cached.values))

        t = self.pool.acquire(self.vocab_size)
        for tok in self._iter_tokens(text):
            idx = hash(tok) % self.vocab_size
            t.values[idx] = t.values.get(idx, 0.0) + 1.0
        result = t.clipped()
        self.cache.set(key, result)
        self.pool.release(t)
        return result

    def merge_texts(self, texts: List[str], normalize: bool = True) -> SparseTensor:
        merged = SparseTensor(size=self.vocab_size)
        for text in texts:
            merged.add_(self.tokenize_sparse(text))
        if normalize and merged.nnz() > 0:
            inv = 1.0 / merged.l1_norm()
            merged = merged.scale(inv)
        return merged

    def _iter_tokens(self, text: str) -> Iterator[str]:
        token = []
        for char in text.lower():
            if char.isalnum() or char in {"_", "-"}:
                token.append(char)
            else:
                if token:
                    yield "".join(token)
                    token = []
        if token:
            yield "".join(token)

    @staticmethod
    def topk(scores: Iterable[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
        return sorted(scores, key=lambda item: item[1], reverse=True)[:k]
