from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FailureRecord:
    signature: str
    task_type: str
    details: str
    exploitability: str


@dataclass
class FixRecord:
    signature: str
    patch_summary: str
    success: bool


@dataclass
class ExternalMemory:
    """External memory store to prioritize retrieval over retraining."""

    failures: List[FailureRecord] = field(default_factory=list)
    fixes: List[FixRecord] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=dict)

    def add_failure(self, rec: FailureRecord) -> None:
        self.failures.append(rec)
        self.counters[rec.signature] = self.counters.get(rec.signature, 0) + 1

    def add_fix(self, rec: FixRecord) -> None:
        self.fixes.append(rec)

    def repeated_failures(self, signature: str, threshold: int = 3) -> bool:
        return self.counters.get(signature, 0) >= threshold

    def find_similar_fix(self, signature: str) -> FixRecord | None:
        for fix in reversed(self.fixes):
            if fix.signature == signature and fix.success:
                return fix
        return None
