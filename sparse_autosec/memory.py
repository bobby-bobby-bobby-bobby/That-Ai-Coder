from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FailureRecord:
    signature: str
    task_type: str
    details: str
    exploitability: str
    confidence: float = 1.0


@dataclass
class FixRecord:
    signature: str
    patch_summary: str
    patch_code: str
    success: bool


@dataclass
class PatternRecord:
    signature: str
    tokens: Dict[str, int]


@dataclass
class ExternalMemory:
    failures: List[FailureRecord] = field(default_factory=list)
    fixes: List[FixRecord] = field(default_factory=list)
    patterns: Dict[str, PatternRecord] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)

    def add_failure(self, rec: FailureRecord) -> None:
        self.failures.append(rec)
        self.counters[rec.signature] = self.counters.get(rec.signature, 0) + 1
        tokens = self.patterns.get(rec.signature, PatternRecord(rec.signature, {})).tokens
        for tok in rec.details.lower().split():
            tokens[tok] = tokens.get(tok, 0) + 1
        self.patterns[rec.signature] = PatternRecord(signature=rec.signature, tokens=tokens)

    def add_fix(self, rec: FixRecord) -> None:
        self.fixes.append(rec)

    def repeated_failures(self, signature: str, threshold: int = 3) -> bool:
        return self.counters.get(signature, 0) >= threshold

    def find_similar_fix(self, signature: str) -> FixRecord | None:
        for fix in reversed(self.fixes):
            if fix.signature == signature and fix.success:
                return fix
        return None

    def related_signatures(self, signature: str, topk: int = 3) -> List[str]:
        source = self.patterns.get(signature)
        if not source:
            return []
        src_tokens = set(source.tokens)
        scores = []
        for sig, record in self.patterns.items():
            if sig == signature:
                continue
            overlap = len(src_tokens.intersection(set(record.tokens)))
            if overlap > 0:
                scores.append((sig, overlap))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [sig for sig, _ in scores[:topk]]
