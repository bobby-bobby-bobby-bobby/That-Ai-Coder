from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .execution import PatchEngine, StructuredMutator
from .symbolic import SymbolicPatchVerifier


@dataclass
class PatchCandidate:
    signature: str
    code: str
    symbolic_ok: bool
    validation_ok: bool
    score: float


class PatchRanker:
    """Ranks and validates candidate patches with low-cost heuristics + dynamic checks."""

    def __init__(self, patcher: PatchEngine, symbolic: SymbolicPatchVerifier, mutator: StructuredMutator):
        self.patcher = patcher
        self.symbolic = symbolic
        self.mutator = mutator

    def generate_candidates(self, original_code: str, signatures: List[str]) -> List[PatchCandidate]:
        candidates: List[PatchCandidate] = []
        for sig in signatures:
            code = self.patcher.propose_patch(original_code, sig)
            if code == original_code:
                continue
            symbolic_ok = self.symbolic.verify_python_source(code).safe
            score = 0.6 + (0.4 if symbolic_ok else 0.0)
            candidates.append(PatchCandidate(sig, code, symbolic_ok, False, score))
        return sorted(candidates, key=lambda x: x.score, reverse=True)

    def validate(self, target_file: Path, candidates: List[PatchCandidate], top_k: int = 3) -> PatchCandidate | None:
        for candidate in candidates[:top_k]:
            if not candidate.symbolic_ok:
                continue
            candidate.validation_ok = self.patcher.validate_patch(target_file, candidate.code, self.mutator)
            if candidate.validation_ok:
                candidate.score += 0.5
                return candidate
        return None
