from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .core import CoreModel
from .execution import Analyzer, PatchEngine, SandboxedExecutor, StructuredMutator
from .experts import ExpertModule, ExpertPool
from .memory import ExternalMemory, FailureRecord, FixRecord
from .router import SparseRouter
from .runtime import SparseRuntime
from .training import ControlledLearningSystem


@dataclass
class TaskReport:
    findings: List[str]
    patched: bool
    patch_signature: str | None
    routing: Dict[str, float]


class SparseExpertAutoSec:
    """End-to-end sparse expert system for vulnerability discovery and repair."""

    def __init__(self) -> None:
        self.runtime = SparseRuntime(vocab_size=1024)
        self.core = CoreModel(self.runtime)
        self.memory = ExternalMemory()
        self.experts = ExpertPool()
        self._seed_experts()
        self.router = SparseRouter(self.runtime, self.experts, min_k=2, max_k=5)
        self.learning = ControlledLearningSystem(self.runtime, self.core, self.experts, self.memory)
        self.analyzer = Analyzer()
        self.patcher = PatchEngine()
        self.executor = SandboxedExecutor()
        self.mutator = StructuredMutator()

    def _seed_experts(self) -> None:
        # 10 starter experts
        specs = [
            "input_validation", "memory_safety", "injection_detection", "auth_logic",
            "crypto_misuse", "deserialization", "race_condition", "null_handling",
            "path_traversal", "dependency_risk",
        ]
        for idx, spec in enumerate(specs, start=1):
            e = ExpertModule(name=f"expert_{idx}_{spec}", specialty_keywords=[spec])
            e.weights[hash(spec) % 1024] = 0.7
            self.experts.add_expert(e)

    # Innovation 1: event-triggered expert lifecycle management
    def _maybe_spawn_expert(self, signature: str, evidence: List[str]) -> None:
        if self.memory.repeated_failures(signature, threshold=3) and signature not in " ".join(self.experts.names()):
            self.experts.maybe_create_expert(signature, evidence)

    # Innovation 2: success-history adaptive routing nudges
    def _routing_feedback(self, selected: List[str], success: bool) -> None:
        target = 1.0 if success else -0.2
        text = "routing feedback " + " ".join(selected)
        self.learning.record_replay(text=text, expert=selected[0], success_target=max(0.0, target))

    def process_target(self, target_file: Path) -> TaskReport:
        code = target_file.read_text()
        static_findings = self.analyzer.detect_static(code)

        task_text = f"analyze {target_file.name} vulnerabilities"
        task_tensor = self.core.encode_task(task_text)
        decision = self.router.route(task_tensor)

        dynamic_findings = []
        for payload in self.mutator.generate(20):
            res = self.executor.run(target_file, payload)
            finding = self.analyzer.detect_dynamic(res)
            if finding:
                dynamic_findings.append(finding)

        all_findings = static_findings + dynamic_findings
        patch_signature = None
        patched = False

        for finding in all_findings:
            self.memory.add_failure(FailureRecord(
                signature=finding.signature,
                task_type="vuln_scan",
                details=finding.details,
                exploitability=finding.exploitability,
            ))
            self._maybe_spawn_expert(finding.signature, [finding.details])

            maybe_known = self.memory.find_similar_fix(finding.signature)
            if maybe_known:
                continue

            patch = self.patcher.propose_patch(code, finding.signature)
            if patch != code and self.patcher.validate_patch(target_file, patch, self.mutator):
                target_file.write_text(patch)
                patched = True
                patch_signature = finding.signature
                self.memory.add_fix(FixRecord(signature=finding.signature, patch_summary="auto-generated", success=True))
                break

        if patch_signature and self.learning.enter_training(patch_signature):
            self.learning.train_step(task_text, decision.selected, success_target=1.0)
            self.learning.leave_training()

        self._routing_feedback(decision.selected, patched)
        finding_names = [f.signature for f in all_findings]
        return TaskReport(
            findings=finding_names,
            patched=patched,
            patch_signature=patch_signature,
            routing=decision.raw_scores,
        )
