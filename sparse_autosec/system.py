from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .config import AutoSecConfig
from .core import CoreModel
from .execution import Analyzer, PatchEngine, SandboxedExecutor, StructuredMutator, VulnerabilitySignal
from .experts import ExpertModule, ExpertPool
from .lifecycle import ExpertLifecycleManager
from .memory import ExternalMemory, FailureRecord, FixRecord
from .router import SparseRouter
from .runtime import SparseRuntime
from .symbolic import SymbolicPatchVerifier
from .telemetry import Telemetry
from .training import ControlledLearningSystem


@dataclass
class TaskReport:
    findings: List[str]
    exploitability: Dict[str, str]
    patched: bool
    patch_signature: str | None
    routing: Dict[str, float]
    route_rationale: Dict[str, str]
    lifecycle_events: Dict[str, List[str]] = field(default_factory=dict)
    telemetry: Dict[str, int] = field(default_factory=dict)


class SparseExpertAutoSec:
    """Complete sparse-expert vulnerability discovery and repair system."""

    def __init__(self, config: AutoSecConfig | None = None) -> None:
        self.config = config or AutoSecConfig()
        self.runtime = SparseRuntime(
            vocab_size=self.config.runtime.vocab_size,
            cache_items=self.config.runtime.cache_items,
        )
        self.core = CoreModel(self.runtime)
        self.memory = ExternalMemory()
        self.experts = ExpertPool()
        self._seed_experts()
        self.router = SparseRouter(
            self.runtime,
            self.experts,
            min_k=self.config.runtime.min_active_experts,
            max_k=self.config.runtime.max_active_experts,
        )
        self.learning = ControlledLearningSystem(
            self.runtime,
            self.core,
            self.experts,
            self.memory,
            self.config.learning,
        )
        self.telemetry = Telemetry()
        self.analyzer = Analyzer()
        self.patcher = PatchEngine()
        self.symbolic = SymbolicPatchVerifier()
        self.executor = SandboxedExecutor(
            timeout_s=self.config.execution.timeout_s,
            sandbox_prefix=self.config.execution.sandbox_prefix,
        )
        self.mutator = StructuredMutator()
        self.lifecycle = ExpertLifecycleManager(
            self.experts,
            self.memory,
            failure_threshold=self.config.learning.trigger_failures,
        )

    def _seed_experts(self) -> None:
        specs = [
            "input_validation",
            "memory_safety",
            "injection_detection",
            "auth_logic",
            "crypto_misuse",
            "deserialization",
            "race_condition",
            "null_handling",
            "path_traversal",
            "dependency_risk",
        ]
        for idx, spec in enumerate(specs, start=1):
            ex = ExpertModule(name=f"expert_{idx}_{spec}", specialty_keywords=[spec])
            ex.weights[hash(spec) % self.config.runtime.vocab_size] = 0.85
            ex.weights[hash("security") % self.config.runtime.vocab_size] = 0.25
            self.experts.add_expert(ex)

    def process_target(self, target_file: Path) -> TaskReport:
        self.telemetry.emit("process_start", file=str(target_file))
        code = target_file.read_text()
        static_findings = self.analyzer.detect_static(code)
        task_text = self._task_text(target_file, static_findings)
        complexity = self.core.score_task_complexity(task_text)
        decision = self.router.route(self.core.encode_task(task_text), complexity=complexity)
        self.telemetry.emit("routed", selected=len(decision.selected), confidence=decision.confidence)

        dynamic_findings = self._run_dynamic_scan(target_file)
        all_findings = static_findings + dynamic_findings

        patch_signature = None
        patched = False
        exploitability: Dict[str, str] = {}

        for finding in all_findings:
            exploitability[finding.signature] = finding.exploitability
            self.memory.add_failure(
                FailureRecord(
                    signature=finding.signature,
                    task_type="vuln_scan",
                    details=finding.details,
                    exploitability=finding.exploitability,
                    confidence=finding.confidence,
                )
            )
            self.telemetry.emit("failure_recorded", signature=finding.signature, confidence=int(finding.confidence * 100))

            cached_fix = self.memory.find_similar_fix(finding.signature)
            if cached_fix:
                patched, patch_signature = self._apply_cached_patch(target_file, code, cached_fix.patch_code, finding.signature)
                if patched:
                    break

            patched_code = self.patcher.propose_patch(code, finding.signature)
            if patched_code == code:
                continue

            symbolic_result = self.symbolic.verify_python_source(patched_code)
            if not symbolic_result.safe:
                self.telemetry.emit("symbolic_reject", signature=finding.signature)
                continue

            if self.patcher.validate_patch(target_file, patched_code, self.mutator):
                target_file.write_text(patched_code)
                patched = True
                patch_signature = finding.signature
                self.memory.add_fix(
                    FixRecord(
                        signature=finding.signature,
                        patch_summary="auto_generated_patch",
                        patch_code=patched_code,
                        success=True,
                    )
                )
                self.telemetry.emit("patch_applied", signature=finding.signature)
                break

        lifecycle_decision = self.lifecycle.step()
        self._training_feedback(task_text, decision.selected, patch_signature, patched)

        self.telemetry.emit("process_complete", patched=int(patched))
        return TaskReport(
            findings=[f.signature for f in all_findings],
            exploitability=exploitability,
            patched=patched,
            patch_signature=patch_signature,
            routing=decision.raw_scores,
            route_rationale=decision.rationale,
            lifecycle_events={
                "spawned": lifecycle_decision.spawned,
                "quiesced": lifecycle_decision.quiesced,
                "retired": lifecycle_decision.retired,
            },
            telemetry=self.telemetry.summary(),
        )

    def _run_dynamic_scan(self, target_file: Path) -> List[VulnerabilitySignal]:
        findings: List[VulnerabilitySignal] = []
        payloads = self.mutator.generate(self.config.execution.mutation_rounds)
        for payload in payloads[: self.config.execution.fuzz_rounds]:
            result = self.executor.run(target_file, payload)
            finding = self.analyzer.detect_dynamic(result)
            if finding:
                findings.append(finding)
                self.telemetry.emit("dynamic_finding", signature=finding.signature)
        return findings

    def _training_feedback(self, task_text: str, selected_experts: List[str], patch_signature: str | None, patched: bool) -> None:
        if not selected_experts:
            return

        target = 1.0 if patched else 0.0
        self.learning.record_replay(task_text, selected_experts[0], target)

        if patch_signature and self.learning.enter_training(patch_signature):
            metrics = self.learning.train_cycle(task_text, selected_experts, success_target=1.0)
            self.learning.leave_training()
            self.telemetry.emit("train_cycle", mean_loss=int(metrics.get("mean_loss", 0.0) * 1000))

        for expert_name in selected_experts:
            ex = self.experts.get(expert_name)
            if patched:
                ex.success_count += 1
            else:
                ex.tag_failure()

    def _apply_cached_patch(self, target_file: Path, original_code: str, cached_patch: str, signature: str) -> tuple[bool, str | None]:
        if cached_patch == original_code:
            return False, None
        if self.patcher.validate_patch(target_file, cached_patch, self.mutator):
            target_file.write_text(cached_patch)
            self.telemetry.emit("cached_patch_applied", signature=signature)
            return True, signature
        return False, None

    @staticmethod
    def _task_text(target_file: Path, static_findings: List[VulnerabilitySignal]) -> str:
        sigs = " ".join(item.signature for item in static_findings)
        return f"analyze file={target_file.name} findings={sigs}"
