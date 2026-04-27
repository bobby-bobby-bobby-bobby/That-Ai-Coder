from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List

from .budget import ResourcePolicy
from .config import AutoSecConfig
from .core import CoreModel
from .execution import Analyzer, PatchEngine, SandboxedExecutor, StructuredMutator, VulnerabilitySignal
from .experts import ExpertModule, ExpertPool
from .lifecycle import ExpertLifecycleManager
from .memory import ExternalMemory, FailureRecord, FixRecord
from .planner import TaskPlanner
from .router import SparseRouter
from .runtime import SparseRuntime
from .symbolic import SymbolicPatchVerifier
from .telemetry import Telemetry
from .training import ControlledLearningSystem
from .validator import PatchRanker


@dataclass
class TaskReport:
    findings: List[str]
    exploitability: Dict[str, str]
    exploitability_scores: Dict[str, float]
    patched: bool
    patch_signature: str | None
    routing: Dict[str, float]
    route_rationale: Dict[str, str]
    selected_experts: List[str]
    plan_steps: List[str]
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
        self.planner = TaskPlanner()
        self.policy = ResourcePolicy(
            max_latency_ms=self.config.policy.max_latency_ms,
            max_fuzz_cases=self.config.policy.max_fuzz_cases,
            max_train_steps=self.config.policy.max_train_steps,
        )
        self.ranker = PatchRanker(self.patcher, self.symbolic, self.mutator)

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
        budget = self.policy.new_state()
        self.telemetry.emit("process_start", file=str(target_file))

        code = target_file.read_text()
        static_findings = self.analyzer.detect_static(code)
        related = []
        for item in static_findings:
            related.extend(self.memory.related_signatures(item.signature))
        plan = self.planner.build_plan([item.signature for item in static_findings], related)

        task_text = self._task_text(target_file, static_findings)
        complexity = self.core.score_task_complexity(task_text)
        decision = self.router.route(self.core.encode_task(task_text), complexity=complexity)
        self.telemetry.emit("routed", selected=len(decision.selected), confidence=decision.confidence)

        dynamic_findings = self._run_dynamic_scan(target_file, budget)
        all_findings = static_findings + dynamic_findings

        patch_signature = None
        patched = False
        exploitability: Dict[str, str] = {}
        exploitability_scores: Dict[str, float] = {}

        for finding in all_findings:
            exploitability[finding.signature] = finding.exploitability
            exploitability_scores[finding.signature] = self.analyzer.exploitability_score(finding)
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

        signatures = [f.signature for f in all_findings]
        if signatures:
            cached = self._find_best_cached_fix(signatures)
            if cached and self.patcher.validate_patch(target_file, cached.patch_code, self.mutator):
                target_file.write_text(cached.patch_code)
                patched = True
                patch_signature = cached.signature
                self.telemetry.emit("cached_patch_applied", signature=cached.signature)

        if not patched and signatures:
            candidates = self.ranker.generate_candidates(code, signatures)
            accepted = self.ranker.validate(target_file, candidates)
            if accepted:
                target_file.write_text(accepted.code)
                patched = True
                patch_signature = accepted.signature
                self.memory.add_fix(
                    FixRecord(
                        signature=accepted.signature,
                        patch_summary="ranked_auto_patch",
                        patch_code=accepted.code,
                        success=True,
                    )
                )
                self.telemetry.emit("patch_applied", signature=accepted.signature)

        lifecycle_decision = self.lifecycle.step()
        self._training_feedback(task_text, decision.selected, patch_signature, patched, budget)

        reward = 1.0 if patched else 0.0
        self.router.update_reward(decision.selected, reward)

        self.telemetry.emit("budget_elapsed_ms", value=int(budget.elapsed_ms()))
        self.telemetry.emit("process_complete", patched=int(patched))
        return TaskReport(
            findings=[f.signature for f in all_findings],
            exploitability=exploitability,
            exploitability_scores=exploitability_scores,
            patched=patched,
            patch_signature=patch_signature,
            routing=decision.raw_scores,
            route_rationale=decision.rationale,
            selected_experts=decision.selected,
            plan_steps=[s.name for s in plan.active_steps()],
            lifecycle_events={
                "spawned": lifecycle_decision.spawned,
                "quiesced": lifecycle_decision.quiesced,
                "retired": lifecycle_decision.retired,
            },
            telemetry=self.telemetry.summary(),
        )

    def _run_dynamic_scan(self, target_file: Path, budget) -> List[VulnerabilitySignal]:
        findings: List[VulnerabilitySignal] = []
        payloads = self.mutator.generate(self.config.execution.mutation_rounds)
        for payload in payloads[: self.config.execution.fuzz_rounds]:
            if not budget.can_fuzz():
                self.telemetry.emit("fuzz_budget_stop", used=budget.used_fuzz_cases)
                break
            budget.used_fuzz_cases += 1
            result = self.executor.run(target_file, payload)
            finding = self.analyzer.detect_dynamic(result)
            if finding:
                findings.append(finding)
                self.telemetry.emit("dynamic_finding", signature=finding.signature)
        return findings

    def _training_feedback(self, task_text: str, selected_experts: List[str], patch_signature: str | None, patched: bool, budget) -> None:
        if not selected_experts:
            return

        target = 1.0 if patched else 0.0
        self.learning.record_replay(task_text, selected_experts[0], target)

        if patch_signature and self.learning.enter_training(patch_signature) and budget.can_train():
            metrics = self.learning.train_cycle(task_text, selected_experts, success_target=1.0)
            budget.used_train_steps += self.config.learning.max_train_steps_per_cycle
            self.learning.leave_training()
            self.telemetry.emit("train_cycle", mean_loss=int(metrics.get("mean_loss", 0.0) * 1000))

        for expert_name in selected_experts:
            ex = self.experts.get(expert_name)
            if patched:
                ex.success_count += 1
            else:
                ex.tag_failure()

    def _find_best_cached_fix(self, signatures: List[str]):
        for signature in signatures:
            fix = self.memory.find_similar_fix(signature)
            if fix:
                return fix
        return None


    def export_state(self) -> Dict[str, object]:
        return {
            "config": {
                "runtime": self.config.runtime.__dict__,
                "learning": self.config.learning.__dict__,
                "execution": self.config.execution.__dict__,
                "policy": self.config.policy.__dict__,
            },
            "core_adapter": self.core.adapter.delta,
            "experts": {
                name: {
                    "weights": ex.weights,
                    "success_count": ex.success_count,
                    "fail_count": ex.fail_count,
                    "use_count": ex.use_count,
                    "state": ex.state.value,
                }
                for name, ex in self.experts.experts.items()
            },
            "memory_counters": self.memory.counters,
            "memory_fixes": [fix.__dict__ for fix in self.memory.fixes],
        }

    def save_state(self, output_file: Path) -> Path:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(self.export_state(), indent=2))
        return output_file

    @classmethod
    def load_state(cls, state_file: Path) -> "SparseExpertAutoSec":
        payload = json.loads(state_file.read_text())
        system = cls()
        system.core.adapter.delta = {int(k): float(v) for k, v in payload.get("core_adapter", {}).items()}
        for name, rec in payload.get("experts", {}).items():
            if name in system.experts.experts:
                ex = system.experts.get(name)
                ex.weights = {int(k): float(v) for k, v in rec.get("weights", {}).items()}
                ex.success_count = int(rec.get("success_count", ex.success_count))
                ex.fail_count = int(rec.get("fail_count", ex.fail_count))
                ex.use_count = int(rec.get("use_count", ex.use_count))
        system.memory.counters = {str(k): int(v) for k, v in payload.get("memory_counters", {}).items()}
        for fix in payload.get("memory_fixes", []):
            system.memory.add_fix(FixRecord(**fix))
        return system

    @staticmethod
    def _task_text(target_file: Path, static_findings: List[VulnerabilitySignal]) -> str:
        sigs = " ".join(item.signature for item in static_findings)
        return f"analyze file={target_file.name} findings={sigs}"
