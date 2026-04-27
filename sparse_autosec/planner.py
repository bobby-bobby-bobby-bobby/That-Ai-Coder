from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PlanStep:
    name: str
    priority: int
    enabled: bool = True
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    steps: List[PlanStep]

    def active_steps(self) -> List[PlanStep]:
        return [step for step in sorted(self.steps, key=lambda s: s.priority) if step.enabled]


class TaskPlanner:
    """Builds a compact event-driven plan from static evidence and history."""

    def build_plan(self, findings: List[str], related_signatures: List[str]) -> ExecutionPlan:
        steps: List[PlanStep] = [
            PlanStep(name="static_scan", priority=1),
            PlanStep(name="route_experts", priority=2),
            PlanStep(name="fuzz_execute", priority=3),
            PlanStep(name="patch_rank", priority=4),
            PlanStep(name="validate_patch", priority=5),
        ]

        if any(("memory" in sig or "overflow" in sig or sig.startswith("c_")) for sig in findings + related_signatures):
            steps.append(PlanStep(name="memory_guard_checks", priority=3, metadata={"mode": "extra"}))
        if any("deserial" in sig for sig in findings + related_signatures):
            steps.append(PlanStep(name="deserialization_hardening", priority=4, metadata={"mode": "extra"}))

        return ExecutionPlan(steps=steps)
