from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from .core import CoreModel
from .experts import ExpertPool
from .memory import ExternalMemory
from .runtime import SparseRuntime


class LearningMode(str, Enum):
    INFERENCE = "inference"
    TRAINING = "training"


@dataclass
class ReplayItem:
    text: str
    expert: str
    success_target: float


class ControlledLearningSystem:
    """Govern training lifecycle and prevent uncontrolled online updates."""

    def __init__(self, runtime: SparseRuntime, core: CoreModel, experts: ExpertPool, memory: ExternalMemory):
        self.runtime = runtime
        self.core = core
        self.experts = experts
        self.memory = memory
        self.mode = LearningMode.INFERENCE
        self.replay: List[ReplayItem] = []
        self.min_failure_threshold = 3

    def should_train(self, signature: str) -> bool:
        return self.memory.repeated_failures(signature, self.min_failure_threshold)

    def enter_training(self, signature: str) -> bool:
        if self.should_train(signature):
            self.mode = LearningMode.TRAINING
            return True
        return False

    def leave_training(self) -> None:
        self.mode = LearningMode.INFERENCE

    def record_replay(self, text: str, expert: str, success_target: float) -> None:
        self.replay.append(ReplayItem(text=text, expert=expert, success_target=success_target))
        if len(self.replay) > 128:
            self.replay.pop(0)

    def train_step(self, text: str, selected_experts: List[str], success_target: float) -> Dict[str, float]:
        if self.mode != LearningMode.TRAINING:
            return {}
        tensor = self.runtime.tokenize_sparse(text)
        metrics: Dict[str, float] = {}
        grad = {idx: val * success_target for idx, val in tensor.values.items()}
        self.core.adapter.update(grad, lr=0.001)

        for name in selected_experts:
            ex = self.experts.get(name)
            ex.train(tensor, target=success_target)
            metrics[name] = ex.confidence()
        # replay to prevent forgetting
        for item in self.replay[-16:]:
            t = self.runtime.tokenize_sparse(item.text)
            self.experts.get(item.expert).train(t, target=item.success_target, lr=0.01)
        return metrics
