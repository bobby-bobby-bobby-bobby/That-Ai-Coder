from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import mean
from typing import Dict, List

from .config import LearningConfig
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
    """Controlled adaptation with explicit training cycles and validation gates."""

    def __init__(
        self,
        runtime: SparseRuntime,
        core: CoreModel,
        experts: ExpertPool,
        memory: ExternalMemory,
        config: LearningConfig,
    ):
        self.runtime = runtime
        self.core = core
        self.experts = experts
        self.memory = memory
        self.config = config
        self.mode = LearningMode.INFERENCE
        self.replay: List[ReplayItem] = []

    def should_train(self, signature: str) -> bool:
        return self.memory.repeated_failures(signature, self.config.trigger_failures)

    def enter_training(self, signature: str) -> bool:
        if self.should_train(signature):
            self.mode = LearningMode.TRAINING
            return True
        return False

    def leave_training(self) -> None:
        self.mode = LearningMode.INFERENCE

    def record_replay(self, text: str, expert: str, success_target: float) -> None:
        self.replay.append(ReplayItem(text=text, expert=expert, success_target=success_target))
        if len(self.replay) > self.config.replay_size:
            self.replay.pop(0)

    def train_cycle(self, task_text: str, selected_experts: List[str], success_target: float) -> Dict[str, float]:
        if self.mode != LearningMode.TRAINING:
            return {}
        metrics: Dict[str, float] = {}
        losses: List[float] = []

        for step in range(self.config.max_train_steps_per_cycle):
            text = task_text if step == 0 else self.replay_text(step)
            tensor = self.runtime.tokenize_sparse(text)
            grads = {idx: val * success_target for idx, val in tensor.values.items()}
            self.core.adapter.update(grads, lr=self.config.adapter_lr)

            for expert_name in selected_experts:
                ex = self.experts.get(expert_name)
                before = ex.score(tensor)
                ex.train(tensor, target=success_target, lr=self.config.expert_lr)
                after = ex.score(tensor)
                losses.append(abs(success_target - after))
                metrics[f"{expert_name}_improvement"] = after - before

        # replay to reduce forgetting
        for item in self.replay[-32:]:
            t = self.runtime.tokenize_sparse(item.text)
            self.experts.get(item.expert).train(t, target=item.success_target, lr=self.config.expert_lr * 0.5)

        metrics["mean_loss"] = mean(losses) if losses else 0.0
        metrics["mode"] = 1.0 if self.mode == LearningMode.TRAINING else 0.0
        return metrics

    def replay_text(self, step: int) -> str:
        if not self.replay:
            return ""
        item = self.replay[step % len(self.replay)]
        return item.text
