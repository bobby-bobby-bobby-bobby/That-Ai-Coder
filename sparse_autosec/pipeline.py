from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .config import AutoSecConfig
from .dataset import OpenSourceDatasetLoader, TrainingSample
from .system import SparseExpertAutoSec


class TrainingPipeline:
    """High-level training/export pipeline for CLI and UI workflows."""

    def __init__(self, system: SparseExpertAutoSec, dataset_loader: OpenSourceDatasetLoader):
        self.system = system
        self.dataset_loader = dataset_loader

    def train_bootstrap(self, epochs: int = 2) -> Dict[str, float]:
        samples = self.dataset_loader.build_bootstrap_samples()
        trained = 0
        for _ in range(max(1, epochs)):
            for sample in samples:
                trained += self._train_sample(sample)
        return {"samples": float(len(samples)), "trained_steps": float(trained)}

    def _train_sample(self, sample: TrainingSample) -> int:
        task = self.dataset_loader.as_task_text(sample)
        decision = self.system.router.route(
            self.system.core.encode_task(task),
            complexity=self.system.core.score_task_complexity(task),
        )
        target = 1.0 if sample.label == 1 else 0.0
        self.system.learning.record_replay(task, decision.selected[0], target)
        if target > 0.0:
            self.system.memory.counters[sample.signature] = max(3, self.system.memory.counters.get(sample.signature, 0))
            if self.system.learning.enter_training(sample.signature):
                self.system.learning.train_cycle(task, decision.selected, success_target=1.0)
                self.system.learning.leave_training()
                self.system.router.update_reward(decision.selected, reward=1.0)
                return 1
        self.system.router.update_reward(decision.selected, reward=target)
        return 0

    def export_model(self, output_file: Path) -> Path:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = self.system.export_state()
        output_file.write_text(json.dumps(payload, indent=2))
        return output_file


def build_default_pipeline(artifact_dir: Path | None = None) -> TrainingPipeline:
    config = AutoSecConfig()
    if artifact_dir is not None:
        config.artifact_dir = artifact_dir
    system = SparseExpertAutoSec(config=config)
    loader = OpenSourceDatasetLoader(config.artifact_dir / "datasets")
    return TrainingPipeline(system=system, dataset_loader=loader)
