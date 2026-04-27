from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RuntimeConfig:
    vocab_size: int = 2048
    cache_items: int = 1024
    max_active_experts: int = 5
    min_active_experts: int = 2


@dataclass
class LearningConfig:
    replay_size: int = 1024
    trigger_failures: int = 3
    adapter_lr: float = 0.0005
    expert_lr: float = 0.02
    max_train_steps_per_cycle: int = 24


@dataclass
class ExecutionConfig:
    timeout_s: float = 1.5
    fuzz_rounds: int = 64
    mutation_rounds: int = 40
    sandbox_prefix: str = "autosec_sandbox_"


@dataclass
class MemoryConfig:
    max_failures: int = 20_000
    max_fixes: int = 10_000
    signature_decay: float = 0.995


@dataclass
class AutoSecConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    artifact_dir: Path = Path("artifacts")
