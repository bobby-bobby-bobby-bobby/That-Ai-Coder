# Sparse Expert AutoSec Architecture (Expanded)

## 1) Stable Core Model
- `CoreModel` is intentionally small and stable.
- Only `Adapter` deltas update during controlled training cycles.
- Core functions:
  - task encoding (`encode_task`)
  - complexity estimation (`score_task_complexity`)
  - planning tags (`propose_plan_tags`)

## 2) Sparse Expert Modules
- 10 seed experts initialized at boot across distinct vulnerability domains.
- Expert structure:
  - sparse weights
  - independent success/failure counters
  - health/confidence signals
  - lifecycle state (`ACTIVE`, `QUIESCENT`, `RETIRED`)

## 3) Controlled Expert Lifecycle (Innovation #1)
- `ExpertLifecycleManager` handles:
  - spawn: only when repeated signature failures pass threshold
  - quiesce: low-health experts moved out of active routing set
  - retire: persistent low-health quiescent experts retired
- Prevents random/frequent expert growth and caps instability.

## 4) Routing / Gating
- Router computes sparse scores with four terms:
  - exploit score (expert sparse dot)
  - confidence bonus
  - UCB exploration bonus
  - load penalty
- Router selects only 2–5 experts depending on estimated task complexity.

## 5) Controlled Learning System
- Two modes:
  - `INFERENCE`: no updates
  - `TRAINING`: explicit, threshold-gated cycle only
- Includes:
  - replay buffer
  - adapter update (parameter-efficient)
  - per-expert tiny updates
  - replay rehearsal to reduce forgetting

## 6) External Memory
- Stores failure records, fix records (with patch code), and tokenized pattern summaries.
- Retrieval-first behavior: cached successful fix is attempted before generating a new patch.
- Memory informs:
  - expert spawning conditions
  - related signature retrieval
  - training triggers

## 7) Execution & Fuzzing
- `SandboxedExecutor` runs target in ephemeral temp sandbox.
- `StructuredMutator` performs hybrid seed + structural mutations.
- Captures:
  - crashes
  - command execution behavior
  - anomaly categories (runtime, unhandled exception, memory error)

## 8) Analysis + Patching + Validation
- `Analyzer` combines static and dynamic signals.
- `PatchEngine` deterministically transforms vulnerable patterns.
- Patch is accepted only after:
  1. symbolic safety checks (`SymbolicPatchVerifier`) (Innovation #2: hybrid symbolic+neural control)
  2. dynamic re-validation under mutated payloads

## 9) Lightweight Runtime
- `SparseTensor` for compact representation.
- `TensorPool` to reduce allocations.
- `RuntimeCache` for incremental inference on repeated tasks.

## 10) Colab training workflow
- Notebook: `notebooks/sparse_autosec_training_colab.ipynb`
  - clones CVEfixes open-source dataset repo
  - builds bootstrapped and dataset-driven samples
  - runs controlled training cycles
  - exports adapter checkpoint
- Notebook: `notebooks/sparse_autosec_router_calibration_colab.ipynb`
  - simulates repeated failure streams
  - demonstrates lifecycle spawn/quiesce/retire behavior

## 11) Why this design is efficient
- Sparse activation across experts means only a few modules run per task.
- Adapter-only core updates avoid catastrophic overwrite and reduce memory pressure.
- Retrieval-first memory avoids unnecessary retraining.
- Lifecycle pruning limits model growth and long-term computational drift.
