# Sparse Expert AutoSec Architecture (Further Expanded)

## 1) Stable Core Model
- `CoreModel` remains intentionally small and stable.
- Only `Adapter` deltas update during controlled training cycles.
- Core produces complexity estimates and plan tags.

## 2) Sparse Expert Modules
- 10 seed experts initialize distinct vulnerability competencies.
- Experts remain tiny and independently trainable.
- Each expert tracks confidence, failures, health, and lifecycle state.

## 3) Controlled Expert Lifecycle
- `ExpertLifecycleManager` supports:
  - spawn on repeated statistically significant failures
  - quiesce low-health experts
  - retire persistently weak experts
- This keeps growth intentional and bounded.

## 4) Routing/Gating + Adaptive Bandit
- `SparseRouter` computes sparse scores with:
  - exploit score
  - confidence bonus
  - exploration bonus
  - load penalty
- `RoutingBandit` adds history-based adaptation so the router improves from outcomes over time.
- Router still activates only 2–5 experts per task.

## 5) Resource Policy Layer
- `ResourcePolicy` enforces hard limits:
  - end-to-end latency budget
  - fuzz case count budget
  - train step budget
- This ensures operation stays practical on constrained hardware.

## 6) Task Planning Layer
- `TaskPlanner` builds compact event-driven plans (`ExecutionPlan`) from detected findings + memory context.
- The system executes only necessary steps, reducing wasteful computation.

## 7) Controlled Learning System
- Two explicit modes:
  - `INFERENCE`: immutable weights
  - `TRAINING`: gated adaptation only
- Includes replay + capped cycles + adapter/expert lightweight updates.

## 8) External Memory
- Stores failures, fixes (including patch code), and tokenized failure patterns.
- Retrieval-first behavior tries known successful fixes before retraining.
- Memory also drives lifecycle and planning decisions.

## 9) Execution + Fuzzing + Analysis
- `SandboxedExecutor` runs targets in isolated temp sandboxes.
- `StructuredMutator` produces structured adversarial payloads.
- `Analyzer` now detects both Python and selected C-family vulnerability patterns and computes exploitability scores.

## 10) Patching + Verification
- `PatchRanker` generates multiple candidate patches and ranks them.
- `SymbolicPatchVerifier` rejects unsafe patch AST patterns.
- Dynamic re-validation confirms candidate safety before application.

## 11) Novel Ideas
1. **Controlled expert lifecycle with health states** to cap model growth while preserving specialization.
2. **Bandit-augmented sparse routing** that improves expert selection using reward signals without dense retraining.
3. **Budget-aware event planning** that adapts behavior to latency/fuzz/training constraints.

## 12) Colab Training Workflow
- `notebooks/sparse_autosec_training_colab.ipynb`
  - clones CVEfixes
  - runs controlled training cycles
  - exports adapter checkpoints
- `notebooks/sparse_autosec_router_calibration_colab.ipynb`
  - simulates repeated failures
  - demonstrates lifecycle + adaptive routing behavior
