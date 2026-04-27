# Sparse Expert AutoSec Architecture

## 1. Core Model (Stable)
- `CoreModel` provides sparse task encoding and coordination scoring.
- Core weights are fixed except a small `Adapter` delta (LoRA-style sparse updates).
- All task intelligence orchestration flows through this small coordinator.

## 2. Expert Module System
- `ExpertPool` starts with 10 experts in distinct security domains.
- `ExpertModule` instances are independently trainable and tiny (sparse linear weights).
- New expert creation is gated by repeated failure signatures (threshold >= 3).

## 3. Routing / Gating
- `SparseRouter` scores experts via sparse dot products.
- Applies anti-collapse penalty (`load_penalty`) and confidence contribution.
- Selects only 2-5 experts per task (bounded by `min_k`, `max_k`).

## 4. Controlled Learning
- `ControlledLearningSystem` defines strict `INFERENCE` and `TRAINING` modes.
- No updates occur during inference mode.
- Training mode only enters when repeated failures are confirmed.
- Uses replay buffer + lightweight adapter/expert updates + post-patch validation before writing final patch.

## 5. External Memory
- `ExternalMemory` stores historical failures and successful fixes.
- Retrieval path checks prior fixes before retraining.
- Memory informs expert spawning, route nudging, and patch reuse.

## 6. Execution and Fuzzing
- `SandboxedExecutor` runs target code in temporary isolated directory.
- `StructuredMutator` generates payloads from seed corpus + structure-aware mutations.
- Captures crash and behavioral anomalies (e.g., command execution marker).

## 7. Analysis and Patching
- `Analyzer` performs static+dynamic signal extraction.
- `PatchEngine` generates deterministic patches for known classes.
- Patches are validated by re-fuzzing before final application.

## 8. Lightweight Runtime / Low Memory
- `SparseRuntime` uses hash-based sparse tokenization.
- `RuntimeCache` avoids repeated encoding allocations.
- Sparse arithmetic minimizes tensor memory footprint and computation.

## 9. Novel Innovations
1. **Event-triggered expert lifecycle management**: expert spawning happens only when statistically repeated unresolved failures happen, preventing parameter growth explosions.
2. **Success-history adaptive routing nudges**: replay integrates routing success feedback to bias future sparse selection toward historically effective experts without full retraining.

## 10. Data Flow
1. Input target enters scanner.
2. Core encodes task; router selects 2-5 experts.
3. Static analysis + sandboxed fuzzing emit vulnerability signals.
4. Memory lookup attempts fix retrieval.
5. Patch generation + isolated validation.
6. On repeated failure signatures, controlled training mode updates adapters/experts with replay.
