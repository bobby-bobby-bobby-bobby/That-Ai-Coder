# Sparse Expert AutoSec

Sparse Expert AutoSec is a full, modular system for autonomous vulnerability discovery and repair under constrained hardware.

## What is included
- Stable small core model with adapter-only updates (`core.py`)
- Sparse runtime with tensor pooling + cache (`runtime.py`)
- Expert pool with lifecycle states and independent training (`experts.py`, `lifecycle.py`)
- Router with load-balancing + UCB + adaptive bandit feedback (`router.py`, `bandit.py`)
- Event-driven task planner (`planner.py`)
- Runtime resource policy (latency/fuzz/training budgets) (`budget.py`)
- Strict controlled learning system (`training.py`)
- Retrieval-first external memory (`memory.py`)
- Sandboxed execution + structured fuzzing + anomaly capture (`execution.py`)
- Hybrid symbolic patch verifier + ranked patch selection (`symbolic.py`, `validator.py`)
- End-to-end orchestrator and telemetry (`system.py`, `telemetry.py`)
- Open-source dataset loader + pipeline (`dataset.py`, `pipeline.py`)
- CLI + local web UI server for easy use (`cli.py`, `ui_server.py`, `ui/*`)

## Open-source training data integration
The project integrates with the public **CVEfixes** repository:
- https://github.com/secureIT-project/CVEfixes

## Fast local usage
### 1) Train + export model
```bash
python -m sparse_autosec.cli train --epochs 2 --output artifacts/model_export.json
```

### 2) Launch UI (opens browser)
```bash
python -m sparse_autosec.cli ui --model artifacts/model_export.json --open-browser
```

### 3) Terminal scan mode
```bash
python -m sparse_autosec.cli scan --model artifacts/model_export.json --target path/to/target.py
```

## Developer checks
```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m sparse_autosec.demo
```

## Colab notebooks
- `notebooks/sparse_autosec_training_colab.ipynb`
- `notebooks/sparse_autosec_router_calibration_colab.ipynb`







In PROGRESS
