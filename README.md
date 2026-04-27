# Sparse Expert AutoSec

Sparse Expert AutoSec is a full, modular system for autonomous vulnerability discovery and repair under constrained hardware.

## What is included
- Stable small core model with adapter-only updates (`core.py`)
- Sparse runtime with tensor pooling + cache (`runtime.py`)
- Expert pool with lifecycle states and independent training (`experts.py`, `lifecycle.py`)
- Router with load-balancing + UCB-style anti-collapse exploration (`router.py`)
- Strict controlled learning system (`training.py`)
- Retrieval-first external memory (`memory.py`)
- Sandboxed execution + structured fuzzing + anomaly capture (`execution.py`)
- Hybrid symbolic patch verifier + deterministic patching (`symbolic.py`, `execution.py`)
- End-to-end orchestrator and telemetry (`system.py`, `telemetry.py`)
- Open-source dataset loader for training (`dataset.py`)
- Google Colab notebooks for training and router calibration (`notebooks/*.ipynb`)

## Open-source training data integration
The project integrates with the public **CVEfixes** repository:
- https://github.com/secureIT-project/CVEfixes

The Colab training notebook clones this repository and uses it as the external dataset source while supporting bootstrap samples for fast startup.

## Quick start
```bash
python -m sparse_autosec.demo
```

## Run tests
```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Colab notebooks
- `notebooks/sparse_autosec_training_colab.ipynb`
- `notebooks/sparse_autosec_router_calibration_colab.ipynb`
