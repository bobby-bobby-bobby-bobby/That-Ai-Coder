# Sparse Expert AutoSec

Sparse Expert AutoSec is a lightweight, self-evolving AI architecture for autonomous vulnerability discovery and repair under constrained hardware.

## Highlights
- Small stable core + sparse expert pool (2-5 experts active per task)
- Controlled learning with strict inference/training mode separation
- External memory for failures/fixes to avoid unnecessary retraining
- Sandboxed execution + fuzzing + exploitability scoring + patch validation
- Custom lightweight sparse runtime (event-driven tensor/cache abstractions)

## Quick start

```bash
python -m sparse_autosec.demo
```

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
