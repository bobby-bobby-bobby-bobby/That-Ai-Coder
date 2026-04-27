from __future__ import annotations

import tempfile
from pathlib import Path

from .system import SparseExpertAutoSec


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    source_target = repo_root / "sparse_autosec" / "demo_target.py"

    with tempfile.TemporaryDirectory(prefix="autosec_demo_") as td:
        target = Path(td) / "demo_target.py"
        target.write_text(source_target.read_text())

        system = SparseExpertAutoSec()
        report = system.process_target(target)

        print("=== Sparse Expert AutoSec Demo ===")
        print("Findings:", report.findings)
        print("Exploitability:", report.exploitability)
        print("Patched:", report.patched)
        print("Patch signature:", report.patch_signature)
        print("Plan steps:", report.plan_steps)
        print("Selected experts:", report.selected_experts)
        print("Lifecycle events:", report.lifecycle_events)
        print("Exploitability scores:", report.exploitability_scores)
        print("Telemetry:", report.telemetry)
        print("Top routing scores:")
        top = sorted(report.routing.items(), key=lambda x: x[1], reverse=True)[:5]
        for name, score in top:
            rationale = report.route_rationale.get(name, "")
            print(f"  {name}: {score:.3f} | {rationale}")


if __name__ == "__main__":
    main()
