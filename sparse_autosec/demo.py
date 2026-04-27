from pathlib import Path

from .system import SparseExpertAutoSec


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    target = repo_root / "sparse_autosec" / "demo_target.py"

    system = SparseExpertAutoSec()
    report = system.process_target(target)

    print("=== Sparse Expert AutoSec Demo ===")
    print("Findings:", report.findings)
    print("Patched:", report.patched)
    print("Patch signature:", report.patch_signature)
    print("Top routing scores:")
    top = sorted(report.routing.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, score in top:
        print(f"  {name}: {score:.3f}")


if __name__ == "__main__":
    main()
