from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from .pipeline import build_default_pipeline
from .system import SparseExpertAutoSec
from .ui_server import launch_ui
import time


def cmd_train(args: argparse.Namespace) -> int:
    pipeline = build_default_pipeline(Path(args.artifacts))
    metrics = pipeline.train_bootstrap(epochs=args.epochs)
    output = Path(args.output)
    pipeline.export_model(output)
    print(f"trained_steps={metrics['trained_steps']} samples={metrics['samples']}")
    print(f"model_export={output}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    state_file = Path(args.model)
    system = SparseExpertAutoSec.load_state(state_file) if state_file.exists() else SparseExpertAutoSec()
    report = system.process_target(Path(args.target))
    print("patched=", report.patched)
    print("signature=", report.patch_signature)
    print("findings=", report.findings)
    print("experts=", report.selected_experts)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    model = Path(args.model)
    system = SparseExpertAutoSec.load_state(model) if model.exists() else SparseExpertAutoSec()
    server, url = launch_ui(system=system, host=args.host, port=args.port)
    if args.open_browser:
        webbrowser.open(url)
    print(f"UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.stop()
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autosec", description="Sparse Expert AutoSec CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Train bootstrap model and export state")
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--artifacts", default="artifacts")
    train.add_argument("--output", default="artifacts/model_export.json")
    train.set_defaults(func=cmd_train)

    scan = sub.add_parser("scan", help="Scan/patch target source file")
    scan.add_argument("--target", required=True)
    scan.add_argument("--model", default="artifacts/model_export.json")
    scan.set_defaults(func=cmd_scan)

    ui = sub.add_parser("ui", help="Launch web UI")
    ui.add_argument("--model", default="artifacts/model_export.json")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8787)
    ui.add_argument("--open-browser", action="store_true")
    ui.set_defaults(func=cmd_ui)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
