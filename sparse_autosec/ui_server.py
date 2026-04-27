from __future__ import annotations

import json
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .dataset import OpenSourceDatasetLoader
from .pipeline import TrainingPipeline
from .system import SparseExpertAutoSec


class UIServer:
    def __init__(self, system: SparseExpertAutoSec, host: str = "127.0.0.1", port: int = 8787):
        self.system = system
        self.host = host
        self.port = port
        self.httpd: ThreadingHTTPServer | None = None

    def start(self) -> str:
        assets = Path(__file__).resolve().parent / "ui"
        handler = self._make_handler(assets)
        self.httpd = ThreadingHTTPServer((self.host, self.port), handler)
        thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        thread.start()
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def _make_handler(self, assets: Path):
        system = self.system

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/api/status":
                    return self._json({"ok": True, "experts": len(system.experts.names(active_only=False))})
                if parsed.path == "/":
                    return self._file(assets / "index.html", "text/html; charset=utf-8")
                if parsed.path == "/app.js":
                    return self._file(assets / "app.js", "application/javascript")
                if parsed.path == "/style.css":
                    return self._file(assets / "style.css", "text/css")
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def do_POST(self):
                parsed = urlparse(self.path)
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else ""
                payload = json.loads(raw) if raw else {}

                if parsed.path == "/api/train":
                    loader = OpenSourceDatasetLoader(system.config.artifact_dir / "datasets")
                    pipeline = TrainingPipeline(system, loader)
                    metrics = pipeline.train_bootstrap(epochs=int(payload.get("epochs", 1)))
                    export_path = system.config.artifact_dir / "model_export.json"
                    pipeline.export_model(export_path)
                    return self._json({"ok": True, "metrics": metrics, "model": str(export_path)})

                if parsed.path == "/api/scan":
                    code = payload.get("code", "")
                    with tempfile.TemporaryDirectory(prefix="autosec_ui_") as td:
                        target = Path(td) / "ui_target.py"
                        target.write_text(code)
                        report = system.process_target(target)
                        patched = target.read_text()
                    return self._json(
                        {
                            "ok": True,
                            "patched": report.patched,
                            "patch_signature": report.patch_signature,
                            "findings": report.findings,
                            "selected_experts": report.selected_experts,
                            "plan_steps": report.plan_steps,
                            "patched_code": patched,
                        }
                    )

                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def log_message(self, format: str, *args):
                return

            def _file(self, path: Path, content_type: str):
                if not path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND, "missing file")
                    return
                body = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj: dict):
                body = json.dumps(obj).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler


def launch_ui(system: SparseExpertAutoSec, host: str = "127.0.0.1", port: int = 8787) -> tuple[UIServer, str]:
    server = UIServer(system=system, host=host, port=port)
    url = server.start()
    return server, url
