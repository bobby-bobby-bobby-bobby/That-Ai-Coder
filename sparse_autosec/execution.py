from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ExecutionResult:
    crashed: bool
    stderr: str
    stdout: str
    return_code: int
    input_payload: str


class SandboxedExecutor:
    """Runs target script in isolated temp directory with strict timeout."""

    def __init__(self, timeout_s: float = 1.2):
        self.timeout_s = timeout_s

    def run(self, target_file: Path, payload: str) -> ExecutionResult:
        with tempfile.TemporaryDirectory(prefix="autosec_sandbox_") as td:
            tmp_target = Path(td) / target_file.name
            tmp_target.write_text(target_file.read_text())
            proc = subprocess.run(
                ["python", str(tmp_target), payload],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=td,
            )
            crashed = proc.returncode != 0
            return ExecutionResult(
                crashed=crashed,
                stderr=proc.stderr,
                stdout=proc.stdout,
                return_code=proc.returncode,
                input_payload=payload,
            )


class StructuredMutator:
    """Structured fuzzing for command/code injection style payloads."""

    def __init__(self) -> None:
        self.seed_payloads = [
            "hello",
            "1+1",
            "__import__('os').system('echo pwned')",
            "$(echo pwned)",
            "'; rm -rf / #",
            '{"x": "A"}',
        ]

    def generate(self, rounds: int = 20) -> List[str]:
        out = list(self.seed_payloads)
        alphabet = ["'", '"', ";", "|", "&", "$", "(", ")", "_", "{"]
        for _ in range(rounds):
            base = random.choice(self.seed_payloads)
            insert = "".join(random.choice(alphabet) for _ in range(random.randint(1, 4)))
            pos = random.randint(0, len(base))
            out.append(base[:pos] + insert + base[pos:])
        return out


class VulnerabilitySignal:
    def __init__(self, signature: str, details: str, exploitability: str):
        self.signature = signature
        self.details = details
        self.exploitability = exploitability


class Analyzer:
    def detect_static(self, code: str) -> List[VulnerabilitySignal]:
        findings: List[VulnerabilitySignal] = []
        if "eval(" in code:
            findings.append(VulnerabilitySignal("py_eval_user_input", "Unsanitized eval() usage", "high"))
        if "os.system(" in code and "sys.argv" in code:
            findings.append(VulnerabilitySignal("shell_injection", "os.system with user input", "high"))
        return findings

    def detect_dynamic(self, result: ExecutionResult) -> VulnerabilitySignal | None:
        out_lines = [ln.strip().lower() for ln in result.stdout.splitlines()]
        if "pwned" in out_lines:
            return VulnerabilitySignal("runtime_command_execution", "Injected payload executed", "high")
        if result.crashed:
            return VulnerabilitySignal("runtime_crash", result.stderr.strip()[:200], "medium")
        return None


class PatchEngine:
    """Generate deterministic patches and re-validate in isolation."""

    def propose_patch(self, code: str, signature: str) -> str:
        patched = code
        if signature == "py_eval_user_input":
            patched = patched.replace("eval(user)", "safe_eval(user)")
            if "def safe_eval" not in patched:
                patched = (
                    "import ast\n"
                    "def safe_eval(expr: str):\n"
                    "    node = ast.parse(expr, mode='eval')\n"
                    "    allowed = (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)\n"
                    "    for n in ast.walk(node):\n"
                    "        if not isinstance(n, allowed):\n"
                    "            raise ValueError('unsafe expression')\n"
                    "    return eval(compile(node, '<safe_eval>', 'eval'))\n\n"
                ) + patched
        if signature == "shell_injection":
            patched = patched.replace("os.system(user)", "print('blocked shell command')")
        return patched

    def validate_patch(self, target_file: Path, patched_code: str, mutator: StructuredMutator) -> bool:
        temp = target_file.with_suffix(".patched.py")
        temp.write_text(patched_code)
        executor = SandboxedExecutor()
        try:
            for payload in mutator.generate(15):
                result = executor.run(temp, payload)
                out_lines = [ln.strip().lower() for ln in result.stdout.splitlines()]
                if "pwned" in out_lines:
                    return False
            return True
        finally:
            if temp.exists():
                os.remove(temp)
