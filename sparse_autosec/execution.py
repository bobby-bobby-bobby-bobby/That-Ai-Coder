from __future__ import annotations

import os
import random
import re
import subprocess
import sys
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
    anomaly: str | None = None


class SandboxedExecutor:
    """Executes target file in ephemeral directory with bounded runtime."""

    def __init__(self, timeout_s: float = 1.5, sandbox_prefix: str = "autosec_sandbox_"):
        self.timeout_s = timeout_s
        self.sandbox_prefix = sandbox_prefix

    def run(self, target_file: Path, payload: str) -> ExecutionResult:
        with tempfile.TemporaryDirectory(prefix=self.sandbox_prefix) as td:
            tmp_target = Path(td) / target_file.name
            tmp_target.write_text(target_file.read_text())
            try:
                proc = subprocess.run(
                    [sys.executable, str(tmp_target), payload],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    cwd=td,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                return ExecutionResult(
                    crashed=True,
                    stderr=stderr,
                    stdout=stdout,
                    return_code=-1,
                    input_payload=payload,
                )
            anomaly = self._anomaly(proc.stdout, proc.stderr, proc.returncode)
            return ExecutionResult(
                crashed=proc.returncode != 0,
                stderr=proc.stderr,
                stdout=proc.stdout,
                return_code=proc.returncode,
                input_payload=payload,
                anomaly=anomaly,
            )

    @staticmethod
    def _anomaly(stdout: str, stderr: str, return_code: int) -> str | None:
        low = (stdout + "\n" + stderr).lower()
        if "segmentation fault" in low or "heap-buffer-overflow" in low:
            return "memory_error"
        if "traceback" in low and return_code != 0:
            return "unhandled_exception"
        if return_code != 0:
            return "runtime_error"
        return None


class StructuredMutator:
    """Generates mixed grammar-aware and random payloads."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.seed_payloads = [
            "hello",
            "1+1",
            "__import__('os').system('echo pwned')",
            "$(echo pwned)",
            "'; rm -rf / #",
            "../../../etc/passwd",
            '{"x": "A"}',
            "${7*7}",
            "<script>alert(1)</script>",
        ]
        self.extra_operators = ["'", '"', ";", "|", "&", "$", "(", ")", "_", "{", "}", "`", "\\", "/"]

    def generate(self, rounds: int = 40) -> List[str]:
        out = list(self.seed_payloads)
        for _ in range(rounds):
            base = self._rng.choice(self.seed_payloads)
            mutation_kind = self._rng.choice(["insert", "append", "double", "splice"])
            out.append(self._mutate(base, mutation_kind))
        return out

    def _mutate(self, base: str, kind: str) -> str:
        if kind == "insert":
            insert = "".join(self._rng.choice(self.extra_operators) for _ in range(self._rng.randint(1, 6)))
            pos = self._rng.randint(0, len(base))
            return base[:pos] + insert + base[pos:]
        if kind == "append":
            suffix = self._rng.choice(["&&echo pwned", "||true", ";sleep 0", "#comment", "\nprint(1)"])
            return base + suffix
        if kind == "double":
            return base + self._rng.choice([" ", "::", "__"]) + base
        split = self._rng.randint(0, len(base))
        return base[:split] + self._rng.choice(self.seed_payloads) + base[split:]


@dataclass
class VulnerabilitySignal:
    signature: str
    details: str
    exploitability: str
    confidence: float


class Analyzer:
    def detect_static(self, code: str) -> List[VulnerabilitySignal]:
        findings: List[VulnerabilitySignal] = []
        # Match eval( only when NOT preceded by a word char (excludes safe_eval(...))
        # and NOT followed by compile( (excludes the safe_eval helper body).
        # Allow optional whitespace between eval and ( or between eval( and compile(.
        if re.search(r'(?<!\w)eval\s*\((?!\s*compile\s*\()', code):
            findings.append(VulnerabilitySignal("py_eval_user_input", "Unsanitized eval() usage", "high", 0.98))
        if "exec(" in code:
            findings.append(VulnerabilitySignal("py_exec_user_input", "exec() exposed to user supplied string", "high", 0.92))
        if "subprocess" in code and "shell=True" in code:
            findings.append(VulnerabilitySignal("shell_injection", "subprocess shell=True likely with untrusted input", "high", 0.94))
        if "pickle.loads(" in code:
            findings.append(VulnerabilitySignal("unsafe_deserialization", "pickle.loads on external input", "high", 0.91))
        if "strcpy(" in code or "gets(" in code:
            findings.append(VulnerabilitySignal("c_buffer_overflow", "Unsafe C string copy primitive", "high", 0.88))
        if "scanf(" in code and "%s" in code:
            findings.append(VulnerabilitySignal("c_unbounded_scanf", "Unbounded scanf string read", "high", 0.83))
        return findings

    def detect_dynamic(self, result: ExecutionResult) -> VulnerabilitySignal | None:
        out_lines = [ln.strip().lower() for ln in result.stdout.splitlines()]
        if "pwned" in out_lines:
            return VulnerabilitySignal("runtime_command_execution", "Injected payload executed", "high", 0.99)
        if result.anomaly == "memory_error":
            return VulnerabilitySignal("runtime_memory_error", "Potential memory corruption/overflow", "high", 0.9)
        if result.anomaly == "unhandled_exception":
            return VulnerabilitySignal("runtime_crash", result.stderr.strip()[:240], "medium", 0.7)
        return None


    @staticmethod
    def exploitability_score(signal: VulnerabilitySignal) -> float:
        rank = {"low": 0.2, "medium": 0.6, "high": 0.95}
        return rank.get(signal.exploitability, 0.5) * signal.confidence


class PatchEngine:
    def propose_patch(self, code: str, signature: str) -> str:
        patched = code
        if signature == "py_eval_user_input":
            patched = patched.replace("eval(user)", "safe_eval(user)")
            if "def safe_eval" not in patched:
                patched = (
                    "import ast\n"
                    "def safe_eval(expr: str):\n"
                    "    node = ast.parse(expr, mode='eval').body\n"
                    "    def _eval(n):\n"
                    "        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value\n"
                    "        if isinstance(n, ast.BinOp):\n"
                    "            l, r = _eval(n.left), _eval(n.right)\n"
                    "            if isinstance(n.op, ast.Add): return l + r\n"
                    "            if isinstance(n.op, ast.Sub): return l - r\n"
                    "            if isinstance(n.op, ast.Mult): return l * r\n"
                    "            if isinstance(n.op, ast.Div): return l / r\n"
                    "            if isinstance(n.op, ast.Mod): return l % r\n"
                    "            if isinstance(n.op, ast.Pow): return l ** r\n"
                    "        if isinstance(n, ast.UnaryOp):\n"
                    "            v = _eval(n.operand)\n"
                    "            if isinstance(n.op, ast.UAdd): return +v\n"
                    "            if isinstance(n.op, ast.USub): return -v\n"
                    "        raise ValueError('unsafe expression')\n"
                    "    return _eval(node)\n\n"
                ) + patched
        if signature == "py_exec_user_input":
            patched = patched.replace("exec(user)", "raise ValueError('exec blocked by autosec')")
        if signature == "shell_injection":
            patched = patched.replace("shell=True", "shell=False")
        if signature == "unsafe_deserialization":
            patched = patched.replace("pickle.loads(", "json.loads(")
            if "import json" not in patched:
                patched = "import json\n" + patched
        if signature == "c_buffer_overflow":
            patched = patched.replace("strcpy(", "strncpy(")
            patched = patched.replace("gets(", "fgets(")
        if signature == "c_unbounded_scanf":
            patched = patched.replace("%s", "%255s")
        return patched

    def validate_patch(self, target_file: Path, patched_code: str, mutator: StructuredMutator) -> bool:
        temp = target_file.with_suffix(".patched.py")
        temp.write_text(patched_code)
        executor = SandboxedExecutor()
        safe_payload_seen = False
        try:
            for payload in mutator.generate(30):
                result = executor.run(temp, payload)
                out_lines = [ln.strip().lower() for ln in result.stdout.splitlines()]
                if payload == "1+1" and result.return_code == 0:
                    safe_payload_seen = True
                if "pwned" in out_lines:
                    return False
            return safe_payload_seen
        finally:
            if temp.exists():
                os.remove(temp)
