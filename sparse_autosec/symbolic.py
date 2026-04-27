from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List


@dataclass
class SymbolicConstraintResult:
    safe: bool
    reasons: List[str]


class SymbolicPatchVerifier:
    """Hybrid symbolic rule checker for generated patches.

    This avoids expensive neural validation for obvious unsafe constructs.
    """

    unsafe_calls = {"eval", "exec"}

    def verify_python_source(self, code: str) -> SymbolicConstraintResult:
        reasons: List[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return SymbolicConstraintResult(False, [f"syntax_error:{exc.msg}"])

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._call_name(node)
                if name in self.unsafe_calls:
                    reasons.append(f"unsafe_call:{name}")
            if isinstance(node, ast.keyword) and node.arg == "shell" and getattr(node.value, "value", None) is True:
                reasons.append("shell_true")

        return SymbolicConstraintResult(len(reasons) == 0, reasons)

    def _call_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return "unknown"
