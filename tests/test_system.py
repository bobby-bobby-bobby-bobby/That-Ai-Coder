from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from sparse_autosec.config import AutoSecConfig
from sparse_autosec.dataset import OpenSourceDatasetLoader
from sparse_autosec.runtime import SparseRuntime
from sparse_autosec.symbolic import SymbolicPatchVerifier
from sparse_autosec.system import SparseExpertAutoSec


class TestSparseRuntime(unittest.TestCase):
    def test_sparse_tokenization_cache_and_merge(self):
        rt = SparseRuntime(vocab_size=128, cache_items=16)
        t1 = rt.tokenize_sparse("hello world")
        t2 = rt.tokenize_sparse("hello world")
        merged = rt.merge_texts(["hello", "world", "hello"], normalize=False)
        self.assertEqual(t1.values, t2.values)
        self.assertGreaterEqual(len(merged.values), len(t1.values))


class TestSymbolicVerifier(unittest.TestCase):
    def test_rejects_unsafe_calls(self):
        verifier = SymbolicPatchVerifier()
        result = verifier.verify_python_source("def run(x):\n    return eval(x)")
        self.assertFalse(result.safe)


class TestDatasetLoader(unittest.TestCase):
    def test_bootstrap_samples(self):
        with tempfile.TemporaryDirectory() as td:
            loader = OpenSourceDatasetLoader(Path(td))
            samples = loader.build_bootstrap_samples()
            split = loader.split(samples)
            self.assertGreaterEqual(len(split["train"]), 1)
            self.assertIn(1, loader.class_balance(samples))


class TestEndToEnd(unittest.TestCase):
    def test_vulnerability_detection_and_patch(self):
        source = """
import sys

def run(user):
    return eval(user)

if __name__ == '__main__':
    print(run(sys.argv[1]))
""".strip()
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "target.py"
            target.write_text(source)

            config = AutoSecConfig()
            config.execution.fuzz_rounds = 16
            config.execution.mutation_rounds = 16

            system = SparseExpertAutoSec(config=config)
            report = system.process_target(target)
            patched_code = target.read_text()

            self.assertIn("py_eval_user_input", report.findings)
            self.assertTrue(report.patched)
            self.assertIn("safe_eval", patched_code)
            self.assertIn("process_complete", report.telemetry)


if __name__ == "__main__":
    unittest.main()
