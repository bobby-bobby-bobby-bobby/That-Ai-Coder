from pathlib import Path
import tempfile
import unittest

from sparse_autosec.system import SparseExpertAutoSec
from sparse_autosec.runtime import SparseRuntime


class TestSparseRuntime(unittest.TestCase):
    def test_sparse_tokenization_cache(self):
        rt = SparseRuntime(vocab_size=64)
        t1 = rt.tokenize_sparse("hello world")
        t2 = rt.tokenize_sparse("hello world")
        self.assertEqual(t1.values, t2.values)


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
            system = SparseExpertAutoSec()
            report = system.process_target(target)
            patched_code = target.read_text()

            self.assertIn("py_eval_user_input", report.findings)
            self.assertTrue(report.patched)
            self.assertIn("safe_eval", patched_code)


if __name__ == "__main__":
    unittest.main()
