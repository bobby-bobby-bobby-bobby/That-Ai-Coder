from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import urllib.request

from sparse_autosec.bandit import RoutingBandit
from sparse_autosec.budget import ResourcePolicy
from sparse_autosec.config import AutoSecConfig
from sparse_autosec.dataset import OpenSourceDatasetLoader
from sparse_autosec.execution import Analyzer, PatchEngine, StructuredMutator
from sparse_autosec.pipeline import build_default_pipeline
from sparse_autosec.planner import TaskPlanner
from sparse_autosec.runtime import SparseRuntime
from sparse_autosec.symbolic import SymbolicPatchVerifier
from sparse_autosec.system import SparseExpertAutoSec
from sparse_autosec.ui_server import launch_ui
from sparse_autosec.validator import PatchRanker


class TestSparseRuntime(unittest.TestCase):
    def test_sparse_tokenization_cache_and_merge(self):
        rt = SparseRuntime(vocab_size=128, cache_items=16)
        t1 = rt.tokenize_sparse("hello world")
        t2 = rt.tokenize_sparse("hello world")
        self.assertIs(t1, t2)
        merged = rt.merge_texts(["hello", "world", "hello"], normalize=False)
        self.assertEqual(t1.values, t2.values)
        self.assertGreaterEqual(len(merged.values), len(t1.values))


class TestPlannerBudgetBandit(unittest.TestCase):
    def test_planner_and_policy(self):
        planner = TaskPlanner()
        plan = planner.build_plan(["c_buffer_overflow"], ["runtime_crash"])
        names = [s.name for s in plan.active_steps()]
        self.assertIn("memory_guard_checks", names)

        policy = ResourcePolicy(max_latency_ms=1000, max_fuzz_cases=2, max_train_steps=1)
        state = policy.new_state()
        self.assertTrue(state.can_fuzz())
        state.used_fuzz_cases = 2
        self.assertFalse(state.can_fuzz())

    def test_bandit_reward_update(self):
        bandit = RoutingBandit()
        scores = {"a": 0.2, "b": 0.1}
        ranked_before = bandit.score(["a", "b"], scores)
        bandit.update(["b"], reward=1.0)
        ranked_after = bandit.score(["a", "b"], scores)
        self.assertGreaterEqual(ranked_after[0][1], ranked_before[0][1])


class TestSymbolicAndPatching(unittest.TestCase):
    def test_rejects_unsafe_calls_and_ranks_candidates(self):
        verifier = SymbolicPatchVerifier()
        result = verifier.verify_python_source("def run(x):\n    return eval(x)")
        self.assertFalse(result.safe)

        patcher = PatchEngine()
        mutator = StructuredMutator()
        ranker = PatchRanker(patcher, verifier, mutator)
        code = "def run(user):\n    return eval(user)"
        candidates = ranker.generate_candidates(code, ["py_eval_user_input"])
        self.assertGreaterEqual(len(candidates), 1)


class TestDatasetLoader(unittest.TestCase):
    def test_bootstrap_samples(self):
        with tempfile.TemporaryDirectory() as td:
            loader = OpenSourceDatasetLoader(Path(td))
            samples = loader.build_bootstrap_samples()
            split = loader.split(samples)
            self.assertGreaterEqual(len(split["train"]), 1)
            self.assertIn(1, loader.class_balance(samples))


class TestAnalyzer(unittest.TestCase):
    def test_detects_python_and_c_patterns(self):
        analyzer = Analyzer()
        findings = analyzer.detect_static("def run(x):\n return eval(x)\nstrcpy(buf, src)")
        sigs = [f.signature for f in findings]
        self.assertIn("py_eval_user_input", sigs)
        self.assertIn("c_buffer_overflow", sigs)


class TestPipelineAndUI(unittest.TestCase):
    def test_train_export_load_and_ui_status(self):
        with tempfile.TemporaryDirectory() as td:
            pipeline = build_default_pipeline(Path(td))
            pipeline.train_bootstrap(epochs=1)
            model = Path(td) / "model_export.json"
            pipeline.export_model(model)
            self.assertTrue(model.exists())

            loaded = SparseExpertAutoSec.load_state(model)
            server, url = launch_ui(loaded, host="127.0.0.1", port=8899)
            try:
                status_raw = urllib.request.urlopen(f"{url}/api/status", timeout=2).read().decode("utf-8")
                status = json.loads(status_raw)
                self.assertTrue(status["ok"])
            finally:
                server.stop()


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
            config.policy.max_fuzz_cases = 16

            system = SparseExpertAutoSec(config=config)
            report = system.process_target(target)
            patched_code = target.read_text()

            self.assertIn("py_eval_user_input", report.findings)
            self.assertTrue(report.patched)
            self.assertIn("safe_eval", patched_code)
            self.assertIn("process_complete", report.telemetry)
            self.assertGreaterEqual(len(report.selected_experts), 2)
            self.assertIn("patch_rank", report.plan_steps)


if __name__ == "__main__":
    unittest.main()
