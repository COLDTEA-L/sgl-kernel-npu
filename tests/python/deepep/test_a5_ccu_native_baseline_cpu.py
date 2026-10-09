"""Baseline runner/analyzer regressions that do not load CANN or NPU."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("native_analysis", ROOT / "scripts/analyze_a5_ccu_native_alltoall.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def result(rank, k=4):
    return dict(implementation="native_hccl", api="dist.all_to_all_single",
        rank=rank, ranks=k, correctness="PASS", bytes_per_peer=4194304,
        tensor_bytes=k * 4194304, warmup=100, iterations=2,
        host_call_us=[10.0 + rank, 12.0 + rank], host_avg_us=11.0 + rank)


class NativeBaselineTest(unittest.TestCase):
    def seed(self, root, k=4, repeats=1):
        (root / "cases").mkdir()
        (root / "run_settings.json").write_text(json.dumps(dict(
            devices=list(range(k)), bytes_per_peer=4194304, warmup=100,
            iterations=2, repeats=repeats)))
        directory = root / "r1/results"; directory.mkdir(parents=True)
        for rank in range(k):
            (directory / f"rank{rank}.json").write_text(json.dumps(result(rank, k)))
        (root / "cases/native_alltoall_r1.status").write_text("0\n")
        # Deliberately unparseable interleaved stdout must not hide rank files.
        (root / "cases/native_alltoall_r1.log").write_text("RESULT_JSON {RESULT_JSON {broken\n")

    def test_two_and_four_rank_results(self):
        for k in (2, 4):
            with self.subTest(k=k), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); self.seed(root, k)
                summary = analysis.analyze(root)
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["tensor_bytes"], k * 4194304)
                self.assertEqual(summary["network_send_bytes"], (k - 1) * 4194304)
                self.assertEqual(summary["median_slowest_rank_host_us"], 11 + k - 1)

    def test_cleanup_failure_keeps_correctness_but_rejects_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.seed(root)
            (root / "cases/native_alltoall_r1.status").write_text("137\n")
            summary = analysis.analyze(root)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["cases"][0]["correctness"], "PASS")
            self.assertIsNone(summary["median_slowest_rank_host_us"])

    def test_incomplete_repeat_and_corrupted_rank_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.seed(root, repeats=3)
            bad = result(3); bad["host_avg_us"] = float("nan")
            (root / "r1/results/rank3.json").write_text(json.dumps(bad))
            summary = analysis.analyze(root)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["cases"][0]["invalid_or_missing_ranks"], "3")
            self.assertEqual(summary["cases"][1]["status"], "INTERRUPTED_OR_NOT_RUN")

    def test_help_and_import_do_not_require_runtime(self):
        script = Path(__file__).with_name("test_a5_ccu_native_alltoall.py")
        process = subprocess.run([sys.executable, str(script), "--help"],
            text=True, capture_output=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--bytes", process.stdout)
        self.assertNotIn("--manifest", process.stdout)
        process = subprocess.run([sys.executable, "-c",
            "import sys; import test_a5_ccu_native_alltoall; "
            "assert 'torch' not in sys.modules; assert 'deep_ep' not in sys.modules"],
            cwd=script.parent, text=True, capture_output=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_native_api_and_independent_calls(self):
        script = Path(__file__).with_name("test_a5_ccu_native_alltoall.py")
        tree = ast.parse(script.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        launch = next(node for node in ast.walk(main) if isinstance(node, ast.FunctionDef) and node.name == "launch")
        self.assertEqual(len(launch.body), 1)
        self.assertEqual(ast.unparse(launch.body[0]), "dist.all_to_all_single(recv, send)")
        imports = [alias.name for node in ast.walk(main) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names]
        self.assertFalse(any("deep_ep" in name for name in imports))
        loops = [node for node in ast.walk(main) if isinstance(node, ast.For)
                 and ast.unparse(node.iter) in ("range(args.warmup)", "range(args.iters)")]
        self.assertEqual(len(loops), 2)
        for loop in loops:
            calls = [ast.unparse(node.value) for node in loop.body if isinstance(node, ast.Expr)]
            self.assertLess(calls.index("launch()"), calls.index("torch.npu.synchronize()"))


if __name__ == "__main__":
    unittest.main()
