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

    def test_native_api_and_no_custom_runtime(self):
        script = Path(__file__).with_name("test_a5_ccu_native_alltoall.py")
        tree = ast.parse(script.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        launch = next(node for node in ast.walk(main) if isinstance(node, ast.FunctionDef) and node.name == "launch")
        self.assertEqual(len(launch.body), 1)
        self.assertEqual(ast.unparse(launch.body[0]), "dist.all_to_all_single(recv, send)")
        imports = [alias.name for node in ast.walk(main) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names]
        self.assertFalse(any("deep_ep" in name for name in imports))

    def test_batch_and_per_call_execution_order(self):
        import test_a5_ccu_native_alltoall as benchmark
        for mode in ("batch", "per-call"):
            events = []
            launch = lambda: events.append("launch")
            sync = lambda: events.append("sync")
            benchmark.warmup_calls(launch, sync, 3, mode)
            expected = ["launch"] * 3 + ["sync"] if mode == "batch" else ["launch", "sync"] * 3 + ["sync"]
            self.assertEqual(events, expected)
            events.clear()
            ticks = iter([0, 120000] if mode == "batch" else [0, 50000, 60000, 130000])
            timing = benchmark.measure_calls(launch, sync, 2, mode, lambda: next(ticks))
            expected = ["launch", "launch", "sync"] if mode == "batch" else ["launch", "sync"] * 2
            self.assertEqual(events, expected)
            self.assertEqual(timing["host_avg_us"], 60)
            self.assertEqual(timing["host_call_us"], [] if mode == "batch" else [50, 70])

    def test_batch_results_and_mode_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.seed(root)
            settings_path = root / "run_settings.json"
            settings = json.loads(settings_path.read_text()); settings["sync_mode"] = "batch"
            settings_path.write_text(json.dumps(settings))
            for rank in range(4):
                item = result(rank)
                item.update(sync_mode="batch", host_call_us=[], host_batch_total_us=2 * item["host_avg_us"])
                (root / f"r1/results/rank{rank}.json").write_text(json.dumps(item))
            self.assertTrue(analysis.analyze(root)["complete"])
            item["host_batch_total_us"] += 10
            self.assertFalse(analysis.valid_result(item, 3, settings))
            item["host_batch_total_us"] -= 10
            settings["sync_mode"] = "per-call"
            self.assertFalse(analysis.valid_result(item, 3, settings))

    def test_defaults_and_measured_only_profiler_window(self):
        script = Path(__file__).with_name("test_a5_ccu_native_alltoall.py")
        source = script.read_text()
        tree = ast.parse(source)
        defaults = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                for kw in node.keywords:
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                        defaults[node.args[0].value] = kw.value.value
        self.assertEqual(defaults["--warmup"], 100)
        self.assertEqual(defaults["--iters"], 20)
        self.assertEqual(defaults["--sync-mode"], "batch")
        runner = (ROOT / "scripts/run_a5_ccu_native_alltoall_baseline.sh").read_text()
        self.assertIn("\nwarmup=100\n", runner)
        self.assertIn("\niterations=20\n", runner)
        self.assertIn("\nrepeats=1\n", runner)
        self.assertIn("\nsync_mode=batch\n", runner)
        self.assertIn('--sync-mode "$sync_mode"', runner)
        self.assertLess(source.index('barrier("warmup_done")'), source.index("profiler.start()"))
        self.assertLess(source.index("profiler.start()"), source.index("timing = measure_calls("))
        self.assertLess(source.index("timing = measure_calls("), source.index("profiler.stop()"))


if __name__ == "__main__":
    unittest.main()
