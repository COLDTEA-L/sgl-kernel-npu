"""No-card regression tests for worker entry-point isolation and barriers."""
import concurrent.futures
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from a5_ccu_peer_test_support import file_barrier, measure_calls, warmup_calls, validate_queued_calls


class PeerEntrypointTest(unittest.TestCase):
    def test_worker_preloads_before_torch_and_prepares_capture_stream_once(self):
        tree = ast.parse(Path(__file__).with_name("test_a5_ccu_peer_plan_alltoall.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        preload = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "prepare_runtime")
        torch_import = next(node for node in ast.walk(main) if isinstance(node, ast.Import) and
                            any(alias.name == "torch" for alias in node.names))
        self.assertLess(preload.lineno, torch_import.lineno)
        prepare = [node for node in calls if isinstance(node.func, ast.Attribute) and
                   node.func.attr == "prepare_ccu_urma_peer_plan"]
        self.assertEqual(len(prepare), 1)
        stream = next(node for node in main.body if isinstance(node, ast.Assign) and
                      any(isinstance(target, ast.Name) and target.id == "capture_stream" for target in node.targets))
        self.assertLess(stream.lineno, prepare[0].lineno)
        self.assertTrue(any(isinstance(node, ast.With) and
                           any(child is prepare[0] for child in ast.walk(node)) for node in main.body))

    def test_import_has_no_runtime_or_legacy_benchmark_side_effects(self):
        code = (
            "import sys; import test_a5_ccu_peer_plan_alltoall; "
            "assert 'test_a5_ccu_urma_multiroute_all2all' not in sys.modules; "
            "assert 'torch' not in sys.modules; "
            "assert 'torch_npu' not in sys.modules; print('isolated')"
        )
        result = subprocess.run([sys.executable, "-c", code],
            cwd=Path(__file__).parent, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "isolated")

    def test_peer_cli_is_not_replaced_by_legacy_cli(self):
        env = os.environ.copy()
        env.pop("_A5_CCU_A2A_REEXEC", None)
        result = subprocess.run([sys.executable,
            str(Path(__file__).with_name("test_a5_ccu_peer_plan_alltoall.py")), "--help"],
            env=env, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--manifest", "--available-cards", "--run-dir", "--graph-backend", "--sync-mode"):
            self.assertIn(option, result.stdout)
        self.assertNotIn("--implementation", result.stdout)
        self.assertNotIn("CASE_ROUTE_RUNTIME_ABI", result.stdout)

    def test_batch_rejects_graph_before_loading_runtime(self):
        result = subprocess.run([sys.executable,
            str(Path(__file__).with_name("test_a5_ccu_peer_plan_alltoall.py")),
            "--sync-mode", "batch", "--graph-backend", "aclgraph"],
            text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("batch validation currently requires", result.stderr)
        self.assertNotIn("runtime_bootstrap", result.stdout)

    def test_batch_validation_checks_every_snapshot_after_one_sync(self):
        events = []
        validate_queued_calls(lambda i: events.append(("prepare", i)),
            lambda: events.append(("launch", None)), lambda i: events.append(("snapshot", i)),
            lambda: events.append(("sync", None)), lambda i: events.append(("check", i)), 3)
        self.assertEqual(events, [(kind, index if kind != "launch" else None)
            for index in range(3) for kind in ("prepare", "launch", "snapshot")]
            + [("sync", None)] + [("check", i) for i in range(3)])
        def reject(index):
            raise AssertionError(f"bad snapshot {index}")
        with self.assertRaises(AssertionError):
            validate_queued_calls(lambda i: None, lambda: None, lambda i: None,
                                  lambda: None, reject, 3)

    def test_measurement_modes_have_explicit_sync_boundaries(self):
        for mode in ("batch", "per-call"):
            events = []
            launch = lambda: events.append("launch")
            sync = lambda: events.append("sync")
            warmup_calls(launch, sync, 2, mode)
            self.assertEqual(events, ["launch"] * 2 + ["sync"] if mode == "batch"
                             else ["launch", "sync"] * 2 + ["sync"])
            events.clear()
            ticks = iter([0, 120000] if mode == "batch" else [0, 50000, 60000, 130000])
            timing = measure_calls(launch, sync, 2, mode, lambda: next(ticks))
            self.assertEqual(timing["host_avg_us"], 60)
            self.assertEqual(events, ["launch", "launch", "sync"] if mode == "batch"
                             else ["launch", "sync"] * 2)
            self.assertEqual(timing["host_call_us"], [] if mode == "batch" else [50, 70])

    def test_preflight_precedes_warmup_and_profiling(self):
        source = Path(__file__).with_name("test_a5_ccu_peer_plan_alltoall.py").read_text()
        self.assertLess(source.index("validate_queued_calls(prepare"), source.index('phase("warmup")'))
        self.assertLess(source.index('barrier("warmup_done")'), source.index("profiler.start()"))
        self.assertLess(source.index("profiler.start()"), source.index("timing = measure_calls("))
        self.assertLess(source.index("timing = measure_calls("), source.index("profiler.stop()"))
        self.assertLess(source.index("write_rank_result(args.run_dir"), source.index('phase("final_synchronize_begin")'))

    def test_four_rank_barrier_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(file_barrier, directory, "ready", rank, 4, 2)
                           for rank in range(4)]
                for future in futures:
                    future.result(timeout=3)
            with self.assertRaises(TimeoutError):
                file_barrier(directory, "missing_rank", 0, 2, timeout=0)


if __name__ == "__main__":
    unittest.main()
