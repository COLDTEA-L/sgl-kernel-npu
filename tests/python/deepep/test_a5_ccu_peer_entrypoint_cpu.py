"""No-card regression tests for worker entry-point isolation and barriers."""
import concurrent.futures
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from a5_ccu_peer_test_support import file_barrier


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
        for option in ("--manifest", "--available-cards", "--run-dir", "--graph-backend"):
            self.assertIn(option, result.stdout)
        self.assertNotIn("--implementation", result.stdout)
        self.assertNotIn("CASE_ROUTE_RUNTIME_ABI", result.stdout)

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
