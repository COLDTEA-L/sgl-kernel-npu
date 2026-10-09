"""Runtime setup and measurement rendezvous regressions without NPU."""
import concurrent.futures
import ctypes
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from a5_ccu_peer_test_support import measurement_start
from a5_ccu_test_runtime import prepare_runtime


class RuntimeSetupTests(unittest.TestCase):
    def test_reexec_targets_caller_and_preserves_peer_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            package = root / "site/deep_ep"; package.mkdir(parents=True)
            (package / "deep_ep_cpp.fake.so").touch()
            route = root / "route/liba5_ccu_urma_route_probe.so"
            route.parent.mkdir(); route.touch()
            caller = root / "repo/tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py"
            with mock.patch("a5_ccu_test_runtime.site.getsitepackages", return_value=[str(package.parent)]), \
                 mock.patch.dict(os.environ, {"A5_CCU_ROUTE_PROBE_LIB": str(route), "LD_LIBRARY_PATH": ""}, clear=True), \
                 mock.patch.object(sys, "argv", [str(caller), "--manifest", "plan.tsv", "--graph-backend", "none"]), \
                 mock.patch("a5_ccu_test_runtime.os.execvpe", side_effect=RuntimeError("reexec requested")) as execute:
                with self.assertRaisesRegex(RuntimeError, "reexec requested"):
                    prepare_runtime(caller)
                command = execute.call_args[0][1]
                self.assertEqual(command[1], str(caller))
                self.assertEqual(command[2:], ["--manifest", "plan.tsv", "--graph-backend", "none"])
                self.assertNotIn("test_a5_ccu_urma_multiroute_all2all.py", command)

    def test_preload_is_global_and_identical_for_two_and_four_rank_callers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            package = root / "site/deep_ep"; package.mkdir(parents=True)
            extension = package / "deep_ep_cpp.fake.so"; extension.touch()
            route = root / "route/liba5_ccu_urma_route_probe.so"
            route.parent.mkdir(); route.touch()
            entry = root / "repo/tests/python/deepep"
            with mock.patch("a5_ccu_test_runtime.site.getsitepackages", return_value=[str(package.parent)]), \
                 mock.patch.dict(os.environ, {"A5_CCU_ROUTE_PROBE_LIB": str(route), "LD_LIBRARY_PATH": str(route.parent)}, clear=True), \
                 mock.patch.object(sys, "path", sys.path.copy()), \
                 mock.patch("a5_ccu_test_runtime.ctypes.CDLL") as load, \
                 mock.patch("a5_ccu_test_runtime.os.execvpe") as execute:
                for name in ("test_a5_ccu_urma_multiroute_all2all.py", "test_a5_ccu_peer_plan_alltoall.py"):
                    self.assertEqual(prepare_runtime(entry / name), (extension, route))
                self.assertEqual(load.call_args_list,
                    [mock.call(str(route), mode=ctypes.RTLD_GLOBAL)] * 2)
                execute.assert_not_called()

    def test_measurement_rendezvous_has_a_single_shared_start_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(measurement_start, tmp, rank, 4, 0.01) for rank in range(4)]
                for future in futures:
                    self.assertGreaterEqual(future.result(timeout=3), 0)
            self.assertGreater(int((Path(tmp) / "measurement_start_ns").read_text()), 0)


if __name__ == "__main__":
    unittest.main()
