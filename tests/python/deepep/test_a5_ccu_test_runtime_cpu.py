"""Runtime setup and measurement rendezvous regressions without NPU."""
import concurrent.futures
import ctypes
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock

from a5_ccu_peer_test_support import measurement_start
from a5_ccu_test_runtime import prepare_runtime


class RuntimeSetupTests(unittest.TestCase):
    def run_import_fixture(self, initializer, extension_code="Config = object()\n", setup="", check=""):
        # Exercise real Python import behavior in an isolated interpreter,
        # using a tiny Python stand-in for the NPU extension, not a fake .so.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "deep_ep"; package.mkdir()
            (package / "__init__.py").write_text(initializer)
            extension = package / "deep_ep_cpp.py"
            extension.write_text(extension_code)
            code = (
                "import sys, types, importlib\n"
                f"sys.path.insert(0, {str(root)!r})\n"
                "from a5_ccu_test_runtime import import_deep_ep\n"
                "sys.modules['torch'] = types.ModuleType('torch')\n"
                + setup + "\n"
                f"package, ext = import_deep_ep({str(extension)!r})\n"
                "assert sys.modules['deep_ep_cpp'] is ext\n"
                "assert sys.modules['deep_ep.deep_ep_cpp'] is ext\n"
                "assert importlib.import_module('deep_ep.deep_ep_cpp') is ext\n"
                "assert package.Config is ext.Config\n"
                + check + "\n"
            )
            return subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent,
                                  text=True, capture_output=True, timeout=10)

    def test_old_wheel_top_level_import_with_binary_inside_package(self):
        result = self.run_import_fixture("from deep_ep_cpp import Config\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CASE_TEST_EXTENSION", result.stdout)

    def test_current_wheel_relative_import_uses_the_same_module(self):
        result = self.run_import_fixture("from . import deep_ep_cpp\nConfig = deep_ep_cpp.Config\n",
            extension_code="import builtins\nbuiltins._deep_ep_loads = getattr(builtins, '_deep_ep_loads', 0) + 1\nConfig = object()\n",
            check="import builtins\nassert builtins._deep_ep_loads == 1\n"
                  "import_deep_ep(ext.__file__)\nassert builtins._deep_ep_loads == 1")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_conflicting_preloaded_extension_is_rejected(self):
        result = self.run_import_fixture("from deep_ep_cpp import Config\n",
            setup="wrong = types.ModuleType('deep_ep_cpp')\nwrong.__file__ = '/tmp/wrong_deep_ep_cpp.so'\n"
                  "sys.modules['deep_ep_cpp'] = wrong")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already imported deep_ep_cpp differs", result.stderr)

    def test_extension_dependency_error_is_not_hidden_by_an_import_fallback(self):
        result = self.run_import_fixture("from deep_ep_cpp import Config\n",
            extension_code="raise ImportError('lib_required_dependency.so missing')\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("lib_required_dependency.so missing", result.stderr)
        self.assertNotIn("No module named 'deep_ep_cpp'", result.stderr)

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
