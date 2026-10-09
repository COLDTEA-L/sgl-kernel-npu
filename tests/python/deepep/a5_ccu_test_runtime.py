"""Shared two-/four-rank runtime bootstrap, with no import-time side effects."""
import ctypes
import os
from pathlib import Path
import site
import sys

_LOADED_LIBRARIES = []


def prepend_env_path(name, path):
    items = [item for item in os.environ.get(name, "").split(":") if item]
    os.environ[name] = ":".join([str(path), *(item for item in items if item != str(path))])


def prepare_runtime(entrypoint, require_route_library=True):
    """Normalize loader paths, re-exec the CALLER if needed, then preload globally.

    Call before importing torch/torch_npu. Never re-exec this helper's __file__
    or import an executable benchmark to obtain its runtime setup.
    """
    entrypoint = Path(entrypoint).resolve()
    packages = [Path(path) / "deep_ep" for path in site.getsitepackages()]
    packages.append(entrypoint.parents[3] / "python/deep_ep/deep_ep")
    explicit = os.environ.get("A5_CCU_ROUTE_PROBE_LIB")
    if explicit and not Path(explicit).is_file():
        raise RuntimeError(f"configured route library does not exist: {explicit}")
    candidates = [Path(explicit)] if explicit else []
    if os.environ.get("ASCEND_HOME_PATH"):
        candidates.append(Path(os.environ["ASCEND_HOME_PATH"]) /
                          "opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so")
    candidates.append(Path("/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/"
                           "liba5_ccu_urma_route_probe.so"))
    route_lib = next((path.resolve() for path in candidates if path.is_file()), None)
    if require_route_library and route_lib is None:
        raise RuntimeError("route library not found; install the route package")
    for package in packages:
        extensions = sorted(package.glob("deep_ep_cpp*.so"))
        if not extensions:
            continue
        changed = False
        if route_lib is not None:
            changed = os.environ.get("A5_CCU_ROUTE_PROBE_LIB") != str(route_lib)
            os.environ["A5_CCU_ROUTE_PROBE_LIB"] = str(route_lib)
        old_ld = os.environ.get("LD_LIBRARY_PATH", "")
        vendor_root = package / "vendors/hwcomputing"
        if vendor_root.is_dir():
            old_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
            prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_root)
            changed |= old_opp != os.environ["ASCEND_CUSTOM_OPP_PATH"]
        if (vendor_root / "op_api/lib").is_dir():
            prepend_env_path("LD_LIBRARY_PATH", vendor_root / "op_api/lib")
        if route_lib is not None:
            prepend_env_path("LD_LIBRARY_PATH", route_lib.parent)
        changed |= old_ld != os.environ.get("LD_LIBRARY_PATH", "")
        if changed:
            marker = "_A5_CCU_TEST_RUNTIME_ENTRYPOINT"
            if os.environ.get(marker) == str(entrypoint):
                raise RuntimeError("runtime loader paths changed again after re-exec; stop")
            env = os.environ.copy()
            env[marker] = str(entrypoint)
            print(f"CASE_TEST_RUNTIME_REEXEC entrypoint={entrypoint}", flush=True)
            os.execvpe(sys.executable, [sys.executable, str(entrypoint), *sys.argv[1:]], env)
            raise RuntimeError("execvpe unexpectedly returned")
        if route_lib is not None:
            _LOADED_LIBRARIES.append(ctypes.CDLL(str(route_lib), mode=ctypes.RTLD_GLOBAL))
        sys.path.insert(0, str(package.parent))
        print(f"CASE_TEST_RUNTIME entrypoint={entrypoint} route={route_lib} "
              f"extension={extensions[0].resolve()} load=before_torch,RTLD_GLOBAL", flush=True)
        return extensions[0].resolve(), route_lib
    raise RuntimeError("deep_ep_cpp not found; install the DeepEP wheel")
