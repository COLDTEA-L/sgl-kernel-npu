"""Shared two-/four-rank runtime bootstrap, with no import-time side effects."""
import ctypes
import importlib
import importlib.util
import os
from pathlib import Path
import site
import sys

_LOADED_LIBRARIES = []


def import_deep_ep(extension_path):
    """Load the selected binary once under both historical import names.

    Call after importing torch/torch_npu. Older wheels use a top-level
    ``deep_ep_cpp`` import even though the binary lives inside ``deep_ep``.
    Do not add the package directory to sys.path or load a second binary.
    Missing binary dependencies must remain visible, not trigger a fallback.
    """
    if "torch" not in sys.modules:
        raise RuntimeError("import torch before import_deep_ep")
    extension_path = Path(extension_path).resolve(strict=True)
    names = ("deep_ep_cpp", "deep_ep.deep_ep_cpp")
    loaded = [sys.modules[name] for name in names if name in sys.modules]
    package = sys.modules.get("deep_ep")
    if package is not None and Path(package.__file__).resolve().parent != extension_path.parent:
        raise RuntimeError("already imported deep_ep package differs from selected extension")
    if loaded and any(module is not loaded[0] for module in loaded):
        raise RuntimeError("deep_ep_cpp aliases refer to different loaded modules")
    if loaded:
        extension = loaded[0]
        if Path(extension.__file__).resolve() != extension_path:
            raise RuntimeError("already imported deep_ep_cpp differs from selected extension")
    else:
        spec = importlib.util.spec_from_file_location("deep_ep_cpp", extension_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load DeepEP extension: {extension_path}")
        extension = importlib.util.module_from_spec(spec)
    inserted = [name for name in names if name not in sys.modules]
    for name in inserted:
        sys.modules[name] = extension
    try:
        if not loaded:
            spec.loader.exec_module(extension)
        package = importlib.import_module("deep_ep")
        if Path(package.__file__).resolve().parent != extension_path.parent:
            raise RuntimeError("imported deep_ep package differs from selected extension")
        if getattr(package, "deep_ep_cpp", extension) is not extension:
            raise RuntimeError("deep_ep package refers to a different extension")
        package.deep_ep_cpp = extension
    except BaseException:
        for name in inserted:
            if sys.modules.get(name) is extension:
                del sys.modules[name]
        raise
    print(f"CASE_TEST_EXTENSION path={extension_path} "
          "aliases=deep_ep_cpp,deep_ep.deep_ep_cpp", flush=True)
    return package, extension


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
