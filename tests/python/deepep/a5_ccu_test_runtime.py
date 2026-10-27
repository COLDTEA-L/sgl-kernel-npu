"""Shared CCU test runtime bootstrap, with no import-time side effects."""
import ctypes
import importlib
import importlib.util
import json
import os
from pathlib import Path
import site
import sys

_LOADED_LIBRARIES = []


def prepare_ccu_environment():
    """Match the historical CCU workers before loading any native runtime."""
    mode = os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "CCU_SCHED")
    if mode != "CCU_SCHED":
        raise RuntimeError(f"CCU worker requires HCCL_OP_EXPANSION_MODE=CCU_SCHED, got {mode!r}")
    os.environ.setdefault("HCCL_BUFFSIZE", "2300")
    pg_mode = os.environ.get("A5_CCU_PG_MODE", "env")
    if pg_mode not in ("env", "config6"):
        raise RuntimeError(f"invalid A5_CCU_PG_MODE={pg_mode!r}; expected env or config6")


def log_ccu_environment(stage):
    fields = {name: os.environ.get(name) for name in (
        "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE", "HCCL_ALGO", "A5_CCU_PG_MODE",
        "ASCEND_HOME_PATH", "ASCEND_RT_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "LD_PRELOAD")}
    fields.update(stage=stage, rank=os.environ.get("RANK", "NA"), pid=os.getpid(),
                  python=sys.executable, torch_loaded="torch" in sys.modules,
                  torch_npu_loaded="torch_npu" in sys.modules)
    print("CCU_RUNTIME_ENV " + json.dumps(fields, sort_keys=True), flush=True)


def ccu_pg_kwargs(torch_npu):
    """Default to the old branches' environment-only communicator creation.

    config6 is an explicit comparison, not an automatic retry. Public SDK
    HcclCommConfig uses 6 for CCU_SCHED; T560 behavior still needs hardware
    validation. Do not require a newer SDK query enum to gate old runtimes.
    """
    log_ccu_environment("before_process_group")
    prepare_ccu_environment()
    mode = os.environ.get("A5_CCU_PG_MODE", "env")
    print(f"CCU_PG_CONFIG rank={os.environ.get('RANK', 'NA')} mode={mode} "
          f"pg_options={'omitted' if mode == 'env' else 'hccl_op_expansion_mode:6'} "
          f"torch_npu={getattr(torch_npu, '__version__', 'unknown')}", flush=True)
    if mode == "env":
        return {}
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {"hccl_op_expansion_mode": 6}
    return {"pg_options": options}


def log_ccu_communicator(group_name):
    """Record raw query evidence; ChannelAcquire remains the capability gate.

    T560 returned success/raw=0 after config6 in an actual run. We have not
    established its query implementation/enum matches the public 9.1 SDK.
    A diagnostic query must not turn this into an artificial test failure.
    """
    log_ccu_environment("before_channel_prepare")
    try:
        mappings = Path("/proc/self/maps").read_text().splitlines()
        paths = set()
        for line in mappings:
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and any(name in Path(fields[5]).name for name in (
                    "libhccl.so", "libhcomm.so", "liba5_ccu_urma_route_probe.so", "deep_ep_cpp")):
                paths.add(fields[5])
        print("CCU_RUNTIME_LIBS " + json.dumps(sorted(paths)), flush=True)
    except OSError as error:
        print(f"CCU_RUNTIME_LIBS unavailable={error}", flush=True)
    try:
        if not group_name:
            raise ValueError("unnamed communicator")
        hccl = ctypes.CDLL("libhccl.so")
        get_comm = hccl.HcomGetCommHandleByGroup
        get_comm.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        get_comm.restype = ctypes.c_int
        get_config = hccl.HcclConfigGetInfo
        get_config.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p]
        get_config.restype = ctypes.c_int
        comm = ctypes.c_void_p()
        status = get_comm(group_name.encode(), ctypes.byref(comm))
        if status != 0 or not comm.value:
            raise ValueError(f"HcomGetCommHandleByGroup status={status}, null={not comm.value}")
        mode = ctypes.c_int(-1)
        status = get_config(comm, 0, ctypes.sizeof(mode), ctypes.byref(mode))
        print(f"CCU_PG_MODE rank={os.environ.get('RANK', 'NA')} group={group_name} "
              f"query_status={status} raw_mode={mode.value} diagnostic_only=1", flush=True)
        return status, mode.value
    except (AttributeError, OSError, ValueError) as error:
        print(f"CCU_PG_MODE group={group_name!r} unavailable={error} diagnostic_only=1", flush=True)
        return None


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
    prepare_ccu_environment()
    log_ccu_environment("before_runtime_load")
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
