#!/usr/bin/env python3
"""Two-rank native-HCCL or CCU+URMA multi-route AllToAll benchmark."""

import argparse
import ctypes
import faulthandler
import json
import os
import re
import site
import sys
import time
from pathlib import Path

os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "CCU_SCHED")
CURRENT_PHASE = "module_import"


def mark_phase(name):
    global CURRENT_PHASE
    CURRENT_PHASE = name
    print(
        f"CASE_PHASE pid={os.getpid()} rank={os.environ.get('RANK', 'NA')} phase={name}",
        flush=True,
    )


def start_phase_watchdog():
    text = os.environ.get("A5_CCU_PHASE_WATCHDOG_SECONDS", "0")
    try:
        seconds = int(text)
    except ValueError as error:
        raise RuntimeError(f"invalid A5_CCU_PHASE_WATCHDOG_SECONDS={text!r}") from error
    faulthandler.enable(all_threads=True)
    if seconds > 0:
        faulthandler.dump_traceback_later(seconds, repeat=True)


start_phase_watchdog()


def sanitize(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def prepend_env_path(name, path):
    items = [item for item in os.environ.get(name, "").split(":") if item]
    os.environ[name] = ":".join([path, *(item for item in items if item != path)])


def prepare_runtime(require_route_library=True):
    root = Path(__file__).resolve().parents[3]
    packages = [root / "python" / "deep_ep" / "deep_ep"]
    packages.extend(Path(path) / "deep_ep" for path in site.getsitepackages())
    candidates = []
    if os.environ.get("A5_CCU_ROUTE_PROBE_LIB"):
        candidates.append(Path(os.environ["A5_CCU_ROUTE_PROBE_LIB"]))
    if os.environ.get("ASCEND_HOME_PATH"):
        candidates.append(Path(os.environ["ASCEND_HOME_PATH"]) /
                          "opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so")
    candidates.append(Path("/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/"
                           "liba5_ccu_urma_route_probe.so"))
    route_lib = next((path.resolve() for path in candidates if path.is_file()), None)
    if require_route_library and route_lib is None:
        raise RuntimeError("route library not found; rebuild and install the latest probe package")

    for package in packages:
        extensions = sorted(package.glob("deep_ep_cpp*.so"))
        if not extensions:
            continue
        changed = False
        if route_lib is not None:
            changed = os.environ.get("A5_CCU_ROUTE_PROBE_LIB") != str(route_lib)
            os.environ["A5_CCU_ROUTE_PROBE_LIB"] = str(route_lib)
        old_ld = os.environ.get("LD_LIBRARY_PATH", "")
        vendor_root = package / "vendors" / "hwcomputing"
        vendor_lib = vendor_root / "op_api" / "lib"
        if vendor_root.is_dir():
            old_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
            prepend_env_path("ASCEND_CUSTOM_OPP_PATH", str(vendor_root))
            changed |= old_opp != os.environ["ASCEND_CUSTOM_OPP_PATH"]
        if vendor_lib.is_dir():
            prepend_env_path("LD_LIBRARY_PATH", str(vendor_lib))
        if route_lib is not None:
            prepend_env_path("LD_LIBRARY_PATH", str(route_lib.parent))
        changed |= old_ld != os.environ["LD_LIBRARY_PATH"]
        if changed and os.environ.get("_A5_CCU_A2A_REEXEC") != "1":
            env = os.environ.copy()
            env["_A5_CCU_A2A_REEXEC"] = "1"
            os.execvpe(sys.executable,
                       [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
        if route_lib is not None:
            ctypes.CDLL(str(route_lib), mode=ctypes.RTLD_GLOBAL)
        sys.path.insert(0, str(package.parent))
        return extensions[0], route_lib
    raise RuntimeError("deep_ep_cpp not found; rebuild and install the latest DeepEP wheel")


def requested_implementation():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--implementation", choices=("native", "multiroute", "explicit", "standard"),
                        default="multiroute")
    return parser.parse_known_args()[0].implementation


mark_phase("requested_implementation")
IMPLEMENTATION = requested_implementation()
EXTENSION = None
ROUTE_LIB = None
if IMPLEMENTATION != "native":
    mark_phase("prepare_runtime")
    EXTENSION, ROUTE_LIB = prepare_runtime(IMPLEMENTATION != "standard")
    mark_phase(f"prepare_runtime_done_extension_{sanitize(EXTENSION)}")

mark_phase("import_torch")
import torch
mark_phase("import_torch_distributed")
import torch.distributed as dist
mark_phase("import_torch_npu")
import torch_npu
mark_phase("import_torch_npu_done")
if IMPLEMENTATION != "native":
    mark_phase("import_deep_ep")
    import deep_ep
    mark_phase("import_deep_ep_done")


def file_barrier(directory, tag, rank, world_size, timeout=180):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{tag}.rank{rank}").touch()
    deadline = time.monotonic() + timeout
    expected = [directory / f"{tag}.rank{item}" for item in range(world_size)]
    while not all(path.exists() for path in expected):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"file barrier {tag} timed out")
        time.sleep(0.01)


def make_profiler(output_dir, rank):
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        raise RuntimeError(f"profiling directory is not writable: {output_dir}")
    export_types = [torch_npu.profiler.ExportType.Text]
    if hasattr(torch_npu.profiler.ExportType, "Db"):
        export_types.append(torch_npu.profiler.ExportType.Db)
    kwargs = dict(
        export_type=export_types,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
    )
    try:
        config = torch_npu.profiler._ExperimentalConfig(mstx=False, **kwargs)
    except TypeError:
        config = torch_npu.profiler._ExperimentalConfig(msprof_tx=False, **kwargs)
    return torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(output_dir), worker_name=f"rank{rank}"),
        record_shapes=True,
        experimental_config=config,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("native", "multiroute", "explicit", "standard"),
                        default="multiroute")
    parser.add_argument("--bytes", type=int, default=2 * 1024 * 1024,
                        help="bytes in each destination slice")
    parser.add_argument("--route-index", type=int, default=0)
    parser.add_argument("--route-indices", default="")
    parser.add_argument("--path-uids", default="",
                        help="comma-separated PATH_CATALOG path_uid values; preferred over route ordinals")
    parser.add_argument("--path-weights", default="",
                        help="comma-separated positive weights, one per selected path")
    parser.add_argument("--synthetic-route-manifest", default="",
                        help="TSV of explicitly resolved relay EID pairs")
    parser.add_argument("--direct-route", type=int, default=0,
                        help="HCCL-discovered direct candidate prepended in explicit mode")
    parser.add_argument("--relay-manifest", default="",
                        help="ordered explicit relay TSV for the formal direct+relay operator")
    parser.add_argument("--plan-id", default="a5-explicit-multipath",
                        help="stable control-plane key used by the standard graph operator")
    parser.add_argument("--schedule", choices=("concurrent", "serial"),
                        default="concurrent")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iters", type=int, default=20)
    parser.add_argument("--profile-root", default="/home/l00934901/profiling")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.bytes <= 0 or args.bytes % 4:
        parser.error("--bytes must be a positive multiple of sizeof(float)")
    if args.warmup < 0 or args.iters <= 0 or args.profile_iters <= 0:
        parser.error("invalid warmup/iters/profile-iters")
    if args.implementation == "native" and args.schedule != "concurrent":
        parser.error("--schedule applies only to --implementation multiroute")
    if args.implementation in ("explicit", "standard"):
        if not args.relay_manifest:
            parser.error(f"--implementation {args.implementation} requires --relay-manifest")
        if args.schedule != "concurrent":
            parser.error(
                "the formal explicit-path operator is concurrent-only; "
                "use --implementation multiroute for serialized control experiments"
            )
        if not args.path_weights:
            parser.error(f"--implementation {args.implementation} requires direct+relay --path-weights")
        weights = [item for item in args.path_weights.split(",") if item]
        if len(weights) < 2 or len(weights) > 8 or any(
                not item.isdigit() or int(item) <= 0 for item in weights):
            parser.error("explicit --path-weights must contain 2..8 positive integers")
    return args


def select_routes(args):
    if args.synthetic_route_manifest:
        manifest = Path(args.synthetic_route_manifest).resolve()
        if not manifest.is_file():
            raise RuntimeError(f"synthetic route manifest not found: {manifest}")
        os.environ["A5_CCU_SYNTHETIC_ROUTE_MANIFEST"] = str(manifest)
        os.environ["A5_CCU_REBUILD_PUBLIC_FIELDS"] = "1"
        os.environ.pop("A5_CCU_SOURCE_ROUTE_MANIFEST", None)
        os.environ.pop("A5_CCU_SOURCE_ROUTE_PROVIDER", None)
        os.environ.pop("A5_CCU_SYNTHETIC_RANK0_LOCAL_EID", None)
        os.environ.pop("A5_CCU_SYNTHETIC_RANK0_REMOTE_EID", None)
    else:
        os.environ.pop("A5_CCU_SYNTHETIC_ROUTE_MANIFEST", None)
    if args.path_uids:
        os.environ["A5_CCU_PATH_UIDS"] = args.path_uids
        os.environ.pop("A5_CCU_ROUTE_INDICES", None)
        os.environ.pop("A5_CCU_ROUTE_INDEX", None)
    elif args.route_indices:
        os.environ.pop("A5_CCU_PATH_UIDS", None)
        os.environ["A5_CCU_ROUTE_INDICES"] = args.route_indices
        os.environ.pop("A5_CCU_ROUTE_INDEX", None)
    else:
        os.environ.pop("A5_CCU_PATH_UIDS", None)
        os.environ.pop("A5_CCU_ROUTE_INDICES", None)
        os.environ["A5_CCU_ROUTE_INDEX"] = str(args.route_index)
    os.environ["A5_CCU_ROUTE_SCHEDULE"] = args.schedule
    if args.path_weights:
        os.environ["A5_CCU_PATH_WEIGHTS"] = args.path_weights
    else:
        os.environ.pop("A5_CCU_PATH_WEIGHTS", None)


def main():
    mark_phase("parse_args")
    args = parse_args()
    if args.implementation == "multiroute":
        select_routes(args)
    elif args.implementation in ("explicit", "standard"):
        os.environ.pop("A5_CCU_ROUTE_SCHEDULE", None)
        for name in ("A5_CCU_SYNTHETIC_ROUTE_MANIFEST", "A5_CCU_PATH_UIDS",
                     "A5_CCU_ROUTE_INDICES", "A5_CCU_ROUTE_INDEX",
                     "A5_CCU_PATH_WEIGHTS"):
            os.environ.pop(name, None)
        if args.implementation == "standard":
            manifest = str(Path(args.relay_manifest).resolve())
            os.environ["A5_CCU_EXPLICIT_MULTIPATH_MANIFEST"] = manifest
            os.environ["A5_CCU_EXPLICIT_MULTIPATH_PLAN_ID"] = args.plan_id
            os.environ["A5_CCU_EXPLICIT_MULTIPATH_WEIGHTS"] = args.path_weights
    if args.debug:
        os.environ["A5_CCU_DEBUG"] = "1"
    else:
        os.environ.pop("A5_CCU_DEBUG", None)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("this test requires WORLD_SIZE=2")
    mark_phase("set_device")
    torch.npu.set_device(local_rank)
    mark_phase("init_process_group")
    dist.init_process_group("hccl")
    mark_phase("init_process_group_done")
    buffer = None
    if args.implementation != "native":
        mark_phase("deep_ep_buffer_init")
        buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
        mark_phase("deep_ep_buffer_init_done")

    explicit_weights = [int(item) for item in args.path_weights.split(",") if item]
    relay_manifest = str(Path(args.relay_manifest).resolve()) if args.relay_manifest else ""
    if relay_manifest and not Path(relay_manifest).is_file():
        raise RuntimeError(f"relay manifest not found: {relay_manifest}")

    elements = args.bytes // 4
    send = torch.stack([
        torch.full((elements,), float(rank * 100 + dst + 1),
                   dtype=torch.float32, device="npu")
        for dst in range(world_size)
    ])
    recv = torch.empty_like(send)
    expected = torch.stack([
        torch.full((elements,), float(src * 100 + rank + 1),
                   dtype=torch.float32, device="npu")
        for src in range(world_size)
    ])

    def run_op():
        nonlocal recv
        if args.implementation == "native":
            dist.all_to_all_single(recv, send)
        elif args.implementation == "standard":
            recv = buffer.explicit_multipath_all2all_ccu(
                send, args.plan_id, explicit_weights
            )
        elif args.implementation == "explicit":
            buffer.ccu_urma_explicit_multipath_alltoall_out(
                send, recv, relay_manifest, args.direct_route, explicit_weights
            )
        else:
            buffer.ccu_urma_multiroute_alltoall_out(send, recv)

    run_id = sanitize(
        f"{os.environ.get('TORCHELASTIC_RUN_ID', 'standalone')}_"
        f"{os.environ.get('MASTER_PORT', '0')}")
    case_tag = sanitize(
        "native" if args.implementation == "native" else
        (f"{args.implementation}_direct{args.direct_route}_{len(explicit_weights) - 1}relay_{args.schedule}"
         if args.implementation in ("explicit", "standard") else
         f"multiroute_{args.path_uids or args.route_indices or args.route_index}_{args.schedule}"))
    sync_dir = Path("/tmp") / f"a5_ccu_urma_a2a_{case_tag}_{run_id}"

    mark_phase("warmup")
    for _ in range(args.warmup):
        run_op()
    torch.npu.synchronize()
    mark_phase("warmup_correctness")
    torch.testing.assert_close(recv, expected)
    file_barrier(sync_dir, "warmup_done", rank, world_size)

    mark_phase("timing")
    torch.npu.synchronize()
    begin = time.perf_counter()
    for _ in range(args.iters):
        run_op()
    torch.npu.synchronize()
    average_us = (time.perf_counter() - begin) * 1e6 / args.iters
    torch.testing.assert_close(recv, expected)

    (sync_dir / f"average.rank{rank}.json").write_text(json.dumps(average_us))
    file_barrier(sync_dir, "timing_results", rank, world_size)
    averages = [json.loads((sync_dir / f"average.rank{item}.json").read_text())
                for item in range(world_size)]
    result_us = max(averages)

    if args.profile:
        mark_phase("profiling")
        profile_dir = (Path(args.profile_root) /
                       f"a5_ccu_alltoall_{case_tag}_{run_id}" / f"rank{rank}")
        profiler = make_profiler(profile_dir, rank)
        profiler.start()
        for _ in range(args.profile_iters):
            run_op()
        torch.npu.synchronize()
        profiler.stop()
        print(f"[rank={rank}] profiling={profile_dir.resolve()}", flush=True)
        torch.testing.assert_close(recv, expected)

    if rank == 0:
        result = {
            "implementation": args.implementation,
            "paths": ((f"direct{args.direct_route}+{len(explicit_weights) - 1}relay")
                      if args.implementation in ("explicit", "standard") else
                      (args.path_uids or args.route_indices or str(args.route_index)
                       if args.implementation == "multiroute" else "native")),
            "weights": args.path_weights or "1",
            "schedule": args.schedule if args.implementation != "native" else "native",
            "bytes_per_peer": args.bytes,
            "warmup": args.warmup,
            "iterations": args.iters,
            "host_batch_avg_us": result_us,
            "synthetic_route_manifest": args.synthetic_route_manifest or "",
            "relay_manifest": relay_manifest,
        }
        print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        print(f"PASS: implementation={result['implementation']} paths={result['paths']} "
              f"schedule={result['schedule']} bytes_per_peer={args.bytes} "
              f"host_batch_avg_us={result_us:.3f}", flush=True)
        if args.implementation == "multiroute":
            print(f"deep_ep_cpp={EXTENSION}", flush=True)
            print(f"route_library={ROUTE_LIB}", flush=True)
    mark_phase("reported_barrier")
    file_barrier(sync_dir, "reported", rank, world_size)
    mark_phase("destroy_process_group")
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()
    mark_phase("complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(
            f"CASE_FAILURE rank={os.environ.get('RANK', 'NA')} "
            f"phase={CURRENT_PHASE} type={type(error).__name__} error={error}",
            flush=True,
        )
        raise
