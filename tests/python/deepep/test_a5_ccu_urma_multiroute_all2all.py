#!/usr/bin/env python3
"""Two-rank HCCL-style CCU+URMA multi-route AllToAll benchmark."""

import argparse
import ctypes
import json
import os
import re
import site
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "CCU_SCHED")


def prepend_env_path(name, path):
    items = [item for item in os.environ.get(name, "").split(":") if item]
    os.environ[name] = ":".join([path, *(item for item in items if item != path)])


def prepare_runtime():
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
    if route_lib is None:
        raise RuntimeError("route library not found; rebuild and install the latest probe package")

    for package in packages:
        extensions = sorted(package.glob("deep_ep_cpp*.so"))
        if not extensions:
            continue
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
        prepend_env_path("LD_LIBRARY_PATH", str(route_lib.parent))
        changed |= old_ld != os.environ["LD_LIBRARY_PATH"]
        if changed and os.environ.get("_A5_CCU_A2A_REEXEC") != "1":
            env = os.environ.copy()
            env["_A5_CCU_A2A_REEXEC"] = "1"
            os.execvpe(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
        ctypes.CDLL(str(route_lib), mode=ctypes.RTLD_GLOBAL)
        sys.path.insert(0, str(package.parent))
        return extensions[0], route_lib
    raise RuntimeError("deep_ep_cpp not found; rebuild and install the latest DeepEP wheel")


EXTENSION, ROUTE_LIB = prepare_runtime()

import torch
import torch.distributed as dist
import torch_npu
import deep_ep


def file_barrier(directory, tag, rank, world_size, timeout=180):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{tag}.rank{rank}").touch()
    deadline = time.monotonic() + timeout
    expected = [directory / f"{tag}.rank{item}" for item in range(world_size)]
    while not all(path.exists() for path in expected):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"file barrier {tag} timed out")
        time.sleep(0.01)


def make_profiler(output_dir, iterations):
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
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=iterations, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output_dir)),
        record_shapes=True,
        experimental_config=config,
    )


def select_routes(route_indices, route_index, schedule):
    if route_indices:
        os.environ["A5_CCU_ROUTE_INDICES"] = route_indices
        os.environ.pop("A5_CCU_ROUTE_INDEX", None)
    else:
        os.environ.pop("A5_CCU_ROUTE_INDICES", None)
        os.environ["A5_CCU_ROUTE_INDEX"] = str(route_index)
    os.environ["A5_CCU_ROUTE_SCHEDULE"] = schedule


def benchmark_case(buffer, sync_dir, rank, world_size, label, routes, schedule,
                   bytes_per_peer, warmup, iterations):
    select_routes(routes, 0, schedule)
    elements = bytes_per_peer // 4
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
    for _ in range(warmup):
        buffer.ccu_urma_multiroute_alltoall_out(send, recv)
    torch.npu.synchronize()
    torch.testing.assert_close(recv, expected)
    file_barrier(sync_dir, f"{label}_warmup", rank, world_size)

    begin = time.perf_counter()
    for _ in range(iterations):
        buffer.ccu_urma_multiroute_alltoall_out(send, recv)
    torch.npu.synchronize()
    average_us = (time.perf_counter() - begin) * 1e6 / iterations
    torch.testing.assert_close(recv, expected)

    result_file = sync_dir / f"{label}.rank{rank}.json"
    result_file.write_text(json.dumps(average_us))
    file_barrier(sync_dir, f"{label}_results", rank, world_size)
    rank_averages = [json.loads((sync_dir / f"{label}.rank{item}.json").read_text())
                     for item in range(world_size)]
    result = max(rank_averages)
    if rank == 0:
        print(f"CASE label={label} routes={routes} schedule={schedule} "
              f"bytes_per_peer={bytes_per_peer} host_batch_avg_us={result:.3f}", flush=True)
    file_barrier(sync_dir, f"{label}_reported", rank, world_size)
    return result


def verify_concurrency(buffer, sync_dir, rank, world_size, args):
    # Match the C++ splitter exactly: route0 gets floor(2B/3) aligned to
    # 256 bytes and route2 receives the remainder.
    route0_bytes = (args.bytes * 2 // 3) // 256 * 256
    route2_bytes = args.bytes - route0_bytes
    if route0_bytes <= 0 or route2_bytes <= 0:
        raise ValueError("--bytes is too small for the aligned 2:1 split")

    route0_us = benchmark_case(
        buffer, sync_dir, rank, world_size, "route0_share", "0", "concurrent",
        route0_bytes, args.warmup, args.iters)
    route2_us = benchmark_case(
        buffer, sync_dir, rank, world_size, "route2_share", "2", "concurrent",
        route2_bytes, args.warmup, args.iters)

    serial_samples = []
    concurrent_samples = []
    for round_index in range(args.verification_rounds):
        # Alternate order to reduce thermal/load/order bias. Both kernels were
        # registered together and share the exact same route resources.
        schedules = ("serial", "concurrent") if round_index % 2 == 0 else (
            "concurrent", "serial")
        for schedule in schedules:
            value = benchmark_case(
                buffer, sync_dir, rank, world_size,
                f"combined_{schedule}_round{round_index}", "0,2", schedule,
                args.bytes, args.warmup, args.iters)
            (serial_samples if schedule == "serial" else concurrent_samples).append(value)

    serial_us = statistics.median(serial_samples)
    concurrent_us = statistics.median(concurrent_samples)
    shorter_isolated_us = min(route0_us, route2_us)
    overlap_ratio = ((serial_us - concurrent_us) / shorter_isolated_us
                     if shorter_isolated_us > 0 else float("nan"))
    speedup = serial_us / concurrent_us if concurrent_us > 0 else float("inf")
    if overlap_ratio >= 0.50:
        evidence = "strong"
    elif overlap_ratio >= 0.15:
        evidence = "partial"
    elif overlap_ratio <= 0.05:
        evidence = "none"
    else:
        evidence = "inconclusive"

    if rank == 0:
        print("CONCURRENCY_RESULT "
              f"total_bytes={args.bytes} route0_bytes={route0_bytes} "
              f"route2_bytes={route2_bytes} route0_share_us={route0_us:.3f} "
              f"route2_share_us={route2_us:.3f} serial_median_us={serial_us:.3f} "
              f"concurrent_median_us={concurrent_us:.3f} speedup={speedup:.4f} "
              f"overlap_ratio={overlap_ratio:.4f} evidence={evidence}", flush=True)
        print("Interpretation: overlap_ratio=(serial-concurrent)/min(route0_share,route2_share); "
              "forced serial is the primary control, while port counters are still required "
              "to prove physical-link overlap.", flush=True)
    if (args.require_overlap_ratio is not None and
            overlap_ratio < args.require_overlap_ratio):
        raise AssertionError(
            f"overlap_ratio {overlap_ratio:.4f} is below required "
            f"{args.require_overlap_ratio:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=2 * 1024 * 1024,
                        help="bytes in each destination slice")
    parser.add_argument("--route-index", type=int, default=0)
    parser.add_argument("--route-indices", default="")
    parser.add_argument("--schedule", choices=("concurrent", "serial"),
                        default="concurrent")
    parser.add_argument("--verify-concurrency", action="store_true",
                        help="compare route0/route2 concurrent execution with a forced-serial kernel")
    parser.add_argument("--verification-rounds", type=int, default=3)
    parser.add_argument("--require-overlap-ratio", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-root", default="/home/l00934901/profiling")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.bytes <= 0 or args.bytes % 4:
        parser.error("--bytes must be a positive multiple of sizeof(float)")
    if args.warmup < 0 or args.iters <= 0:
        parser.error("invalid warmup/iters")
    if args.verification_rounds <= 0:
        parser.error("--verification-rounds must be positive")
    if args.verify_concurrency and args.profile:
        parser.error("run --verify-concurrency without --profile; profile individual modes separately")

    select_routes(args.route_indices, args.route_index, args.schedule)
    if args.debug:
        os.environ["A5_CCU_DEBUG"] = "1"
    else:
        os.environ.pop("A5_CCU_DEBUG", None)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        parser.error("this test requires WORLD_SIZE=2")
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)

    run_id = re.sub(r"[^A-Za-z0-9_.-]", "_",
                    f"{os.environ.get('TORCHELASTIC_RUN_ID', 'standalone')}_"
                    f"{os.environ.get('MASTER_PORT', '0')}")
    sync_dir = Path("/tmp") / f"a5_ccu_urma_a2a_sync_{run_id}"
    if args.verify_concurrency:
        verify_concurrency(buffer, sync_dir, rank, world_size, args)
        file_barrier(sync_dir, "verification_done", rank, world_size)
        dist.destroy_process_group()
        return

    elements = args.bytes // 4
    send = torch.stack([
        torch.full((elements,), float(rank * 100 + dst + 1),
                   dtype=torch.float32, device="npu")
        for dst in range(world_size)
    ])
    recv = torch.empty_like(send)

    for _ in range(args.warmup):
        buffer.ccu_urma_multiroute_alltoall_out(send, recv)
    torch.npu.synchronize()
    expected = torch.stack([
        torch.full((elements,), float(src * 100 + rank + 1),
                   dtype=torch.float32, device="npu")
        for src in range(world_size)
    ])
    torch.testing.assert_close(recv, expected)
    file_barrier(sync_dir, "warmup_done", rank, world_size)

    profiler = None
    if args.profile:
        routes = args.route_indices or str(args.route_index)
        profile_dir = (Path(args.profile_root) /
                       f"a5_ccu_urma_all2all_routes_{routes}_{run_id}" / f"rank{rank}")
        profile_dir.mkdir(parents=True, exist_ok=True)
        profiler = make_profiler(profile_dir, args.iters)
        profiler.start()
        print(f"[rank={rank}] profiling={profile_dir}", flush=True)

    torch.npu.synchronize()
    begin = time.perf_counter()
    for _ in range(args.iters):
        buffer.ccu_urma_multiroute_alltoall_out(send, recv)
        if profiler is not None:
            profiler.step()
    torch.npu.synchronize()
    average_us = (time.perf_counter() - begin) * 1e6 / args.iters
    if profiler is not None:
        profiler.stop()
    torch.testing.assert_close(recv, expected)

    (sync_dir / f"average.rank{rank}.json").write_text(json.dumps(average_us))
    file_barrier(sync_dir, "results", rank, world_size)
    if rank == 0:
        averages = [json.loads((sync_dir / f"average.rank{item}.json").read_text())
                    for item in range(world_size)]
        routes = args.route_indices or str(args.route_index)
        print(f"PASS: CCU+URMA two-rank AllToAll routes={routes} "
              f"schedule={args.schedule} "
              f"bytes_per_peer={args.bytes} warmup={args.warmup} iterations={args.iters} "
              f"host_batch_avg_us={max(averages):.3f}", flush=True)
        print(f"deep_ep_cpp={EXTENSION}", flush=True)
        print(f"route_library={ROUTE_LIB}", flush=True)
    file_barrier(sync_dir, "reported", rank, world_size)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
