#!/usr/bin/env python3
"""Two-rank CCU+URMA multi-route write with integrated NPU profiling."""

import argparse
import ctypes
import os
import re
import shutil
import site
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "CCU_SCHED")


def prepend_env_path(name, path):
    current = [item for item in os.environ.get(name, "").split(":") if item]
    os.environ[name] = ":".join([path, *(item for item in current if item != path)])


def prepare_runtime():
    root = Path(__file__).resolve().parents[3]
    packages = [root / "python" / "deep_ep" / "deep_ep"]
    packages.extend(Path(path) / "deep_ep" for path in site.getsitepackages())
    route_candidates = []
    configured = os.environ.get("A5_CCU_ROUTE_PROBE_LIB")
    if configured:
        route_candidates.append(Path(configured))
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        route_candidates.append(
            Path(ascend_home) / "opp" / "vendors" / "cust" / "lib64"
            / "liba5_ccu_urma_route_probe.so"
        )
    route_candidates.append(
        Path("/usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64")
        / "liba5_ccu_urma_route_probe.so"
    )
    route_lib = next((path.resolve() for path in route_candidates if path.is_file()), None)
    if route_lib is None:
        raise RuntimeError(
            "liba5_ccu_urma_route_probe.so not found; install the latest CCU route package "
            "or set A5_CCU_ROUTE_PROBE_LIB"
        )

    for package in packages:
        extensions = sorted(package.glob("deep_ep_cpp*.so"))
        if not extensions:
            continue
        env_changed = os.environ.get("A5_CCU_ROUTE_PROBE_LIB") != str(route_lib)
        os.environ["A5_CCU_ROUTE_PROBE_LIB"] = str(route_lib)
        old_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        prepend_env_path("LD_LIBRARY_PATH", str(route_lib.parent))
        env_changed |= old_ld_path != os.environ["LD_LIBRARY_PATH"]
        if env_changed and os.environ.get("_A5_CCU_PYTHON_REEXEC") != "1":
            env = os.environ.copy()
            env["_A5_CCU_PYTHON_REEXEC"] = "1"
            os.execvpe(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
        ctypes.CDLL(str(route_lib), mode=ctypes.RTLD_GLOBAL)
        sys.path.insert(0, str(package.parent))
        return extensions[0], route_lib
    raise RuntimeError("deep_ep_cpp not found; build and install the latest DeepEP wheel")


EXTENSION, ROUTE_LIB = prepare_runtime()

import torch
import torch.distributed as dist
import torch_npu
import deep_ep


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
        experimental_config = torch_npu.profiler._ExperimentalConfig(mstx=False, **kwargs)
    except TypeError:
        experimental_config = torch_npu.profiler._ExperimentalConfig(msprof_tx=False, **kwargs)
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=iterations, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output_dir)),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    )


HCCN_COUNTERS = (
    "nic_tx_all_pkg_num",
    "nic_tx_all_oct_num",
    "nic_rx_all_pkg_num",
    "nic_rx_all_oct_num",
    "roce_new_pkt_rty_num",
)
HCCN_COUNTER_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*([0-9]+)\s*$")


def resolve_hccn_tool(configured):
    candidates = [configured, shutil.which("hccn_tool"), "/usr/local/Ascend/driver/tools/hccn_tool"]
    return next((Path(path).resolve() for path in candidates if path and os.access(path, os.X_OK)), None)


def capture_hccn_stats(tool, devices, output_dir, phase, allowed_devices=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    supported = []
    clean_env = os.environ.copy()
    clean_env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    clean_env.pop("ASCEND_VISIBLE_DEVICES", None)
    for device in devices:
        if allowed_devices is not None and device not in allowed_devices:
            continue
        output_file = output_dir / f"{phase}_device{device}.txt"
        command = [str(tool), "-i", str(device), "-stat", "-g"]
        result = subprocess.run(command, env=clean_env, text=True, capture_output=True, check=False)
        header = (
            f"# command: {' '.join(command)}\n"
            f"# timestamp: {datetime.now(timezone.utc).astimezone().isoformat()}\n"
            f"# returncode: {result.returncode}\n"
        )
        output_file.write_text(header + result.stdout + result.stderr, errors="replace")
        if result.returncode == 0:
            supported.append(device)
            print(f"HCCN snapshot {phase}: physical device {device} -> {output_file}", flush=True)
        else:
            print(
                f"WARNING: hccn_tool -stat unavailable for physical device {device}; "
                f"see {output_file}",
                file=sys.stderr,
                flush=True,
            )
    (output_dir / f"{phase}_supported_devices.txt").write_text(
        "".join(f"{device}\n" for device in supported)
    )
    return supported


def load_hccn_counters(path):
    counters = {}
    for line in path.read_text(errors="replace").splitlines():
        match = HCCN_COUNTER_RE.match(line)
        if match:
            counters[match.group(1)] = int(match.group(2))
    return counters


def report_hccn_delta(output_dir, devices):
    rows = []
    print("HCCN counter deltas (after - before):", flush=True)
    for device in devices:
        before = load_hccn_counters(output_dir / f"before_device{device}.txt")
        after = load_hccn_counters(output_dir / f"after_device{device}.txt")
        common = sorted(before.keys() & after.keys())
        deltas = {key: after[key] - before[key] for key in common}
        summary = " ".join(
            f"{key}={deltas[key]}" if key in deltas else f"{key}=N/A"
            for key in HCCN_COUNTERS
        )
        print(f"  physical_device={device} {summary}", flush=True)
        rows.extend((device, key, before[key], after[key], deltas[key]) for key in common)
    output = output_dir / "hccn_counter_deltas.tsv"
    with output.open("w") as handle:
        handle.write("physical_device\tcounter\tbefore\tafter\tdelta\n")
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")
    print(f"HCCN raw snapshots and delta table: {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--route-index", type=int, default=0)
    parser.add_argument("--route-indices", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-root", default="/home/l00934901/profiling")
    parser.add_argument("--hccn-stat", action="store_true")
    parser.add_argument("--hccn-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--hccn-tool", default="")
    parser.add_argument("--hccn-stat-root", default="")
    parser.add_argument("--remote-only", action="store_true")
    args = parser.parse_args()
    if args.bytes <= 0 or args.bytes % 4:
        parser.error("--bytes must be a positive multiple of sizeof(float)")
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be non-negative and --iters must be positive")
    try:
        hccn_devices = [int(item.strip()) for item in args.hccn_devices.split(",") if item.strip()]
    except ValueError:
        parser.error("--hccn-devices must be a comma-separated list of physical device IDs")
    if args.hccn_stat and not hccn_devices:
        parser.error("--hccn-devices cannot be empty with --hccn-stat")

    if args.route_indices:
        os.environ["A5_CCU_ROUTE_INDICES"] = args.route_indices
    else:
        os.environ.pop("A5_CCU_ROUTE_INDICES", None)
        os.environ["A5_CCU_ROUTE_INDEX"] = str(args.route_index)
    if args.remote_only:
        os.environ["A5_CCU_REMOTE_ONLY"] = "1"
    else:
        os.environ.pop("A5_CCU_REMOTE_ONLY", None)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        parser.error("CCU multi-route write currently requires WORLD_SIZE=2")
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
    run_id = re.sub(
        r"[^A-Za-z0-9_.-]", "_", os.environ.get("TORCHELASTIC_RUN_ID", str(os.getppid()))
    )

    elements = args.bytes // 4
    send = torch.full((elements,), float(rank + 1), dtype=torch.float32, device="npu")
    recv = None
    for _ in range(args.warmup):
        recv = buffer.ccu_urma_multiroute_write(send)
        torch.npu.synchronize()
    dist.barrier()

    hccn_dir = None
    hccn_before = []
    if args.hccn_stat and rank == 0:
        hccn_tool = resolve_hccn_tool(args.hccn_tool)
        hccn_root = Path(args.hccn_stat_root or args.profile_root)
        routes = args.route_indices or str(args.route_index)
        hccn_dir = hccn_root / f"hccn_ccu_write_routes_{routes}_{run_id}"
        if hccn_tool is None:
            print(
                "WARNING: hccn_tool not found; use --hccn-tool /path/to/hccn_tool",
                file=sys.stderr,
                flush=True,
            )
        else:
            hccn_before = capture_hccn_stats(
                hccn_tool, hccn_devices, hccn_dir, "before"
            )
            if not hccn_before:
                print("WARNING: HCCN statistics unavailable on all requested devices", file=sys.stderr)
    dist.barrier()

    profiler = None
    if args.profile:
        profile_dir = Path(args.profile_root) / f"a5_ccu_urma_write_{run_id}" / f"rank{rank}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        profiler = make_profiler(profile_dir, args.iters)
        profiler.start()
        print(f"[rank={rank}] profiling={profile_dir}", flush=True)

    samples_us = []
    for _ in range(args.iters):
        begin = time.perf_counter()
        recv = buffer.ccu_urma_multiroute_write(send)
        torch.npu.synchronize()
        samples_us.append((time.perf_counter() - begin) * 1e6)
        if profiler is not None:
            profiler.step()
    if profiler is not None:
        profiler.stop()

    dist.barrier()
    if args.hccn_stat and rank == 0 and hccn_dir is not None and hccn_before:
        hccn_after = capture_hccn_stats(
            hccn_tool, hccn_devices, hccn_dir, "after", set(hccn_before)
        )
        common_devices = sorted(set(hccn_before) & set(hccn_after))
        if common_devices:
            report_hccn_delta(hccn_dir, common_devices)
        else:
            print("WARNING: no HCCN device has both before and after snapshots", file=sys.stderr)
    dist.barrier()

    sample_tensor = torch.tensor(samples_us, dtype=torch.float32, device="npu")
    dist.all_reduce(sample_tensor, op=dist.ReduceOp.MAX)
    if args.remote_only:
        peer = 1 - rank
        torch.testing.assert_close(recv[peer], torch.full_like(send, float(peer + 1)))
    else:
        expected = torch.stack(
            [torch.full_like(send, float(source_rank + 1)) for source_rank in range(world_size)]
        )
        torch.testing.assert_close(recv, expected)
    if rank == 0:
        maximum_rank_samples = sample_tensor.cpu().tolist()
        avg_us = sum(maximum_rank_samples) / len(maximum_rank_samples)
        routes = args.route_indices or str(args.route_index)
        print(
            f"PASS: CCU+URMA multi-route write routes={routes} bytes_per_rank={args.bytes} "
            f"warmup={args.warmup} iterations={args.iters} avg_us={avg_us:.3f}",
            flush=True,
        )
        print(f"deep_ep_cpp={EXTENSION}", flush=True)
        print(f"route_library={ROUTE_LIB}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
