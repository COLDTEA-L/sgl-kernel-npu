#!/usr/bin/env python3
"""Two-rank CCU+URMA multi-route write with integrated NPU profiling."""

import argparse
import ctypes
import os
import re
import site
import sys
import time
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--route-index", type=int, default=0)
    parser.add_argument("--route-indices", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-root", default="/home/l00934901/profiling")
    parser.add_argument("--remote-only", action="store_true")
    args = parser.parse_args()
    if args.bytes <= 0 or args.bytes % 4:
        parser.error("--bytes must be a positive multiple of sizeof(float)")
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be non-negative and --iters must be positive")

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

    elements = args.bytes // 4
    send = torch.full((elements,), float(rank + 1), dtype=torch.float32, device="npu")
    recv = None
    for _ in range(args.warmup):
        recv = buffer.ccu_urma_multiroute_write(send)
        torch.npu.synchronize()
    dist.barrier()

    profiler = None
    if args.profile:
        run_id = re.sub(
            r"[^A-Za-z0-9_.-]", "_", os.environ.get("TORCHELASTIC_RUN_ID", str(os.getppid()))
        )
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

    sample_tensor = torch.tensor(samples_us, dtype=torch.float32, device="npu")
    dist.all_reduce(sample_tensor, op=dist.ReduceOp.MAX)
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
