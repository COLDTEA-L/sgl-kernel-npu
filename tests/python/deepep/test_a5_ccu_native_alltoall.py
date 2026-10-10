#!/usr/bin/env python3
"""Native HCCL AllToAll baseline using the same peer-plan measurement helpers.

No DeepEP, custom Channel/plan preparation, route preload or patched HCCL.
Keep imports safe for --help and CPU regression tests.
"""
import argparse
import atexit
from datetime import timedelta
import faulthandler
import json
import os
from pathlib import Path
import time

from a5_ccu_peer_test_support import file_barrier, make_profiler, measurement_start


def phase(name):
    print(f"NATIVE_CASE_PHASE rank={os.environ.get('RANK', 'NA')} pid={os.getpid()} phase={name}", flush=True)


def write_rank_result(run_dir, rank, result):
    # Separate atomic files avoid four workers interleaving RESULT_JSON lines.
    directory = Path(run_dir) / "results"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"rank{rank}.json"
    temp = target.with_suffix(f".pid{os.getpid()}.tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(target)


def warmup_calls(launch, synchronize, count, sync_mode):
    for _ in range(count):
        launch()
        if sync_mode == "per-call":
            synchronize()
    synchronize()


def measure_calls(launch, synchronize, count, sync_mode, clock=time.perf_counter_ns):
    """Batch averages are not fabricated individual-call latency samples."""
    if sync_mode == "batch":
        start = clock()
        for _ in range(count):
            launch()
        synchronize()
        total_us = (clock() - start) / 1000
        return dict(sync_mode=sync_mode, host_call_us=[],
                    host_batch_total_us=total_us, host_avg_us=total_us / count)
    samples = []
    for _ in range(count):
        start = clock()
        launch()
        synchronize()
        samples.append((clock() - start) / 1000)
    return dict(sync_mode=sync_mode, host_call_us=samples,
                host_avg_us=sum(samples) / count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=4194304, help="FP32 bytes per destination, including self")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--sync-mode", choices=("batch", "per-call"), default="batch",
                        help="batch: synchronize after all calls; per-call: isolated-call control")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.bytes <= 0 or args.bytes % 4 or args.warmup < 0 or args.iters < 1:
        parser.error("positive FP32-aligned bytes/iterations and nonnegative warmup required")
    rank, k = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if k not in (2, 4) or not 0 <= rank < k:
        parser.error("requires two or four valid ranks")
    atexit.register(lambda: phase("python_atexit"))
    phase("import_runtime")
    import torch
    import torch.distributed as dist
    import torch_npu  # registers the NPU/HCCL backend

    print(f"NATIVE_RUNTIME torch={torch.__version__} torch_npu={getattr(torch_npu, '__version__', 'unknown')} "
          f"api=dist.all_to_all_single expansion={os.environ.get('HCCL_OP_EXPANSION_MODE')} "
          f"visible_devices={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}", flush=True)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    sync = args.run_dir / "sync"
    def barrier(tag):
        file_barrier(sync, tag, rank, k)
    phase("communicator_init")
    dist.init_process_group("hccl", init_method=f"file://{(args.run_dir / 'pg_init').resolve()}",
                            rank=rank, world_size=k, timeout=timedelta(seconds=180))
    n = args.bytes // 4
    pattern = torch.arange(n, dtype=torch.int32, device="npu").remainder_(8191).float()
    send = torch.empty((k, n), dtype=torch.float32, device="npu")
    recv = torch.empty_like(send)
    for peer in range(k):
        send[peer].copy_(pattern + rank * 1000000 + peer * 100000)
    expected = torch.empty_like(send)
    for source in range(k):
        expected[source].copy_(pattern + source * 1000000 + rank * 100000)
    torch.npu.synchronize()
    barrier("prepared")
    def check(epoch):
        torch.testing.assert_close(recv, expected + epoch, rtol=0, atol=0)
    def launch():
        # Same equal-size AllToAll layout as custom [k, elements_per_peer].
        dist.all_to_all_single(recv, send)
    phase("changed_payload_checks")
    for epoch in range(3):
        if epoch:
            send.add_(1)
        recv.fill_(-999)
        torch.npu.synchronize()
        barrier(f"correctness_{epoch}")
        launch()
        torch.npu.synchronize()
        check(epoch)
    # First-call communicator/algorithm resource creation is already outside timing.
    phase("warmup")
    warmup_calls(launch, torch.npu.synchronize, args.warmup, args.sync_mode)
    check(2)
    barrier("warmup_done")
    # Start only after every rank finishes warmup. The trace excludes resource
    # creation, payload prechecks and warmup; it covers the measured calls below.
    profiler = make_profiler(args.run_dir / "profiling" / f"rank{rank}", rank) if args.profile else None
    if profiler:
        profiler.start()
    try:
        # Drain device work after enabling profiling, before the timed batch.
        torch.npu.synchronize()
        phase("measurement_rendezvous")
        lateness = measurement_start(sync, rank, k)
        phase("measure")
        print(f"NATIVE_MEASUREMENT rank={rank} sync_mode={args.sync_mode} "
              f"calls={args.iters} warmup_in_profile=0", flush=True)
        timing = measure_calls(launch, torch.npu.synchronize, args.iters, args.sync_mode)
    finally:
        if profiler:
            profiler.stop()
    check(2)
    barrier("measured")
    # Read-only evidence of the runtime actually loaded; no CDLL preload of a patch.
    maps = Path("/proc/self/maps")
    hccl_libraries = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                             if "libhccl" in line and "/" in line}) if maps.is_file() else []
    result = dict(implementation="native_hccl", api="dist.all_to_all_single",
        rank=rank, ranks=k, correctness="PASS", bytes_per_peer=args.bytes,
        tensor_bytes=k * args.bytes, network_send_bytes=(k - 1) * args.bytes,
        warmup=args.warmup, iterations=args.iters, graph_backend="none",
        **timing,
        measurement_start_lateness_us=lateness, hccl_libraries=hccl_libraries)
    write_rank_result(args.run_dir, rank, result)
    print("RESULT_JSON " + json.dumps(result), flush=True)
    watchdog = int(os.environ.get("A5_CCU_PEER_CLEANUP_WATCHDOG_SECONDS", "30"))
    faulthandler.enable(all_threads=True)
    if watchdog > 0:
        faulthandler.dump_traceback_later(watchdog, repeat=True)
    phase("final_synchronize_begin")
    torch.npu.synchronize()
    phase("final_synchronize_done")
    barrier("reported")
    phase("destroy_process_group_begin")
    dist.destroy_process_group()
    phase("destroy_process_group_done")
    faulthandler.cancel_dump_traceback_later()
    phase("main_return_begin")


if __name__ == "__main__":
    try:
        main()
        phase("main_return_done")
    except Exception as exc:
        print(f"NATIVE_CASE_FAILURE rank={os.environ.get('RANK')} error={type(exc).__name__}:{exc}", flush=True)
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
