#!/usr/bin/env python3
"""Hardware correctness/performance test for explicit 2/4-rank peer plans."""
import argparse
import atexit
import ctypes
from contextlib import nullcontext
from datetime import timedelta
import faulthandler
import json
import os
from pathlib import Path
import time

from a5_ccu_peer_test_support import (file_barrier, make_profiler, measurement_start,
    warmup_calls, measure_calls, validate_queued_calls, write_rank_result)
from a5_ccu_test_runtime import prepare_runtime


def phase(name):
    print(f"PEER_CASE_PHASE rank={os.environ.get('RANK', 'NA')} pid={os.getpid()} phase={name}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path)
    p.add_argument("--available-cards", type=int, default=8)
    p.add_argument("--bytes", type=int, default=4194304, help="bytes per destination, including self")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--sync-mode", choices=("per-call", "batch"), default="per-call",
                   help="batch is opt-in and requires queued changed-payload validation")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--runtime-check", action="store_true", help="check worker runtime/ABI without initializing devices or communicator")
    p.add_argument("--graph-backend", choices=("none", "aclgraph"), default="none")
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    if args.bytes <= 0 or args.bytes % 4 or args.warmup < 0 or args.iters < 1:
        p.error("positive FP32-aligned byte count and iterations required")
    if args.sync_mode == "batch" and args.graph_backend != "none":
        p.error("batch validation currently requires --graph-backend none; validate graph separately in per-call mode")
    if not args.runtime_check:
        if args.manifest is None or args.run_dir is None:
            p.error("--manifest and --run-dir are required for a hardware run")
        if not args.manifest.is_file():
            p.error(f"manifest does not exist: {args.manifest}")
        rank, k = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        if k not in (2, 4) or not 0 <= rank < k:
            p.error("initial implementation supports valid ranks in WORLD_SIZE=2 or 4")
    # This marker does not imply native/static destructors have finished.
    atexit.register(lambda: phase("python_atexit"))
    # Parse this entry point's arguments before loading any NPU runtime. Never
    # import an executable benchmark for helpers: its module initialization can
    # execvpe its own __file__ and replace this four-rank worker with a two-rank CLI.
    phase("runtime_bootstrap")
    extension_path, route_path = prepare_runtime(__file__)
    phase("import_runtime")
    import torch
    import torch.distributed as dist
    import torch_npu
    from deep_ep import Buffer
    import deep_ep.deep_ep_cpp as ext

    if Path(ext.__file__).resolve() != extension_path:
        raise RuntimeError(f"worker extension differs from bootstrap: {ext.__file__} != {extension_path}")
    route_library = ctypes.CDLL(str(route_path), mode=ctypes.RTLD_GLOBAL)
    deep_library = ctypes.CDLL(str(extension_path))
    abi_versions = {}
    for library, symbol in ((route_library, "A5CcuPeerPlanAbiVersion"),
                            (deep_library, "A5DeepEpPeerPlanAbiVersion")):
        version = getattr(library, symbol)
        version.restype = ctypes.c_int
        abi_versions[symbol] = version()
        if abi_versions[symbol] < 1:
            raise RuntimeError(f"invalid worker ABI: {symbol}")
    for api in ("prepare_ccu_urma_peer_plan", "bind_ccu_urma_peer_plan", "ccu_urma_peer_plan_alltoall_out"):
        if not hasattr(Buffer, api):
            raise RuntimeError(f"missing peer-plan Buffer API: {api}")
    if not hasattr(torch.ops.deep_ep, "ccu_urma_peer_plan_alltoall"):
        raise RuntimeError("missing peer-plan torch operator")
    print(f"PEER_RUNTIME_VERIFIED route={route_path} extension={extension_path} "
          f"route_ABI={abi_versions['A5CcuPeerPlanAbiVersion']} "
          f"deep_ABI={abi_versions['A5DeepEpPeerPlanAbiVersion']}", flush=True)
    if args.runtime_check:
        return

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    sync = args.run_dir / "sync"
    def barrier(tag):
        file_barrier(sync, tag, rank, k)
    phase("communicator_init")
    dist.init_process_group("hccl", init_method=f"file://{(args.run_dir / 'pg_init').resolve()}",
                            rank=rank, world_size=k, timeout=timedelta(seconds=180))
    buffer = Buffer(dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
    capture_stream = torch.npu.Stream() if args.graph_backend == "aclgraph" else None
    phase("prepare_plan")
    # Like the validated two-rank harness, prepare directly on the capture
    # stream in graph mode. Do not first create a second set on default stream.
    with torch.npu.stream(capture_stream) if capture_stream is not None else nullcontext():
        handle = buffer.prepare_ccu_urma_peer_plan("peer-plan", str(args.manifest.resolve()), args.available_cards)
    n = args.bytes // 4
    # Position + source + destination identify each slice. Poisoned receive
    # buffers also reject missing chunks, overlapping writes and stale outputs.
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
    def check(tensor, epoch):
        torch.testing.assert_close(tensor, expected + epoch, rtol=0, atol=0)
    def eager():
        if capture_stream is not None:
            capture_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(capture_stream):
                return buffer.ccu_urma_peer_plan_alltoall_out(send, recv, handle)
        return buffer.ccu_urma_peer_plan_alltoall_out(send, recv, handle)
    # Multiple changed payloads catch stale output and notify reuse bugs.
    phase("changed_payload_checks")
    for epoch in range(3):
        if epoch:
            send.add_(1)
        recv.fill_(-999)
        torch.npu.synchronize()
        barrier(f"correctness_{epoch}")
        eager()
        torch.npu.synchronize()
        check(recv, epoch)
    launch = eager
    epoch = 2
    graph_output = recv
    batch_validation = dict(status="NOT_REQUESTED", rounds=0, calls_per_round=0)
    if args.sync_mode == "batch":
        phase("batch_validation_begin")
        queued_count = min(max(args.iters, 2), 20)
        # Keep native/current stream fixed. Torch producers and snapshots must
        # be ordered around the raw CCU launch by the existing bridge/runtime.
        # Reuse send/recv exactly as timing does, but vary inputs and poison
        # output each call so a stale or missing execution cannot pass.
        for round_index in range(2):
            deltas = [100 * (round_index + 1) + i + 1 for i in range(queued_count)]
            inputs = [send + delta for delta in deltas]
            snapshots = [torch.empty_like(recv) for _ in deltas]
            torch.npu.synchronize()
            barrier(f"batch_validation_{round_index}_begin")
            def prepare(index):
                send.copy_(inputs[index])
                recv.fill_(-999)
            def snapshot(index):
                snapshots[index].copy_(recv)
            def check_snapshot(index):
                check(snapshots[index], epoch + deltas[index])
            validate_queued_calls(prepare, eager, snapshot, torch.npu.synchronize,
                                  check_snapshot, queued_count)
            epoch += deltas[-1]
            check(recv, epoch)
            barrier(f"batch_validation_{round_index}_passed")
            del inputs, snapshots
        batch_validation = dict(status="PASS", rounds=2, calls_per_round=queued_count)
        print(f"PEER_BATCH_VALIDATION rank={rank} PASS rounds=2 calls_per_round={queued_count} "
              "changed_input=1 poisoned_output=1 every_snapshot_checked=1", flush=True)
    if args.graph_backend == "aclgraph":
        phase("bind_capture_stream")
        capture_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(capture_stream):
            buffer.bind_ccu_urma_peer_plan(handle)
            graph_output = torch.ops.deep_ep.ccu_urma_peer_plan_alltoall(send, handle, [])
        torch.npu.synchronize()
        check(graph_output, epoch)
        barrier("capture_begin")
        graph = torch.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=capture_stream, auto_dispatch_capture=True):
            graph_output = torch.ops.deep_ep.ccu_urma_peer_plan_alltoall(send, handle, [])
        torch.npu.synchronize()
        barrier("capture_done")
        for index in range(2):
            send.add_(1)
            epoch += 1
            torch.npu.synchronize()
            barrier(f"replay_{index}")
            graph.replay()
            torch.npu.synchronize()
            check(graph_output, epoch)
        print(f"PEER_GRAPH_CAPTURE_REPLAY rank={rank} PASS", flush=True)
        launch = graph.replay
    phase("warmup")
    warmup_calls(launch, torch.npu.synchronize, args.warmup, args.sync_mode)
    check(graph_output, epoch)
    barrier("warmup_done")
    profiler = make_profiler(args.run_dir / "profiling" / f"rank{rank}", rank) if args.profile else None
    if profiler:
        profiler.start()
    try:
        torch.npu.synchronize()  # align the profiler-start boundary with native baseline
        phase("measurement_rendezvous")
        start_lateness_us = measurement_start(sync, rank, k)
        phase("measure")
        print(f"PEER_MEASUREMENT rank={rank} sync_mode={args.sync_mode} "
              f"calls={args.iters} warmup_in_profile=0", flush=True)
        timing = measure_calls(launch, torch.npu.synchronize, args.iters, args.sync_mode)
    finally:
        if profiler:
            profiler.stop()
    check(graph_output, epoch)
    barrier("measured")
    result = dict(rank=rank, ranks=k, correctness="PASS",
        bytes_per_peer=args.bytes, tensor_bytes=k * args.bytes, warmup=args.warmup,
        iterations=args.iters, graph_backend=args.graph_backend, **timing,
        batch_validation=batch_validation, measurement_start_lateness_us=start_lateness_us,
        manifest=str(args.manifest))
    write_rank_result(args.run_dir, rank, result)
    print("RESULT_JSON " + json.dumps(result), flush=True)
    # All rank results have been recorded, but cleanup is a separate obligation.
    # Native shutdown monitoring is external; cancel this Python watchdog
    # after normal cleanup just as the validated two-rank harness does.
    cleanup_watchdog = int(os.environ.get("A5_CCU_PEER_CLEANUP_WATCHDOG_SECONDS", "30"))
    faulthandler.enable(all_threads=True)
    if cleanup_watchdog > 0:
        faulthandler.dump_traceback_later(cleanup_watchdog, repeat=True)
    phase("final_synchronize_begin")
    torch.npu.synchronize()
    phase("final_synchronize_done")
    phase("reported_barrier_begin")
    barrier("reported")
    phase("reported_barrier_done")
    phase("destroy_process_group_begin")
    dist.destroy_process_group()
    phase("destroy_process_group_done")
    faulthandler.cancel_dump_traceback_later()
    phase("cleanup_watchdog_cancelled")
    phase("main_return_begin")


if __name__ == "__main__":
    try:
        main()
        phase("main_return_done")
    except Exception as exc:
        print(f"PEER_CASE_FAILURE rank={os.environ.get('RANK')} error={type(exc).__name__}:{exc}", flush=True)
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
