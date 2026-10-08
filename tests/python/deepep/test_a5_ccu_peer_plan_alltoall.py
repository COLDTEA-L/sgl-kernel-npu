#!/usr/bin/env python3
"""Hardware correctness/performance test for explicit 2/4-rank peer plans."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch_npu
from deep_ep import Buffer
from test_a5_ccu_urma_multiroute_all2all import file_barrier, make_profiler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--available-cards", type=int, default=8)
    p.add_argument("--bytes", type=int, default=4194304, help="bytes per destination, including self")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--graph-backend", choices=("none", "aclgraph"), default="none")
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    if args.bytes <= 0 or args.bytes % 4 or args.warmup < 0 or args.iters < 1:
        p.error("positive FP32-aligned byte count and iterations required")
    rank, k = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if k not in (2, 4):
        p.error("initial implementation supports 2 or 4 ranks")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    sync = args.run_dir / "sync"
    def phase(name):
        print(f"PEER_CASE_PHASE rank={rank} phase={name}", flush=True)
    def barrier(tag):
        file_barrier(sync, tag, rank, k)
    phase("communicator_init")
    dist.init_process_group("hccl", init_method=f"file://{(args.run_dir / 'pg_init').resolve()}",
                            rank=rank, world_size=k, timeout=timedelta(seconds=180))
    buffer = Buffer(dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
    phase("prepare_plan")
    handle = buffer.prepare_ccu_urma_peer_plan("peer-plan", str(args.manifest.resolve()), args.available_cards)
    n = args.bytes // 4
    # Values identify both source and destination; self-copy errors cannot pass.
    send = torch.empty((k, n), dtype=torch.float32, device="npu")
    recv = torch.empty_like(send)
    for peer in range(k):
        send[peer].fill_(rank * 100 + peer)
    expected = torch.empty_like(send)
    for source in range(k):
        expected[source].fill_(source * 100 + rank)
    torch.npu.synchronize()
    barrier("prepared")
    def check(tensor, epoch):
        torch.testing.assert_close(tensor, expected + epoch, rtol=0, atol=0)
    def eager():
        return buffer.ccu_urma_peer_plan_alltoall_out(send, recv, handle)
    # Multiple changed payloads catch stale output and notify reuse bugs.
    phase("changed_payload_checks")
    for epoch in range(3):
        if epoch:
            send.add_(1)
        torch.npu.synchronize()
        barrier(f"correctness_{epoch}")
        eager()
        torch.npu.synchronize()
        check(recv, epoch)
    launch = eager
    epoch = 2
    graph_output = recv
    if args.graph_backend == "aclgraph":
        phase("bind_capture_stream")
        capture_stream = torch.npu.Stream()
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
    for _ in range(args.warmup):
        launch()
        torch.npu.synchronize()  # independent completed calls, not a queued batch
    barrier("warmup_done")
    profiler = make_profiler(args.run_dir / "profiling" / f"rank{rank}", rank) if args.profile else None
    if profiler:
        profiler.start()
    phase("measure")
    samples = []
    for _ in range(args.iters):
        start = time.perf_counter_ns()
        launch()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1000)
    if profiler:
        profiler.stop()
    check(graph_output, epoch)
    barrier("measured")
    print("RESULT_JSON " + json.dumps(dict(rank=rank, ranks=k, correctness="PASS",
        bytes_per_peer=args.bytes, tensor_bytes=k * args.bytes, warmup=args.warmup,
        iterations=args.iters, graph_backend=args.graph_backend, host_call_us=samples,
        host_avg_us=sum(samples)/len(samples), manifest=str(args.manifest))), flush=True)
    # Keep Buffer/plan/communicator alive through all asynchronous work and profiling.
    torch.npu.synchronize()
    barrier("reported")
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PEER_CASE_FAILURE rank={os.environ.get('RANK')} error={type(exc).__name__}:{exc}", flush=True)
        raise
