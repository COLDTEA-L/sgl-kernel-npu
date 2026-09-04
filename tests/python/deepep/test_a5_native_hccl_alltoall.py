#!/usr/bin/env python3
"""Benchmark native torch.distributed/HCCL AllToAll on Ascend NPU."""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401: registers the NPU backend with PyTorch


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements-per-peer", type=int, default=11_804_800)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.elements_per_peer <= 0:
        parser.error("--elements-per-peer must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")

    # Equal-split AllToAll: one int32 block for every destination, including self.
    send = torch.empty(
        (world_size, args.elements_per_peer), dtype=torch.int32, device="npu"
    )
    for dst in range(world_size):
        send[dst].fill_(rank * world_size + dst)
    send = send.contiguous()
    recv = torch.empty_like(send)

    dist.barrier()
    for _ in range(args.warmup):
        dist.all_to_all_single(recv, send)
        torch.npu.synchronize()

    dist.barrier()
    times_ms = []
    elapsed = torch.zeros(1, dtype=torch.float32, device="npu")
    for _ in range(args.iters):
        torch.npu.synchronize()
        start = time.perf_counter()
        dist.all_to_all_single(recv, send)
        torch.npu.synchronize()
        local_ms = (time.perf_counter() - start) * 1000.0

        elapsed.fill_(local_ms)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if rank == 0:
            times_ms.append(elapsed.item())

    for src in range(world_size):
        expected = src * world_size + rank
        if not torch.all(recv[src] == expected).item():
            raise AssertionError(f"rank {rank}: invalid data received from rank {src}")

    dist.barrier()
    if rank == 0:
        bytes_per_peer = args.elements_per_peer * 4
        remote_bytes_per_rank = (world_size - 1) * bytes_per_peer
        average_ms = sum(times_ms) / len(times_ms)
        effective_gbps = remote_bytes_per_rank * 8.0 / (average_ms * 1.0e6)
        print("PASS: native torch.distributed/HCCL all_to_all_single", flush=True)
        print(f"HCCL_OP_EXPANSION_MODE={os.environ.get('HCCL_OP_EXPANSION_MODE', '<default>')}")
        print(f"world_size={world_size}")
        print(f"elements_per_peer={args.elements_per_peer}")
        print(f"bytes_per_peer={bytes_per_peer}")
        print(f"remote_bytes_per_rank={remote_bytes_per_rank}")
        print(f"total_input_bytes_per_rank={world_size * bytes_per_peer}")
        print(f"warmup={args.warmup}, iterations={args.iters}")
        print(f"average={average_ms:.3f} ms")
        print(f"p50={percentile(times_ms, 0.50):.3f} ms")
        print(f"p95={percentile(times_ms, 0.95):.3f} ms")
        print(f"min={min(times_ms):.3f} ms")
        print(f"max={max(times_ms):.3f} ms")
        print(f"effective_remote_bandwidth={effective_gbps:.3f} Gbps")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
