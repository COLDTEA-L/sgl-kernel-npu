#!/usr/bin/env python3
"""Standalone two-rank native HCCL AllToAll performance and jitter test."""

import argparse
import json
import math
import os
import statistics
import time

os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "CCU_SCHED")

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401: registers the NPU backend


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples_us):
    mean = statistics.fmean(samples_us)
    stddev = statistics.pstdev(samples_us)
    return {
        "mean_us": mean,
        "p50_us": percentile(samples_us, 0.50),
        "p95_us": percentile(samples_us, 0.95),
        "p99_us": percentile(samples_us, 0.99),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "stddev_us": stddev,
        "cv_percent": stddev / mean * 100.0 if mean else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=2 * 1024 * 1024,
                        help="bytes sent to each rank, including the local rank")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100,
                        help="iterations for host batch-average timing")
    parser.add_argument("--sample-iters", type=int, default=100,
                        help="iterations measured separately with NPU events")
    args = parser.parse_args()
    item_size = 4 if args.dtype == "float32" else 2
    if args.bytes <= 0 or args.bytes % item_size:
        parser.error(f"--bytes must be a positive multiple of {item_size}")
    if args.warmup < 0 or args.iters <= 0 or args.sample_iters <= 0:
        parser.error("warmup/iters/sample-iters are invalid")
    return args


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("this baseline currently requires WORLD_SIZE=2")

    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    item_size = torch.empty((), dtype=dtype).element_size()
    elements_per_peer = args.bytes // item_size

    send = torch.stack([
        torch.full((elements_per_peer,), rank * 100 + dst + 1,
                   dtype=dtype, device="npu")
        for dst in range(world_size)
    ])
    recv = torch.empty_like(send)
    expected = torch.stack([
        torch.full((elements_per_peer,), src * 100 + rank + 1,
                   dtype=dtype, device="npu")
        for src in range(world_size)
    ])

    def run_op():
        dist.all_to_all_single(recv, send)

    for _ in range(args.warmup):
        run_op()
    torch.npu.synchronize()
    torch.testing.assert_close(recv, expected, rtol=0, atol=0)
    dist.barrier()

    torch.npu.synchronize()
    begin = time.perf_counter()
    for _ in range(args.iters):
        run_op()
    torch.npu.synchronize()
    host_batch_avg_us = (time.perf_counter() - begin) * 1e6 / args.iters

    event_pairs = []
    for _ in range(args.sample_iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        run_op()
        end.record()
        event_pairs.append((start, end))
    torch.npu.synchronize()
    samples_us = [start.elapsed_time(end) * 1000.0 for start, end in event_pairs]
    torch.testing.assert_close(recv, expected, rtol=0, atol=0)

    stats = summarize(samples_us)
    stats.update({
        "rank": rank,
        "physical_visibility": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "bytes_per_peer": args.bytes,
        "network_bytes_per_rank": args.bytes,
        "input_bytes_per_rank": args.bytes * world_size,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "batch_iterations": args.iters,
        "sample_iterations": args.sample_iters,
        "host_batch_avg_us": host_batch_avg_us,
        "hccl_expansion_mode": os.environ.get("HCCL_OP_EXPANSION_MODE", ""),
    })

    metric_names = ["host_batch_avg_us", "mean_us", "p50_us", "p95_us",
                    "p99_us", "min_us", "max_us", "cv_percent"]
    local_metrics = torch.tensor([stats[name] for name in metric_names],
                                 dtype=torch.float32, device="npu")
    gathered = [torch.empty_like(local_metrics) for _ in range(world_size)]
    dist.all_gather(gathered, local_metrics)

    print("RANK_RESULT " + json.dumps(stats, sort_keys=True), flush=True)
    if rank == 0:
        rank_metrics = [tensor.cpu().tolist() for tensor in gathered]
        summary = {
            "operator": "torch.distributed.all_to_all_single",
            "engine_requested": "CCU_SCHED",
            "bytes_per_peer": args.bytes,
            "dtype": args.dtype,
            "rank_results": [dict(zip(metric_names, values)) for values in rank_metrics],
            "max_rank_host_batch_avg_us": max(values[0] for values in rank_metrics),
            "max_rank_event_p50_us": max(values[2] for values in rank_metrics),
            "max_rank_event_p95_us": max(values[3] for values in rank_metrics),
            "max_rank_cv_percent": max(values[7] for values in rank_metrics),
        }
        print("BASELINE_RESULT " + json.dumps(summary, sort_keys=True), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
