#!/usr/bin/env python3
"""Two-card pinned-host H2D/D2H contention worker.

This does not initialize HCCL or load DeepEP.  Two torchrun workers use one
NPU each, coordinate with files, and compare solo versus simultaneous copies.
"""

import argparse
import atexit
import faulthandler
import json
import os
from pathlib import Path
import time

from a5_ccu_peer_test_support import file_barrier, measurement_start, write_rank_result
from a5_h2d_io_test_support import (
    DEFAULT_CASES,
    bandwidth_gbps,
    direction_for,
    parse_cases,
    parse_positive_csv,
)


def phase(name):
    print(f"H2D_IO_PHASE rank={os.environ.get('RANK', 'NA')} pid={os.getpid()} phase={name}", flush=True)


def physical_device(local_rank):
    visible = [item.strip() for item in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",") if item.strip()]
    return int(visible[local_rank]) if len(visible) == 2 and all(item.isdecimal() for item in visible) else None


def submit_copy(direction, host_tensor, device_tensor):
    if direction == "h2d":
        device_tensor.copy_(host_tensor, non_blocking=True)
    elif direction == "d2h":
        host_tensor.copy_(device_tensor, non_blocking=True)
    else:
        raise ValueError(f"cannot submit direction {direction}")


def prepare_payload(torch, direction, host_tensor, device_tensor, value):
    if direction == "h2d":
        host_tensor.fill_(value)
        device_tensor.zero_()
    elif direction == "d2h":
        device_tensor.fill_(value)
        host_tensor.zero_()
    else:
        raise ValueError(f"cannot prepare direction {direction}")


def verify_payload(torch, direction, host_tensor, device_tensor, value):
    if direction == "h2d":
        observed = device_tensor.cpu()
    elif direction == "d2h":
        observed = host_tensor
    else:
        raise ValueError(f"cannot verify direction {direction}")
    if observed.numel() == 0:
        raise AssertionError("empty verification tensor")
    minimum = int(observed.min().item())
    maximum = int(observed.max().item())
    if minimum != value or maximum != value:
        raise AssertionError(
            f"{direction} verification failed: expected={value} min={minimum} max={maximum}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="65536,262144,1048576,4194304,16777216",
                        help="comma-separated bytes per copy")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES),
                        help="comma-separated experiment case names")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        sizes = parse_positive_csv(args.sizes, "--sizes")
        cases = parse_cases(args.cases)
    except ValueError as exc:
        parser.error(str(exc))
    if args.warmup < 0 or args.iters < 1 or args.repeats < 1:
        parser.error("nonnegative warmup and positive iters/repeats required")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 or rank not in (0, 1) or local_rank not in (0, 1):
        parser.error("this experiment requires exactly two torchrun workers")
    atexit.register(lambda: phase("python_atexit"))
    faulthandler.enable(all_threads=True)

    phase("import_runtime")
    import torch
    import torch_npu  # noqa: F401 - registers the NPU backend and pinned allocator

    torch.npu.set_device(local_rank)
    maximum = max(sizes)
    phase("allocate_buffers")
    host_buffer = torch.empty(maximum, dtype=torch.uint8, pin_memory=True)
    if not host_buffer.is_pinned():
        raise RuntimeError("pinned host allocation is required; pageable fallback is not accepted")
    device_buffer = torch.empty(maximum, dtype=torch.uint8, device="npu")
    torch.npu.synchronize()

    sync_root = args.run_dir / "sync"
    file_barrier(sync_root / "allocated", "ready", rank, world_size)
    correctness = []
    records = []

    for size in sizes:
        host_view = host_buffer[:size]
        device_view = device_buffer[:size]
        for direction_index, direction in enumerate(("h2d", "d2h")):
            value = 1 + (23 + 41 * rank + 7 * direction_index + (size % 113)) % 250
            prepare_payload(torch, direction, host_view, device_view, value)
            torch.npu.synchronize()
            gate_dir = sync_root / f"gate_{size}_{direction}"
            file_barrier(gate_dir, "prepared", rank, world_size)
            submit_copy(direction, host_view, device_view)
            torch.npu.synchronize()
            verify_payload(torch, direction, host_view, device_view, value)
            correctness.append({"bytes": size, "direction": direction, "status": "PASS"})
            file_barrier(gate_dir, "passed", rank, world_size)

        for case_name in cases:
            direction = direction_for(case_name, rank)
            active = direction != "idle"
            case_dir = sync_root / f"size_{size}" / case_name
            if active:
                value = 1 + (37 + 53 * rank + 11 * len(records)) % 250
                prepare_payload(torch, direction, host_view, device_view, value)
                torch.npu.synchronize()
                for _ in range(args.warmup):
                    submit_copy(direction, host_view, device_view)
                torch.npu.synchronize()
                verify_payload(torch, direction, host_view, device_view, value)
            else:
                value = None
                torch.npu.synchronize()
            file_barrier(case_dir, "warmup_done", rank, world_size)

            for repeat in range(args.repeats):
                repeat_dir = case_dir / f"repeat_{repeat + 1}"
                start_event = torch.npu.Event(enable_timing=True) if active else None
                end_event = torch.npu.Event(enable_timing=True) if active else None
                lateness_us = measurement_start(repeat_dir, rank, world_size)
                started_ns = time.monotonic_ns()
                if active:
                    start_event.record()
                    for _ in range(args.iters):
                        submit_copy(direction, host_view, device_view)
                    end_event.record()
                    torch.npu.synchronize()
                    ended_ns = time.monotonic_ns()
                    device_total_us = start_event.elapsed_time(end_event) * 1000.0
                    host_total_us = (ended_ns - started_ns) / 1000.0
                    device_avg_us = device_total_us / args.iters
                    host_avg_us = host_total_us / args.iters
                    verify_payload(torch, direction, host_view, device_view, value)
                    record = {
                        "case": case_name,
                        "direction": direction,
                        "active": True,
                        "bytes": size,
                        "repeat": repeat + 1,
                        "iterations": args.iters,
                        "start_lateness_us": lateness_us,
                        "started_ns": started_ns,
                        "ended_ns": ended_ns,
                        "host_total_us": host_total_us,
                        "host_avg_us": host_avg_us,
                        "device_total_us": device_total_us,
                        "device_avg_us": device_avg_us,
                        "host_gbps": bandwidth_gbps(size, host_avg_us),
                        "device_gbps": bandwidth_gbps(size, device_avg_us),
                        "correctness": "PASS",
                    }
                else:
                    ended_ns = time.monotonic_ns()
                    record = {
                        "case": case_name,
                        "direction": "idle",
                        "active": False,
                        "bytes": size,
                        "repeat": repeat + 1,
                        "iterations": 0,
                        "start_lateness_us": lateness_us,
                        "started_ns": started_ns,
                        "ended_ns": ended_ns,
                        "correctness": "NOT_APPLICABLE",
                    }
                records.append(record)
                file_barrier(repeat_dir, "finished", rank, world_size)

    result = {
        "rank": rank,
        "local_rank": local_rank,
        "physical_device": physical_device(local_rank),
        "world_size": world_size,
        "pinned_host": bool(host_buffer.is_pinned()),
        "sizes": sizes,
        "cases": cases,
        "warmup": args.warmup,
        "iterations": args.iters,
        "repeats": args.repeats,
        "correctness": correctness,
        "records": records,
    }
    write_rank_result(args.run_dir, rank, result)
    print("H2D_IO_RESULT " + json.dumps({
        "rank": rank,
        "physical_device": result["physical_device"],
        "correctness": "PASS",
        "record_count": len(records),
    }), flush=True)
    phase("reported_barrier_begin")
    file_barrier(sync_root / "reported", "done", rank, world_size)
    phase("reported_barrier_done")
    torch.npu.synchronize()
    phase("main_return_begin")


if __name__ == "__main__":
    try:
        main()
        phase("main_return_done")
    except Exception as exc:
        print(f"H2D_IO_FAILURE rank={os.environ.get('RANK')} error={type(exc).__name__}:{exc}", flush=True)
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
