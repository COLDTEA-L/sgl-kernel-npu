#!/usr/bin/env python3
"""Two-rank A5 ValueWait/ValueWrite + host-callback puncture experiment."""

import argparse
import ctypes
import os
from pathlib import Path

from a5_ccu_test_runtime import prepare_runtime

os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "CCU_SCHED")
EXTENSION, _ = prepare_runtime(__file__, require_route_library=False)

import torch
import torch.distributed as dist
import torch_npu
import deep_ep

PATH_POLICY_ABI = 1
PATH_POLICY_ABI_SHIFT = 60
PATH_POLICY_READY_BIT = 59
PATH_POLICY_READY_MASK = 1 << PATH_POLICY_READY_BIT


def encode_path_policy(weights, ready=False):
    if not 2 <= len(weights) <= 13:
        raise ValueError("path policy requires 2..13 catalog weights")
    if any(weight < 0 or weight > 15 for weight in weights):
        raise ValueError("path weights must be in [0, 15]")
    if not any(weights):
        raise ValueError("at least one path must be enabled")
    word = PATH_POLICY_ABI << PATH_POLICY_ABI_SHIFT
    for index, weight in enumerate(weights):
        word |= weight << (index * 4)
    if ready:
        word |= PATH_POLICY_READY_MASK
    if word >= 1 << 63:
        raise ValueError("path policy must fit in signed int64")
    return word


def parse_weights(text, name):
    try:
        weights = [int(item) for item in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from error
    if not 2 <= len(weights) <= 13 or any(value < 0 or value > 15 for value in weights):
        raise argparse.ArgumentTypeError(f"{name} requires 2..13 values in [0, 15]")
    if not any(weights):
        raise argparse.ArgumentTypeError(f"{name} must enable at least one path")
    return weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=4 * 1024 * 1024,
                        help="bytes per destination slice")
    parser.add_argument("--relay-manifest", required=True)
    parser.add_argument("--plan-id", default="valuewait-direct-plus-2relay")
    parser.add_argument("--direct-weights", default="1,0,0",
                        help="initial scheduler decision; default is direct-only")
    parser.add_argument("--scheduled-weights", default="2,1,1",
                        help="second decision written by the host scheduler")
    parser.add_argument("--timeout-ms", type=int, default=60000)
    args = parser.parse_args()
    try:
        args.direct_weights = parse_weights(args.direct_weights, "--direct-weights")
        args.scheduled_weights = parse_weights(args.scheduled_weights, "--scheduled-weights")
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if len(args.direct_weights) != len(args.scheduled_weights):
        parser.error("direct and scheduled policies must have the same path count")
    if args.bytes <= 0 or args.bytes % 4:
        parser.error("--bytes must be a positive multiple of 4")
    if args.timeout_ms <= 0:
        parser.error("--timeout-ms must be positive")
    return args


def check_extension_abi():
    extension = ctypes.CDLL(str(EXTENSION))
    try:
        fn = extension.A5DeepEpValueWaitCallbackAbiVersion
    except AttributeError as error:
        raise RuntimeError(
            f"stale DeepEP extension without ValueWait/callback ABI marker: {EXTENSION}"
        ) from error
    fn.restype = ctypes.c_int
    if fn() < 1:
        raise RuntimeError("invalid ValueWait/callback ABI")


def init_process_group(rank, world_size):
    init_file = os.environ.get("A5_CCU_PG_INIT_FILE")
    if init_file:
        path = Path(init_file).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        dist.init_process_group("hccl", init_method=f"file://{path}",
                                rank=rank, world_size=world_size)
    else:
        dist.init_process_group("hccl")


def main():
    args = parse_args()
    check_extension_abi()
    manifest = Path(args.relay_manifest).resolve()
    if not manifest.is_file():
        raise RuntimeError(f"relay manifest not found: {manifest}")
    os.environ["A5_CCU_EXPLICIT_MULTIPATH_MANIFEST"] = str(manifest)
    os.environ["A5_CCU_EXPLICIT_MULTIPATH_PLAN_ID"] = args.plan_id

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("puncture experiment requires exactly two ranks")

    torch.npu.set_device(local_rank)
    init_process_group(rank, world_size)
    buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)

    elements = args.bytes // 4
    send = torch.stack([
        torch.full((elements,), float(rank * 100 + dst + 1),
                   dtype=torch.float32, device="npu")
        for dst in range(world_size)
    ])
    expected = torch.stack([
        torch.full((elements,), float(src * 100 + rank + 1),
                   dtype=torch.float32, device="npu")
        for src in range(world_size)
    ])
    direct_word = encode_path_policy(args.direct_weights)
    direct_ready_word = encode_path_policy(args.direct_weights, ready=True)
    scheduled_ready_word = encode_path_policy(args.scheduled_weights, ready=True)
    path_policy = torch.tensor([direct_word], dtype=torch.int64, device="npu")
    observed_policy = torch.empty_like(path_policy)

    capture_stream = torch_npu.npu.Stream()
    capture_stream.wait_stream(torch_npu.npu.current_stream())
    scheduler = deep_ep.A5ValueWaitCallbackScheduler(path_policy)

    # Prime HCCL/CCU lazy resources on the exact graph stream.  READY is set
    # only for this warmup; the graph writes the non-ready direct word at end.
    with torch_npu.npu.stream(capture_stream):
        path_policy.fill_(direct_ready_word)
        recv = buffer.explicit_multipath_all2all_ccu(send, path_policy, args.plan_id)
        scheduler.enqueue_value_write(direct_word)
    torch.npu.synchronize()
    torch.testing.assert_close(recv, expected, rtol=0, atol=0)
    print(f"CASE_WARMUP rank={rank} policy=direct weights={args.direct_weights} PASS",
          flush=True)

    # The captured graph contains only wait -> operator -> reset.  Host
    # callbacks and the scheduler's ValueWrite stay outside capture.
    with torch_npu.npu.stream(capture_stream):
        path_policy.fill_(direct_ready_word)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch_npu.npu.graph(graph, stream=capture_stream,
                             auto_dispatch_capture=True):
        scheduler.enqueue_value_wait(PATH_POLICY_READY_MASK)
        recv = buffer.explicit_multipath_all2all_ccu(send, path_policy, args.plan_id)
        observed_policy.copy_(path_policy)
        scheduler.enqueue_value_write(direct_word)
    torch.npu.synchronize()
    torch.testing.assert_close(recv, expected, rtol=0, atol=0)
    # Capture implementations differ on whether the captured work is also
    # executed once.  Force READY low after capture so the first replay cannot
    # bypass the scheduler using the capture seed value.
    with torch_npu.npu.stream(capture_stream):
        scheduler.enqueue_value_write(direct_word)
    torch.npu.synchronize()
    print(f"CASE_GRAPH_CAPTURE rank={rank} PASS", flush=True)

    def replay_stage(name, release_word, weights, delta):
        nonlocal expected
        send.add_(delta)
        expected = expected + delta
        capture_stream.wait_stream(torch_npu.npu.current_stream())
        scheduler.arm(release_word)
        with torch_npu.npu.stream(capture_stream):
            scheduler.enqueue_arrived_callback()
            graph.replay()
            scheduler.enqueue_done_callback()
        scheduler.wait_done(args.timeout_ms)
        torch.npu.synchronize()
        torch.testing.assert_close(recv, expected, rtol=0, atol=0)
        observed_word = int(observed_policy.cpu().item())
        if observed_word != release_word:
            raise RuntimeError(
                f"captured operator window observed 0x{observed_word:016x}, "
                f"expected 0x{release_word:016x}"
            )
        print(
            f"CASE_STAGE rank={rank} name={name} weights={weights} "
            f"arrival_count={scheduler.arrival_count} "
            f"release_count={scheduler.release_count} "
            f"done_count={scheduler.done_count} "
            f"observed_policy=0x{observed_word:016x} PASS",
            flush=True,
        )

    # Stage 1 proves the default direct-only decision passes through the same
    # callback/ValueWrite control plane. Stage 2 changes only the release word
    # and enables the two pre-provisioned relay paths without re-entering graph.
    replay_stage("direct_only", direct_ready_word, args.direct_weights, 7)
    replay_stage("direct_plus_relay", scheduled_ready_word,
                 args.scheduled_weights, 11)

    if scheduler.arrival_count != 2 or scheduler.release_count != 2 or scheduler.done_count != 2:
        raise RuntimeError("scheduler callback accounting mismatch")
    dist.barrier()
    if rank == 0:
        print(
            "PASS: ValueWait gate stayed in graph; host callbacks observed arrival/completion; "
            f"ValueWrite switched policy {args.direct_weights} -> {args.scheduled_weights}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
