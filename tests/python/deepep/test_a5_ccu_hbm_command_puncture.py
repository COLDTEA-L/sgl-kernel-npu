#!/usr/bin/env python3
"""Two-rank AIV -> HBM mailbox -> persistent CCU worker puncture."""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
import deep_ep


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=4194304)
    parser.add_argument("--relay-manifest", required=True)
    parser.add_argument("--direct-route", type=int, default=0)
    parser.add_argument("--path-weights", default="2,1,1")
    parser.add_argument("--plan-id", default="aiv-hbm-ccu-puncture")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--register-only", action="store_true")
    return parser.parse_args()


def check_ack(ack, phase):
    value = int(ack.cpu().item())
    if value != 0:
        raise RuntimeError(f"{phase}: AIV observed worker ack={value}")
    print(f"PUNCTURE_{phase.upper()} rank={dist.get_rank()} PASS", flush=True)


def expected_tensor(rank, elements):
    return torch.stack([
        torch.full((elements,), float(src * 100 + rank + 1),
                   dtype=torch.float32, device="npu")
        for src in range(2)
    ])


def main():
    args = parse_args()
    if args.bytes <= 0 or args.bytes % 256:
        raise RuntimeError("--bytes must be a positive multiple of 256")
    weights = [int(item) for item in args.path_weights.split(",") if item]
    if not 2 <= len(weights) <= 8 or any(value <= 0 for value in weights):
        raise RuntimeError("--path-weights requires 2..8 positive integers")
    manifest = Path(args.relay_manifest).resolve()
    if not manifest.is_file():
        raise RuntimeError(f"relay manifest not found: {manifest}")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError("this puncture requires exactly two ranks")
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)

    elements = args.bytes // 4
    send = torch.stack([
        torch.full((elements,), float(rank * 100 + dst + 1),
                   dtype=torch.float32, device="npu")
        for dst in range(2)
    ])
    recv = torch.full_like(send, -1.0)
    command_block = torch.zeros((64,), dtype=torch.int64, device="npu")
    expected = expected_tensor(rank, elements)
    worker = 0
    try:
        worker = buffer.runtime.prepare_ccu_hbm_command_worker(
            send, recv, command_block, args.plan_id, str(manifest),
            args.direct_route, weights)
        print(f"PUNCTURE_WORKER rank={rank} handle={worker} paths={len(weights)}", flush=True)

        if args.register_only:
            # The route package has registered and finalized the selected
            # instruction group but deliberately did not launch it.
            print(f"PUNCTURE_REGISTER_ONLY rank={rank} PASS", flush=True)
            worker = 0
        else:
            # Stage A: no payload movement. This isolates cache/coherency and the
            # two-way command/completion handshake from URMA communication.
            ack = buffer.runtime.ccu_hbm_command_puncture(
                send, recv, command_block, weights, False)
            torch.npu.synchronize()
            check_ack(ack, "handshake")
            torch.testing.assert_close(recv, torch.full_like(recv, -1.0))

            # Stage B: the exact same worker consumes addresses/offsets/path bytes
            # published by AIV and performs one real explicit multipath AllToAll.
            ack = buffer.runtime.ccu_hbm_command_puncture(
                send, recv, command_block, weights, True)
            torch.npu.synchronize()
            check_ack(ack, "transfer")
            torch.testing.assert_close(recv, expected)

        if args.graph and not args.register_only:
            capture_stream = torch_npu.npu.Stream()
            graph = torch.npu.NPUGraph()
            with torch_npu.npu.stream(capture_stream):
                warmup_ack = buffer.runtime.ccu_hbm_command_puncture(
                    send, recv, command_block, weights, True)
            torch.npu.synchronize()
            check_ack(warmup_ack, "graph_warmup")

            with torch_npu.npu.graph(
                graph, stream=capture_stream, auto_dispatch_capture=True
            ):
                graph_ack = buffer.runtime.ccu_hbm_command_puncture(
                    send, recv, command_block, weights, True)
            torch.npu.synchronize()
            check_ack(graph_ack, "aclgraph_capture")

            send.add_(1000.0)
            expected.add_(1000.0)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            check_ack(graph_ack, "aclgraph_replay")
            torch.testing.assert_close(recv, expected)
    finally:
        if worker:
            buffer.runtime.stop_ccu_hbm_command_worker(worker)
            print(f"PUNCTURE_STOP rank={rank} handle={worker} PASS", flush=True)
        dist.destroy_process_group()

    print("PUNCTURE_RESULT PASS", flush=True)


if __name__ == "__main__":
    main()
