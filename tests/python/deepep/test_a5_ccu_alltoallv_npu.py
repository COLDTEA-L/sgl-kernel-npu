#!/usr/bin/env python3
"""EP8 eager correctness gate. Requires eight NPUs and matching rebuilt SOs."""
import argparse
import ctypes
from datetime import timedelta
import json
import os
import statistics
import time
from pathlib import Path


def count_matrix(case, epoch):
    if case == 0:
        return [[0] * 8 for _ in range(8)]
    if case == 1:
        return [[s + 1 if s == d else 0 for d in range(8)] for s in range(8)]
    if case == 2:
        result = [[0] * 8 for _ in range(8)]
        result[2][5], result[3][3] = 13, 7
        return result
    result = [[(s * 17 + d * 7 + epoch * 3) % 11 for d in range(8)] for s in range(8)]
    result[7] = [0] * 8
    for row in result:
        row[6] = 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32", "int8", "int32"), default="float32")
    parser.add_argument("--width", type=int, default=257)
    parser.add_argument("--row-scale", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true",
                        help="time the scaled asymmetric case; no profiler trace")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.row_scale < 1 or args.warmup < 0 or args.iters < 1 or args.rounds < 1:
        parser.error("positive row-scale/iters/rounds and nonnegative warmup required")
    if args.width < 1 or not args.manifest.is_file() or int(os.environ.get("WORLD_SIZE", 0)) != 8:
        parser.error("positive width, an existing manifest and WORLD_SIZE=8 are required")
    os.environ["A5_CCU_CHANNEL_SYNC_MODE"] = "baseline"
    from a5_ccu_test_runtime import (prepare_runtime, import_deep_ep, ccu_pg_kwargs,
                                    log_ccu_communicator)
    extension_path, route_path = prepare_runtime(__file__)
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    deep_ep, _ = import_deep_ep(extension_path)
    route = ctypes.CDLL(str(route_path), mode=ctypes.RTLD_GLOBAL)
    bridge = ctypes.CDLL(str(extension_path))
    for lib, symbol in ((route, "A5CcuAllToAllVAbiVersion"), (bridge, "A5DeepEpAllToAllVAbiVersion")):
        marker = getattr(lib, symbol)
        marker.restype = ctypes.c_int
        if marker() != 1:
            raise RuntimeError(f"wrong AllToAllV ABI: {symbol}")
    destroy = route.HcclCcuUrmaPeerPlanDestroy
    destroy.argtypes = [ctypes.c_uint64]
    destroy.restype = ctypes.c_int
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    dist.init_process_group("hccl", timeout=timedelta(seconds=180),
                            **ccu_pg_kwargs(torch_npu))
    buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
    log_ccu_communicator(buffer.moe_all_to_all_group_name)
    handle = buffer.prepare_ccu_urma_peer_plan_alltoallv("ep8-direct", str(args.manifest.resolve()))
    dtype = getattr(torch, args.dtype)
    guard = 37
    checked = 0
    timings = {}
    traffic = {}
    for epoch in range(1 if args.benchmark else 3):
        for case in ([3] if args.benchmark else range(4)):
            matrix = [[n * args.row_scale for n in row] for row in count_matrix(case, epoch)]
            counts = [c for row in matrix for c in row]
            sends = matrix[rank]
            recvs = [matrix[s][rank] for s in range(8)]

            def payload(source):
                rows = sum(matrix[source])
                values = torch.arange(rows * args.width, dtype=torch.int64).reshape(rows, args.width)
                return ((values * 7 + source * 11 + epoch * 13 + case) % 97).to(dtype)

            source = payload(rank)
            send_storage = torch.full((source.numel() + 2 * guard,), -19, device="npu", dtype=dtype)
            recv_storage = torch.full((sum(recvs) * args.width + 2 * guard,), -17, device="npu", dtype=dtype)
            send = send_storage[guard:guard + source.numel()].reshape(sum(sends), args.width)
            recv = recv_storage[guard:guard + sum(recvs) * args.width].reshape(sum(recvs), args.width)
            send.copy_(source)
            input_snapshot = send_storage.cpu()
            dist.barrier()
            buffer.ccu_urma_peer_plan_alltoallv_out(send, recv, handle, counts)
            torch.npu.synchronize()
            pieces = []
            for s in range(8):
                start = sum(matrix[s][:rank])
                pieces.append(payload(s)[start:start + matrix[s][rank]])
            expected = torch.cat(pieces, dim=0)
            if not torch.equal(recv.cpu(), expected):
                raise AssertionError(f"CCU mismatch rank={rank} epoch={epoch} case={case}")
            if not torch.equal(send_storage.cpu(), input_snapshot):
                raise AssertionError("input or input guards modified")
            output_snapshot = recv_storage.cpu()
            if not ((output_snapshot[:guard] == -17).all() and (output_snapshot[-guard:] == -17).all()):
                raise AssertionError("output guard modified")
            # Native reference on the same expanded row layout, including zeros.
            native = torch.empty_like(recv)
            dist.all_to_all_single(native, send, output_split_sizes=recvs, input_split_sizes=sends)
            torch.npu.synchronize()
            if not torch.equal(native.cpu(), expected):
                raise AssertionError("native AllToAllV reference mismatch")
            dist.barrier()
            checked += 1
            if args.benchmark:
                row_bytes = args.width * send.element_size()
                traffic = dict(send_bytes=sum(sends) * row_bytes,
                               recv_bytes=sum(recvs) * row_bytes,
                               self_bytes=matrix[rank][rank] * row_bytes,
                               network_bytes=sum(matrix[s][d] for s in range(8)
                                                 for d in range(8) if s != d) * row_bytes,
                               max_flow_bytes=max(matrix[s][d] for s in range(8)
                                                  for d in range(8) if s != d) * row_bytes)
                operations = {
                    "custom": lambda: buffer.ccu_urma_peer_plan_alltoallv_out(send, recv, handle, counts),
                    "native": lambda: dist.all_to_all_single(native, send,
                        output_split_sizes=recvs, input_split_sizes=sends),
                }
                timings = {name: [] for name in operations}
                for round_id in range(args.rounds):
                    # Reverse order each round to reduce a fixed ordering bias.
                    names = ("custom", "native") if round_id % 2 == 0 else ("native", "custom")
                    for name in names:
                        op = operations[name]
                        output = recv if name == "custom" else native
                        for _ in range(args.warmup):
                            op()
                        torch.npu.synchronize()
                        # New payload and poisoned output avoid accepting a stale warmup result.
                        send.copy_(source + ((round_id + 1) % 7))
                        output.fill_(-17)
                        torch.npu.synchronize()
                        dist.barrier()
                        torch.npu.synchronize()
                        begin = time.perf_counter()
                        for _ in range(args.iters):
                            op()
                        torch.npu.synchronize()
                        elapsed_us = (time.perf_counter() - begin) * 1e6 / args.iters
                        timings[name].append(elapsed_us)
                        if not torch.equal(output.cpu(), expected + ((round_id + 1) % 7)):
                            raise AssertionError(f"timed {name} mismatch rank={rank} round={round_id}")
                        if not torch.equal(send_storage.cpu()[guard:-guard],
                                           (source + ((round_id + 1) % 7)).flatten()):
                            raise AssertionError("timed input modified")
                        for storage, sentinel in ((send_storage, -19), (recv_storage, -17)):
                            snapshot = storage.cpu()
                            if not ((snapshot[:guard] == sentinel).all() and
                                    (snapshot[-guard:] == sentinel).all()):
                                raise AssertionError("timed guard modified")
                        dist.barrier()
    torch.npu.synchronize()
    if destroy(handle) != 0:
        raise RuntimeError("plan destroy failed")
    dist.barrier()
    dist.destroy_process_group()
    result = dict(rank=rank, ranks=8, dtype=args.dtype, correctness="PASS", cases=checked,
                  backend="eager", native_reference=True)
    if args.benchmark:
        result.update(width=args.width, row_scale=args.row_scale, traffic=traffic,
                      warmup=args.warmup, iters=args.iters, rounds=args.rounds,
                      timing_scope="host_wall_batched_submit_and_device_completion",
                      timings={name: dict(round_us=values, mean_us=statistics.mean(values))
                               for name, values in timings.items()})
    destination = args.run_dir / f"rank{rank}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result) + "\n")
    temporary.replace(destination)
    print("ALLTOALLV_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
