#!/usr/bin/env python3
"""Stage 2: verify communication-rank Write + Read through explicit relay ranks."""

import argparse
import ctypes
import faulthandler
import os
import site
import sys
import time
from pathlib import Path


# This operator needs MTE/URMA windows, not the CCU scheduler used by stage 1.
os.environ.pop("HCCL_OP_EXPANSION_MODE", None)
os.environ["DEEP_USE_MODE"] = "default"
os.environ.setdefault("HCCL_BUFFSIZE", "2300")


def prepend_env_path(name, path):
    """Prepend one path idempotently and remove duplicates from the variable."""
    current = [value for value in os.environ.get(name, "").split(":") if value]
    values = [path]
    values.extend(value for value in current if value != path and value not in values)
    updated = ":".join(values)
    changed = os.environ.get(name, "") != updated
    os.environ[name] = updated
    return changed


def prepare_custom_op_runtime():
    root = Path(__file__).resolve().parents[3]
    candidates = [root / "python" / "deep_ep" / "deep_ep"]
    candidates.extend(Path(path) / "deep_ep" for path in site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site) / "deep_ep")
    for package in candidates:
        extensions = sorted(package.glob("deep_ep_cpp*.so"))
        op_api = package / "vendors" / "hwcomputing" / "op_api" / "lib" / "libcust_opapi.so"
        if not extensions or not op_api.is_file():
            continue
        vendor = package / "vendors" / "hwcomputing"
        env_changed = prepend_env_path("ASCEND_CUSTOM_OPP_PATH", str(vendor))
        env_changed |= prepend_env_path("LD_LIBRARY_PATH", str(op_api.parent))
        if env_changed:
            if os.environ.get("_DEEPEP_CUSTOM_OP_RUNTIME_REEXEC") == "1":
                raise RuntimeError("custom-op environment remained unstable after one re-exec")
            env = os.environ.copy()
            env["_DEEPEP_CUSTOM_OP_RUNTIME_REEXEC"] = "1"
            os.execvpe(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
        ctypes.CDLL(str(op_api), mode=ctypes.RTLD_GLOBAL)
        sys.path.insert(0, str(package.parent))
        sys.path.insert(0, str(package))
        return extensions[0], op_api
    raise RuntimeError("deep_ep_cpp or libcust_opapi.so not found; install the newest wheel first")


def parse_comm_ranks(text, world_size):
    ranks = [int(value) for value in text.split(",")]
    if ranks != sorted(set(ranks)) or len(ranks) < 2:
        raise ValueError("--comm-ranks must be sorted, unique, and contain at least two ranks")
    if ranks[0] < 0 or ranks[-1] >= world_size:
        raise ValueError("--comm-ranks contains a rank outside WORLD_SIZE")
    if len(ranks) >= world_size:
        raise ValueError("AIV+URMA detour needs at least one rank outside --comm-ranks")
    return ranks


EXTENSION, OP_API = prepare_custom_op_runtime()

import torch
import torch.distributed as dist
import deep_ep


START = time.monotonic()


def stage(rank, message):
    print(f"[rank{rank}] +{time.monotonic() - START:7.3f}s {message}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comm-ranks", default="0,1")
    parser.add_argument("--elements-per-peer", type=int, default=1792)
    args = parser.parse_args()

    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(60, repeat=False)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    comm_ranks = parse_comm_ranks(args.comm_ranks, world_size)

    stage(rank, f"set_device({local_rank}) begin")
    torch.npu.set_device(local_rank)
    stage(rank, "init_process_group(hccl) begin")
    dist.init_process_group("hccl")
    group = dist.new_group(list(range(world_size)))
    dist.barrier(group=group)
    stage(rank, "preflight HCCL barrier done")

    index = torch.arange(args.elements_per_peer, dtype=torch.int32, device="npu")
    send_data = torch.stack(
        [rank * 100000 + dst * 1000 + index for dst in comm_ranks]
    ).contiguous()
    comm_rank_ids = torch.tensor(comm_ranks, dtype=torch.int32, device="npu")
    stage(rank, "deep_ep.Buffer begin")
    buffer = deep_ep.Buffer(group, num_nvl_bytes=0, num_rdma_bytes=0)
    stage(rank, "AIV+URMA detour enqueue begin")
    recv_data = buffer.all2_all_detour_io_die(send_data, comm_rank_ids)
    stage(rank, "enqueue done; device synchronize begin")
    torch.npu.synchronize()
    stage(rank, "device synchronize done")

    if rank in comm_ranks:
        expected = torch.stack(
            [src * 100000 + rank * 1000 + index for src in comm_ranks]
        )
        torch.testing.assert_close(recv_data, expected)
        stage(rank, "correctness check done")

    dist.barrier(group=group)
    if rank == 0:
        relay_ranks = [value for value in range(world_size) if value not in comm_ranks]
        print(f"Using deep_ep_cpp: {EXTENSION}", flush=True)
        print(f"Using custom op API: {OP_API}", flush=True)
        print("PASS: A5 AIV+URMA AllToAll detour (communication-rank Write + Read)", flush=True)
        print(f"comm_ranks={comm_ranks}, relay_ranks={relay_ranks}", flush=True)
        print(f"elements_per_peer={args.elements_per_peer}", flush=True)
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
