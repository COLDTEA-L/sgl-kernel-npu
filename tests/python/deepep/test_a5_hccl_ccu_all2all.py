#!/usr/bin/env python3
"""Stage 1: prove that this A5 environment can run HCCL's fixed AllToAll on CCU."""

import ctypes
import faulthandler
import os
import site
import sys
import time
from pathlib import Path


os.environ["HCCL_OP_EXPANSION_MODE"] = "CCU_SCHED"
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


EXTENSION, OP_API = prepare_custom_op_runtime()

import torch
import torch.distributed as dist
import deep_ep


START = time.monotonic()


def stage(rank, message):
    print(f"[rank{rank}] +{time.monotonic() - START:7.3f}s {message}", flush=True)


def main():
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(60, repeat=False)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("HCCL CCU AllToAll test requires at least two ranks")

    stage(rank, f"set_device({local_rank}) begin")
    torch.npu.set_device(local_rank)
    stage(rank, "init_process_group(hccl) begin")
    dist.init_process_group("hccl")
    group = dist.new_group(list(range(world_size)))
    dist.barrier(group=group)
    stage(rank, "preflight HCCL barrier done")

    elements_per_peer = 256
    index = torch.arange(elements_per_peer, dtype=torch.int32, device="npu")
    send_data = torch.stack(
        [rank * 100000 + dst * 1000 + index for dst in range(world_size)]
    ).contiguous()
    stage(rank, "deep_ep.Buffer begin")
    buffer = deep_ep.Buffer(group, num_nvl_bytes=0, num_rdma_bytes=0)
    stage(rank, "HcclAll2AllCcu enqueue begin")
    recv_data = buffer.hccl_all2_all_ccu(send_data)
    stage(rank, "enqueue done; device synchronize begin")
    torch.npu.synchronize()
    stage(rank, "device synchronize done")

    expected = torch.stack(
        [src * 100000 + rank * 1000 + index for src in range(world_size)]
    )
    torch.testing.assert_close(recv_data, expected)
    stage(rank, "correctness check done")
    dist.barrier(group=group)
    if rank == 0:
        print(f"Using deep_ep_cpp: {EXTENSION}", flush=True)
        print(f"Using custom op API: {OP_API}", flush=True)
        print("PASS: fixed HCCL AllToAll executed through A5 CCU", flush=True)
        print(f"world_size={world_size}, elements_per_peer={elements_per_peer}", flush=True)
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
