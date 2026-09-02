#!/usr/bin/env python3
import argparse
import ctypes
import faulthandler
import os
import site
import sys
import time
from pathlib import Path


def prepare_a5_ccu_runtime():
    expansion_mode = os.environ.get("HCCL_OP_EXPANSION_MODE", "")
    if expansion_mode and expansion_mode != "CCU_SCHED":
        raise RuntimeError(
            "All2AllDetourIoDie requires HCCL_OP_EXPANSION_MODE=CCU_SCHED on A5; "
            f"got {expansion_mode!r}"
        )
    # These variables must be set before torch.distributed creates the HCCL communicator.
    os.environ["HCCL_OP_EXPANSION_MODE"] = "CCU_SCHED"
    os.environ.setdefault("HCCL_BUFFSIZE", "2300")


def candidate_package_dirs():
    project_root = Path(__file__).resolve().parents[3]
    package_dirs = [project_root / "python" / "deep_ep" / "deep_ep"]
    site_dirs = site.getsitepackages()
    user_site = site.getusersitepackages()
    if user_site:
        site_dirs.append(user_site)
    package_dirs.extend(Path(path) / "deep_ep" for path in site_dirs)
    return package_dirs


def prepend_path(value: str, path: Path) -> str:
    path_str = str(path)
    entries = [entry for entry in value.split(":") if entry]
    return ":".join([path_str, *[entry for entry in entries if entry != path_str]])


def prepare_custom_op_runtime():
    selected = None
    for package_dir in candidate_package_dirs():
        extensions = sorted(package_dir.glob("deep_ep_cpp*.so"))
        op_api = package_dir / "vendors" / "hwcomputing" / "op_api" / "lib" / "libcust_opapi.so"
        if extensions and op_api.is_file():
            selected = (package_dir, extensions[0], op_api)
            break
    if selected is None:
        raise RuntimeError("deep_ep_cpp or All2AllDetourIoDie libcust_opapi.so was not found; build the wheel first")

    package_dir, extension, op_api = selected
    vendor_dir = package_dir / "vendors" / "hwcomputing"
    required_env = {
        "ASCEND_CUSTOM_OPP_PATH": prepend_path(os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""), vendor_dir),
        "LD_LIBRARY_PATH": prepend_path(os.environ.get("LD_LIBRARY_PATH", ""), op_api.parent),
    }
    if any(os.environ.get(name, "") != value for name, value in required_env.items()):
        env = os.environ.copy()
        env.update(required_env)
        os.execvpe(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)

    ctypes.CDLL(str(op_api), mode=ctypes.RTLD_GLOBAL)
    return package_dir, extension, op_api


prepare_a5_ccu_runtime()
DEEP_EP_PACKAGE_DIR, DEEP_EP_EXTENSION, CUSTOM_OP_API = prepare_custom_op_runtime()

import torch
import torch.distributed as dist


TEST_START = time.monotonic()


def stage(rank: int, message: str):
    elapsed = time.monotonic() - TEST_START
    print(f"[rank{rank}] +{elapsed:7.3f}s {message}", flush=True)


def import_deep_ep():
    sys.path.insert(0, str(DEEP_EP_PACKAGE_DIR.parent))
    sys.path.insert(0, str(DEEP_EP_PACKAGE_DIR))
    import deep_ep

    if os.environ.get("RANK", "0") == "0":
        print(f"Using deep_ep_cpp: {DEEP_EP_EXTENSION}", flush=True)
        print(f"Using custom op API: {CUSTOM_OP_API}", flush=True)
        print(
            "Using HCCL runtime: "
            f"HCCL_OP_EXPANSION_MODE={os.environ['HCCL_OP_EXPANSION_MODE']}, "
            f"HCCL_BUFFSIZE={os.environ['HCCL_BUFFSIZE']}",
            flush=True,
        )
        print(
            "Using custom collective: full HCCL_CMD_ALLTOALLV via A5 CCU "
            "(direct sendData -> recvData, no windowsIn/windowsOut)",
            flush=True,
        )
    return deep_ep


def parse_comm_ranks(text: str, world_size: int):
    if not text:
        return list(range(world_size))
    ranks = [int(value) for value in text.split(",")]
    if ranks != sorted(set(ranks)):
        raise ValueError("--comm-ranks must be sorted and contain no duplicates")
    if not ranks or ranks[0] < 0 or ranks[-1] >= world_size:
        raise ValueError("--comm-ranks contains a rank outside WORLD_SIZE")
    return ranks


def main():
    parser = argparse.ArgumentParser(description="A5 CCU/IO Die All2All detour correctness test")
    parser.add_argument("--comm-ranks", default="", help="ordered subset, for example 0,2; default is every rank")
    parser.add_argument("--elements-per-peer", type=int, default=256)
    parser.add_argument(
        "--traceback-timeout",
        type=int,
        default=60,
        help="dump every Python thread stack after this many seconds; 0 disables it",
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("Use at least two A5 NPUs")

    faulthandler.enable(all_threads=True)
    if args.traceback_timeout > 0:
        faulthandler.dump_traceback_later(args.traceback_timeout, repeat=False)

    stage(rank, f"set_device({local_rank}) begin")
    torch.npu.set_device(local_rank)
    stage(rank, "set_device done")
    stage(rank, "init_process_group(hccl) begin")
    dist.init_process_group("hccl")
    stage(rank, "init_process_group done")
    stage(rank, "dist.new_group begin")
    group = dist.new_group(list(range(world_size)))
    stage(rank, "dist.new_group done")
    stage(rank, "preflight HCCL barrier begin")
    dist.barrier(group=group)
    stage(rank, "preflight HCCL barrier done")
    comm_ranks = parse_comm_ranks(args.comm_ranks, world_size)
    stage(rank, "import deep_ep begin")
    deep_ep = import_deep_ep()
    stage(rank, "import deep_ep done")

    stage(rank, "create input tensors begin")
    comm_rank_ids = torch.tensor(comm_ranks, dtype=torch.int32, device="npu")
    index = torch.arange(args.elements_per_peer, dtype=torch.int32, device="npu")
    send_data = torch.stack(
        [rank * 100000 + dst_rank * 1000 + index for dst_rank in comm_ranks]
    ).contiguous()
    torch.npu.synchronize()
    stage(rank, "create input tensors done")

    stage(rank, "deep_ep.Buffer begin")
    buffer = deep_ep.Buffer(group, num_nvl_bytes=0, num_rdma_bytes=0)
    stage(rank, "deep_ep.Buffer done")
    stage(rank, "All2AllDetourIoDie enqueue begin")
    recv_data = buffer.all2_all_detour_io_die(send_data, comm_rank_ids)
    stage(rank, "All2AllDetourIoDie enqueue done; device synchronize begin")
    torch.npu.synchronize()
    stage(rank, "device synchronize done")

    if rank in comm_ranks:
        stage(rank, "correctness check begin")
        expected = torch.stack(
            [src_rank * 100000 + rank * 1000 + index for src_rank in comm_ranks]
        )
        torch.testing.assert_close(recv_data, expected)
        stage(rank, "correctness check done")

    stage(rank, "final HCCL barrier begin")
    dist.barrier(group=group)
    stage(rank, "final HCCL barrier done")
    if rank == 0:
        mode = "subset/IO-Die routing" if len(comm_ranks) < world_size else "full group"
        print(f"PASS: A5 All2AllDetourIoDie ({mode})")
        print(f"world_size={world_size}, comm_ranks={comm_ranks}, elements_per_peer={args.elements_per_peer}")
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
