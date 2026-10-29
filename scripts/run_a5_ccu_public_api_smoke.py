#!/usr/bin/env python3
"""Run the standalone public-API probe; no torch, extensions or rootinfo changes."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--devices", required=True, help="Two physical IDs, e.g. 2,3")
    p.add_argument("--bytes", type=int, default=4096)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--output-root", type=Path, default=Path("/home/l00934901/profiling"))
    args = p.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        p.error("binary does not exist; build first")
    try:
        devices = [int(x) for x in args.devices.split(",")]
    except ValueError:
        p.error("--devices must contain two integer IDs")
    if len(devices) != 2 or min(devices) < 0 or len(set(devices)) != 2:
        p.error("--devices requires two distinct nonnegative IDs")
    if args.bytes < 64 or args.bytes > 1 << 30 or args.bytes % 64 or args.iters < 4 or args.timeout <= 0:
        p.error("bytes must be a multiple of 64 in [64, 1GiB]; iters >=4; timeout >0")
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="a5-ccu-public-api.", dir=args.output_root))
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    env["HCCL_OP_EXPANSION_MODE"] = "CCU_SCHED"
    env.pop("HCCL_ALGO", None)
    children, logs = [], []
    print("RESULT_DIR:", out, flush=True)
    try:
        for rank in range(2):
            log = (out / f"rank{rank}.log").open("w")
            logs.append(log)
            cmd = [str(binary), str(rank), str(rank), str(out / "root.bin"),
                   str(args.bytes), str(args.iters), "CCU_SCHED"]
            children.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                             env=env, start_new_session=True))
        deadline = time.monotonic() + args.timeout
        while any(c.poll() is None for c in children):
            failed = [c.returncode for c in children if c.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"worker failed: {failed}")
            if time.monotonic() >= deadline:
                raise TimeoutError("worker timeout")
            time.sleep(0.1)
        if any(c.returncode != 0 for c in children):
            raise RuntimeError("worker failed")
        for rank in range(2):
            text = (out / f"rank{rank}.log").read_text(errors="replace")
            marker = f"PUBLIC_CCU_RANK_PASS rank={rank} "
            if marker not in text:
                raise RuntimeError(f"rank {rank} exited without success marker")
        print("PUBLIC_CCU_ALL_RANKS_AND_EXIT_PASS", flush=True)
    finally:
        for c in children:
            if c.poll() is None:
                os.killpg(c.pid, signal.SIGTERM)
        for c in children:
            try:
                c.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(c.pid, signal.SIGKILL)
                c.wait()
        for log in logs:
            log.close()
        for rank in range(len(children)):
            text = (out / f"rank{rank}.log").read_text(errors="replace")
            print(f"--- rank {rank} log tail ---")
            print("\n".join(text.splitlines()[-16:]))


if __name__ == "__main__":
    main()
