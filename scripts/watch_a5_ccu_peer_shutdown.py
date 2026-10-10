#!/usr/bin/env python3
"""Observe only this benchmark's processes after all rank results are reported.

Default: bounded read-only /proc snapshots. Optional GDB briefly pauses one
matching test process at a time after measurement; no driver/config changes.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def read(path, limit=1024 * 1024):
    try:
        with Path(path).open("rb") as file:
            return file.read(limit).decode("utf-8", errors="replace")
    except OSError as exc:
        return f"UNAVAILABLE: {exc}"


def process_identity(base):
    stat = read(base / "stat")
    try:
        # Fields after ')' begin at state (field 3); starttime is field 22.
        return stat.rsplit(")", 1)[1].split()[19]
    except (IndexError, ValueError):
        return None


def matching_processes(case_dir, proc_root=Path("/proc")):
    expected = str(Path(case_dir).resolve())
    matches = []
    for base in proc_root.iterdir():
        if not base.name.isdigit() or int(base.name) == os.getpid():
            continue
        argv = read(base / "cmdline").split("\0")
        # Exact worker script + exact argument value, not a substring/pgrep of
        # all python processes on this public machine.
        if not any(Path(arg).name == "test_a5_ccu_peer_plan_alltoall.py" for arg in argv if arg):
            continue
        if not any(arg == "--run-dir" and i + 1 < len(argv) and argv[i + 1] == expected
                   for i, arg in enumerate(argv)):
            continue
        matches.append(dict(pid=int(base.name), argv=[arg for arg in argv if arg],
                            starttime=process_identity(base)))
    return matches


def snapshot(target, proc_root=Path("/proc")):
    base = proc_root / str(target["pid"])
    result = dict(target)
    if target["starttime"] is None or process_identity(base) != target["starttime"]:
        result["state"] = "EXITED_OR_IDENTITY_CHANGED"
        return result
    result["status"] = read(base / "status", 16384)
    result["wchan"] = read(base / "wchan", 4096)
    result["kernel_stack"] = read(base / "stack", 16384)
    result["libraries"] = [line for line in read(base / "maps").splitlines()
                           if any(name in line for name in
                                  ("libhcomm", "libhccl", "libascend", "libacl",
                                   "libtorch", "liba5_ccu", "deep_ep_cpp", "libpython"))]
    result["threads"] = []
    try:
        tasks = sorted((base / "task").iterdir(), key=lambda item: int(item.name))
    except OSError as exc:
        result["task_error"] = str(exc)
        return result
    result["thread_count"] = len(tasks)
    for task in tasks[:128]:
        result["threads"].append(dict(tid=int(task.name),
            status=read(task / "status", 16384), wchan=read(task / "wchan", 4096),
            kernel_stack=read(task / "stack", 16384)))
    return result


def native_backtrace(target, output):
    gdb = shutil.which("gdb")
    if not gdb:
        output.write_text("UNAVAILABLE: gdb not installed; no installation attempted\n")
        return
    base = Path("/proc") / str(target["pid"])
    if target["starttime"] is None or process_identity(base) != target["starttime"]:
        output.write_text("SKIPPED: process exited or identity changed\n")
        return
    with output.open("w") as file:
        debugger = subprocess.Popen([gdb, "-q", "-nx", "-batch",
            "-ex", "set auto-load off", "-ex", "set debuginfod enabled off",
            "-ex", "set pagination off", "-p", str(target["pid"]),
            "-ex", "info sharedlibrary", "-ex", "thread apply all bt 12",
            "-ex", "detach"], stdout=file, stderr=subprocess.STDOUT)
        try:
            debugger.wait(timeout=20)
        except subprocess.TimeoutExpired:
            debugger.send_signal(signal.SIGINT)
            try:
                debugger.wait(timeout=5)
            except subprocess.TimeoutExpired:
                debugger.terminate()
                try:
                    debugger.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    debugger.kill()
                    debugger.wait()
            file.write("\nGDB_TIME_BUDGET_EXCEEDED; do not infer a native root cause from incomplete output\n")


def log_state(log, k, case_dir=None):
    results, phases = {}, {}
    # Logs in this short test are normally small; bound reads for a stalled run.
    for line in read(log, 16 * 1024 * 1024).splitlines():
        if line.startswith("RESULT_JSON "):
            try:
                item = json.loads(line[len("RESULT_JSON "):])
                if item.get("correctness") == "PASS" and item.get("ranks") == k:
                    results[item["rank"]] = item
            except (ValueError, TypeError, KeyError, AttributeError):
                pass
        if line.startswith("PEER_CASE_PHASE "):
            fields = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
            if "rank" in fields and "phase" in fields:
                phases[fields["rank"]] = fields["phase"]
    if case_dir is not None and (Path(case_dir) / "results").is_dir():
        results = {}
        for rank in range(k):
            try:
                item = json.loads((Path(case_dir) / "results" / f"rank{rank}.json").read_text())
                if item.get("rank") == rank and item.get("ranks") == k and item.get("correctness") == "PASS":
                    results[rank] = item
            except (OSError, ValueError, TypeError, AttributeError):
                pass
    return set(results) == set(range(k)), phases


def watch(case_dir, log, k, owner_pid, grace, max_seconds, native=False):
    first_report = None
    deadline = time.monotonic() + max_seconds
    status = log.with_suffix(".status")
    while time.monotonic() < deadline:
        if status.exists() or not Path(f"/proc/{owner_pid}").exists():
            return
        complete, phases = log_state(log, k, case_dir)
        if complete:
            if first_report is None:
                first_report = time.monotonic()
            if time.monotonic() - first_report >= grace:
                output = case_dir / "shutdown_trace"
                output.mkdir(parents=True, exist_ok=True)
                targets = matching_processes(case_dir)
                (output / "scope.json").write_text(json.dumps(dict(
                    case_dir=str(case_dir), last_phase_by_rank=phases,
                    processes=targets, native_backtrace_requested=native,
                    note="/proc wchan/stack are kernel waits, not user-space backtraces"), indent=2) + "\n")
                for index in range(3):
                    (output / f"proc_snapshot{index}.json").write_text(
                        json.dumps([snapshot(t) for t in targets], indent=2) + "\n")
                    if index < 2:
                        time.sleep(1)
                if native:
                    for target in targets:
                        native_backtrace(target, output / f"native_pid{target['pid']}.txt")
                print(f"PEER_SHUTDOWN_TRACE {output}", flush=True)
                return
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--ranks", required=True, type=int)
    parser.add_argument("--owner-pid", required=True, type=int)
    parser.add_argument("--grace-seconds", type=float, default=30)
    parser.add_argument("--max-seconds", type=float, default=620)
    parser.add_argument("--native-backtrace", action="store_true")
    args = parser.parse_args()
    watch(args.case_dir.resolve(), args.log.resolve(), args.ranks, args.owner_pid,
          args.grace_seconds, args.max_seconds, args.native_backtrace)
