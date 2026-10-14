"""Import-safe helpers for the two-job physical-path contention experiment."""

import json
import os
from pathlib import Path
import re
import time


DEFAULT_SIZES = (
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1 * 1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
)


def format_size_label(byte_count):
    """Return a path-safe binary-size label using the requested KB/MB spelling."""
    if not isinstance(byte_count, int) or byte_count <= 0:
        raise ValueError("byte count must be a positive integer")
    mib = 1024 * 1024
    if byte_count % mib == 0:
        return f"{byte_count // mib}MB"
    if byte_count % 1024 == 0:
        return f"{byte_count // 1024}KB"
    return f"{byte_count}B"


def safe_name(value):
    value = str(value)
    if not value or re.search(r"[^A-Za-z0-9_.-]", value):
        raise ValueError("job name must contain only letters, digits, dot, underscore or dash")
    return value


def parse_sizes(text):
    items = [item.strip() for item in text.split(",")]
    if not items or any(not item or not item.isdecimal() for item in items):
        raise ValueError("sizes must be a comma-separated positive integer list")
    values = [int(item) for item in items]
    if any(value <= 0 or value % 256 for value in values):
        raise ValueError("every size must be a positive multiple of 256 bytes")
    if len(values) != len(set(values)):
        raise ValueError("sizes must not contain duplicates")
    return values


def contention_measurement_start(directory, job_name, rank, world_size=2,
                                 lead_seconds=0.1, timeout=180):
    """Release P/Q's four ranks against one host-monotonic target time."""
    if job_name not in ("P", "Q"):
        raise ValueError("contention synchronization requires job name P or Q")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank is outside world size")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"ready.{job_name}.rank{rank}").touch()
    expected = [root / f"ready.{job}.rank{item}"
                for job in ("P", "Q") for item in range(world_size)]
    deadline = time.monotonic() + timeout
    release = root / "measurement_start_ns"
    if job_name == "P" and rank == 0:
        while not all(path.exists() for path in expected):
            if time.monotonic() >= deadline:
                raise TimeoutError("four-rank contention readiness timed out")
            time.sleep(0.001)
        target = time.monotonic_ns() + int(lead_seconds * 1e9)
        temp = root / f"measurement_start_ns.pid{os.getpid()}.tmp"
        temp.write_text(str(target))
        temp.replace(release)
    while not release.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("contention release timestamp timed out")
        time.sleep(0.001)
    target = int(release.read_text())
    remaining = target - time.monotonic_ns()
    if remaining > 0:
        time.sleep(remaining / 1e9)
    return max(0, time.monotonic_ns() - target) / 1000.0


def write_iteration_trace(directory, job_name, rank, metadata, records):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"rank{rank}.json"
    payload = dict(metadata)
    payload.update(job=job_name, rank=rank, records=records)
    temp = target.with_suffix(f".pid{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(target)
    return target


def overlap_ns(left, right):
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))
