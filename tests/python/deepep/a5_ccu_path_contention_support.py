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

HCCL_BUFFER_MIN_MB = 200
HCCL_BUFFER_HEADROOM_MB = 64


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


def resolve_hccl_buffer_mb(bytes_per_peer, setting="auto", world_size=2):
    """Bound the per-domain HCCL buffer while covering one rank's payload.

    The prepared operator transfers directly between user tensors, so this is
    a conservative initialization budget rather than a staging allocation.
    """
    if not isinstance(bytes_per_peer, int) or bytes_per_peer <= 0:
        raise ValueError("bytes_per_peer must be a positive integer")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if setting != "auto":
        text = str(setting)
        if not text.isdecimal() or int(text) <= 0:
            raise ValueError("HCCL buffer setting must be auto or a positive MB integer")
        return int(text)
    mib = 1024 * 1024
    rank_payload_mb = (world_size * bytes_per_peer + mib - 1) // mib
    return max(HCCL_BUFFER_MIN_MB, rank_payload_mb + HCCL_BUFFER_HEADROOM_MB)


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
