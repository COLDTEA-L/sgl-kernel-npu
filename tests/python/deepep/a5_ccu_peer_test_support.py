"""Import-safe peer-plan test helpers: no CLI, runtime initialization or re-exec."""
import os
from pathlib import Path
import time


def file_barrier(directory, tag, rank, world_size, timeout=180):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{tag}.rank{rank}").touch()
    deadline = time.monotonic() + timeout
    expected = [directory / f"{tag}.rank{item}" for item in range(world_size)]
    while not all(path.exists() for path in expected):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"file barrier {tag} rank={rank}/{world_size} timed out")
        time.sleep(0.001)


def measurement_start(directory, rank, world_size, lead_seconds=0.05):
    """One-host start rendezvous outside timing; avoid a 10ms first-call skew.

    This improves host start alignment, not cycle-level hardware synchrony.
    Return scheduling lateness so first-call outliers remain diagnosable.
    """
    directory = Path(directory)
    file_barrier(directory, "measurement_ready", rank, world_size)
    release = directory / "measurement_start_ns"
    if rank == 0:
        temp = directory / "measurement_start_ns.tmp"
        temp.write_text(str(time.monotonic_ns() + int(lead_seconds * 1e9)))
        temp.replace(release)
    deadline = time.monotonic() + 180
    while not release.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("measurement start rendezvous timed out")
        time.sleep(0.001)
    target = int(release.read_text())
    remaining = target - time.monotonic_ns()
    if remaining > 0:
        time.sleep(remaining / 1e9)
    return max(0, time.monotonic_ns() - target) / 1000


def make_profiler(output_dir, rank):
    # Import only when profiling is actually requested, not on helper import.
    import torch_npu

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        raise RuntimeError(f"profiling directory is not writable: {output_dir}")
    export_types = [torch_npu.profiler.ExportType.Text]
    if hasattr(torch_npu.profiler.ExportType, "Db"):
        export_types.append(torch_npu.profiler.ExportType.Db)
    kwargs = dict(
        export_type=export_types,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
    )
    try:
        config = torch_npu.profiler._ExperimentalConfig(mstx=False, **kwargs)
    except TypeError:
        config = torch_npu.profiler._ExperimentalConfig(msprof_tx=False, **kwargs)
    return torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(output_dir), worker_name=f"rank{rank}"),
        record_shapes=True,
        experimental_config=config,
    )
