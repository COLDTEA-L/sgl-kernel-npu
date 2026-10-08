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
        time.sleep(0.01)


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
