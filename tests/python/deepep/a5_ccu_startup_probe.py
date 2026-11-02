"""Instrument startup collectives without changing the communication backend."""
from datetime import datetime, timedelta
import json
import time


def make_phase_logger(path, context, last_phase):
    """Merge stage fields explicitly; tensor dtype can differ from payload dtype.

    Pure Python so the real logger and collective probe can be tested without NPU.
    The caller supplies payload_dtype separately to retain both dtype meanings.
    """
    context = dict(context)

    def phase(name, **fields):
        last_phase.clear()
        last_phase.update(phase=name, register_stage=context.get('register_stage', 'full'))
        last_phase.update(fields)
        record = dict(context)
        record.update(fields)
        record.update(phase=name, time=datetime.now().astimezone().isoformat(timespec='microseconds'))
        line = json.dumps(record)
        with path.open('a') as journal:
            journal.write(line + '\n')
        print('EP8_EID_PHASE ' + line, flush=True)

    return phase


def traced_all_gather(torch, dist, output, input_tensor, phase, prefix='config_allgather'):
    stream = torch.npu.current_stream()
    fields = dict(dtype=str(input_tensor.dtype), shape=list(input_tensor.shape),
                  output_shapes=[list(x.shape) for x in output], async_op=True,
                  input_bytes=input_tensor.numel() * input_tensor.element_size(),
                  output_bytes=sum(x.numel() * x.element_size() for x in output),
                  caller_stream=str(getattr(stream, 'npu_stream', stream)),
                  caller_stream_repr=str(stream))
    start = time.perf_counter()
    stage = 'submit'
    try:
        phase(prefix + '_begin', **fields)
        work = dist.all_gather(output, input_tensor, async_op=True)
        phase(prefix + '_submitted', submit_us=(time.perf_counter()-start)*1e6)
        stage = 'wait'
        phase(prefix + '_wait_begin')
        wait_start = time.perf_counter()
        if work.wait(timeout=timedelta(seconds=180)) is False:
            raise RuntimeError('HCCL Work.wait returned False')
        phase(prefix + '_wait_complete', wait_us=(time.perf_counter()-wait_start)*1e6)
        # A successful Work.wait can establish stream ordering rather than host
        # completion. Keep the explicit device fence and log it separately.
        stage = 'device_sync'
        phase(prefix + '_sync_begin')
        sync_start = time.perf_counter()
        torch.npu.synchronize()
        phase(prefix + '_sync_complete', sync_us=(time.perf_counter()-sync_start)*1e6)
        phase(prefix + '_complete', total_us=(time.perf_counter()-start)*1e6)
    except Exception as error:
        phase(prefix + '_failed', stage=stage, error_type=type(error).__name__,
              error=str(error), total_us=(time.perf_counter()-start)*1e6)
        raise
