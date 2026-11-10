"""Instrument startup collectives without changing the communication backend."""
from datetime import timedelta
import time


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
