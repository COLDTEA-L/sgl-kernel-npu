"""Reusable EP8 count exchange. No payload, channel or plan is cached here."""
import threading


class RoutingCountsExchange:
    """One in-flight exchange of current int64[8] counts on an EP8 HCCL group.

    ``begin`` must run on the producer stream after producing counts. ``finish``
    returns a borrowed, contiguous CPU int64[64], valid until the next begin.
    Consume it synchronously (e.g. host.run_counts_tensor), or clone to retain it.
    Never mutate the input between begin and finish. Close before destroying the
    process group/runtime. All ranks must call collectives in the same order.

    dedicated_stream=True lets current-layer packing be submitted between begin
    and finish. It does not predict future counts or remove producer dependencies.
    No new process group is created; the existing all_gather protocol is retained.
    """
    def __init__(self, example, group=None, *, dedicated_stream=False):
        import torch
        import torch.distributed as dist
        self._torch, self._dist = torch, dist
        self.group = group
        self.device = example.device
        if self.device.type != 'npu':
            raise ValueError('requires NPU int64 local_counts[8]')
        self._validate(example)
        if dist.get_world_size(group) != 8:
            raise ValueError('requires an EP8 process group')
        self._lock = threading.Lock()
        self._closed = self._failed = self._pending = False
        self._work = self._input = None
        self._dedicated = bool(dedicated_stream)
        with torch.npu.device(self.device):
            self._stream = torch.npu.Stream() if dedicated_stream else torch.npu.current_stream()
            self._ready = torch.npu.Event() if dedicated_stream else None
            self._done = torch.npu.Event()
            # Allocate on the stream that owns all writes to this reusable buffer.
            with torch.npu.stream(self._stream):
                self._device_counts = torch.empty((8, 8), device=self.device, dtype=torch.int64)
                self._parts = list(self._device_counts.unbind(0))
                self._host = torch.empty((8, 8), device='cpu', dtype=torch.int64, pin_memory=True)
                self._flat_host = self._host.view(64)

    def _validate(self, counts):
        if (counts.device != self.device or counts.dtype != self._torch.int64 or
                tuple(counts.shape) != (8,) or not counts.is_contiguous()):
            raise ValueError('requires contiguous int64 local_counts[8] on original device')

    def _check(self):
        if self._closed or self._failed:
            raise RuntimeError('count exchange closed or failed; recreate context')

    def begin(self, local_counts):
        with self._lock:
            self._check()
            if self._pending:
                raise RuntimeError('finish the previous count exchange before begin')
            self._validate(local_counts)
            torch = self._torch
            with torch.npu.device(self.device):
                producer = torch.npu.current_stream()
                if not self._dedicated and producer != self._stream:
                    raise RuntimeError('reuse exchange requires original caller stream')
                self._pending = True
                self._input = local_counts  # retain through completion, including exceptional exits
                try:
                    if self._dedicated:
                        self._ready.record(producer)
                        self._stream.wait_event(self._ready)
                    with torch.npu.stream(self._stream):
                        self._work = self._dist.all_gather(
                            self._parts, local_counts, group=self.group, async_op=True)
                except BaseException:
                    self._failed = True
                    raise
        return self

    def finish(self):
        with self._lock:
            self._check()
            if not self._pending:
                raise RuntimeError('begin a count exchange before finish')
            try:
                with self._torch.npu.device(self.device), self._torch.npu.stream(self._stream):
                    # HCCL wait establishes completion/order on this caller stream;
                    # the D2H copy must not race the collective's internal stream.
                    self._work.wait()
                    self._host.copy_(self._device_counts, non_blocking=True)
                    self._done.record(self._stream)
                self._done.synchronize()  # CPU must not read an unfinished D2H
            except BaseException:
                self._failed = True
                raise
            self._pending = False
            self._work = self._input = None
            return self._flat_host

    def gather(self, local_counts):
        self.begin(local_counts)
        return self.finish()

    def close(self):
        with self._lock:
            if self._closed:
                return
            # A submitted collective can run on an internal HCCL stream. Join it
            # before releasing input/output storage, even after a partial failure.
            with self._torch.npu.device(self.device), self._torch.npu.stream(self._stream):
                if self._work is not None:
                    self._work.wait()
                self._stream.synchronize()
            self._work = self._input = None
            self._pending = False
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
