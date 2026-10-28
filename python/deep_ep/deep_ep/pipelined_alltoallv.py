"""EP8 direct + two independently scheduled relay legs, with bounded double buffering.

A chunk is a round of <= chunk_rows rows PER original relayed flow. The existing
CCU collective has a per-round all-peer done handshake, not per-packet readiness.
No direct completion or full-message first-hop barrier gates second-hop launch.
"""
from dataclasses import dataclass
from math import prod
try:
    from .heuristic_alltoallv import N, _uint, _offsets, _coalesce_spans, build_schedule
except ImportError:  # repository test workers also import this module directly
    from heuristic_alltoallv import N, _uint, _offsets, _coalesce_spans, build_schedule


@dataclass(frozen=True)
class RelayPiece:
    src: int
    dst: int
    relay: int
    rows: int
    source: int
    final: int
    first_send: int
    first_recv: int
    second_send: int
    second_recv: int


@dataclass(frozen=True)
class Chunk:
    first_counts: tuple
    second_counts: tuple
    pieces: tuple

    def sizes(self, rank):
        return (sum(self.first_counts[rank * N:(rank + 1) * N]), sum(self.first_counts[rank::N]),
                sum(self.second_counts[rank * N:(rank + 1) * N]), sum(self.second_counts[rank::N]))


@dataclass(frozen=True)
class PipelineSchedule:
    plan: object
    chunk_rows: int
    direct_counts: tuple
    direct_pieces: tuple  # src, dst, rows, original source/final, packed source/final
    chunks: tuple

    def pool_sizes(self, rank):
        return tuple(max((c.sizes(rank)[i] for c in self.chunks), default=0) for i in range(4))

    def scratch_rows(self, rank):
        if not self.plan.uses_relay:
            return 0
        return (sum(self.direct_counts[rank * N:(rank + 1) * N]) + sum(self.direct_counts[rank::N])
                + 2 * sum(self.pool_sizes(rank)))

    def report(self):
        return dict(executor='participant_hbm_pipelined_ccu', chunks=len(self.chunks),
                    chunk_rows_per_flow=self.chunk_rows,
                    chunk_bytes_per_flow=self.chunk_rows * self.plan.row_bytes,
                    ccu_calls_per_execute=1 + 2 * len(self.chunks),
                    buffer_slots=2, readiness='first_hop_chunk_collective_completion',
                    scratch_bytes_per_rank=[self.scratch_rows(r) * self.plan.row_bytes for r in range(N)])


def build_pipeline_schedule(plan, *, chunk_bytes=2 << 20, max_chunks=128):
    # Reuse the serial planner's validation, but NOT its two-stage layout.
    build_schedule(plan)
    rows = max(1, _uint(chunk_bytes, 'chunk_bytes', 1) // plan.row_bytes)
    limit = _uint(max_chunks, 'max_chunks', 1)
    count = (max(plan.relay_rows) + rows - 1) // rows
    if count > limit:
        raise ValueError('relay chunk count exceeds max_chunks; increase chunk_bytes or max_chunks on all ranks')
    source, final = _offsets(plan.counts)
    direct = tuple(n - x for n, x in zip(plan.counts, plan.relay_rows))
    ds, dr = _offsets(direct)
    direct_pieces = tuple((f // N, f % N, n, source[f], final[f], ds[f], dr[f])
                          for f, n in enumerate(direct) if n)
    chunks = []
    for j in range(count):
        first, second, raw = [0] * 64, [0] * 64, []
        for f, total in enumerate(plan.relay_rows):
            n = min(rows, max(0, total - j * rows))
            if not n:
                continue
            s, d, k = f // N, f % N, plan.relay_rank[f]
            offset = direct[f] + j * rows
            raw.append((s, d, k, n, source[f] + offset, final[f] + offset))
            first[s * N + k] += n
            second[k * N + d] += n
        fs, fr = _offsets(first); ss, sr = _offsets(second)
        fu, su, pieces = [0] * 64, [0] * 64, []
        for s, d, k, n, orig, dest in raw:
            a, b = s * N + k, k * N + d
            pieces.append(RelayPiece(s, d, k, n, orig, dest, fs[a] + fu[a], fr[a] + fu[a],
                                     ss[b] + su[b], sr[b] + su[b]))
            fu[a] += n; su[b] += n
        chunks.append(Chunk(tuple(first), tuple(second), tuple(pieces)))
    return PipelineSchedule(plan, rows, direct, direct_pieces, tuple(chunks))


@dataclass
class PipelineLane:
    buffer: object
    handle: int
    stream: object


class _NpuRuntime:
    def __init__(self):
        import torch
        self.torch = torch

    def current(self): return self.torch.npu.current_stream()
    def stream(self, stream): return self.torch.npu.stream(stream)
    def event(self): return self.torch.npu.Event()
    def record(self, event, stream): event.record(stream)
    def wait(self, stream, event): stream.wait_event(event)
    def retain(self, tensor, stream):
        if tensor.numel(): tensor.record_stream(stream)
    def copy(self, dst, di, src, si, n): dst.narrow(0, di, n).copy_(src.narrow(0, si, n))
    def transfer(self, lane, send, recv, counts):
        lane.buffer.ccu_urma_peer_plan_alltoallv_out(send, recv, lane.handle, counts)


class PipelineContext:
    """Persistent lane resources, created once, outside per-batch planning.

    Three distinct HCCL communicators isolate ADDRESS/TOKEN/DONE slots. All lanes
    must contain the same eight ranks in the same order and use the same manifest.
    The direct lane is the caller stream; both relay streams join back onto it.
    A context is exclusive to that caller stream; no concurrent host submissions.
    """
    def __init__(self, lanes, *, _runtime=None):
        if len(lanes) != 3:
            raise ValueError('requires direct, first-hop and second-hop lanes')
        names = [x.buffer.moe_all_to_all_group_name for x in lanes]
        if any(not name for name in names) or len(set(names)) != 3:
            raise ValueError('pipeline requires three distinct HCCL communicator names')
        rank = _uint(lanes[0].buffer.rank, 'rank')
        if rank >= N:
            raise ValueError('pipeline rank must be in the EP8 group')
        if any(x.buffer.rank != rank or x.buffer.group_size != N or x.handle <= 0 for x in lanes):
            raise ValueError('pipeline lanes must be matching EP8 ranks and valid handles')
        if any(lanes[i].stream == lanes[j].stream for i in range(3) for j in range(i)):
            raise ValueError('pipeline requires three distinct streams')
        self.lanes, self.rank = tuple(lanes), rank
        self.runtime = _runtime or _NpuRuntime()
        self.owned_groups = []
        self.closed = False

    @classmethod
    def create(cls, direct_buffer, direct_handle, manifest, *, group_ranks, pg_kwargs=None):
        """Collective setup: all eight EP members create the two groups in the same order.

        group_ranks are WORLD ranks in EP communicator order, not physical IDs.
        Caller owns direct_buffer/handle. This context owns only the two new groups.
        """
        import torch
        import torch.distributed as dist
        if len(group_ranks) != N or len(set(group_ranks)) != N:
            raise ValueError('group_ranks must contain eight distinct WORLD ranks')
        # new_group uses sorted WORLD rank order. Reject silent reordering.
        if list(group_ranks) != sorted(group_ranks):
            raise ValueError('group_ranks must be sorted to preserve PyTorch new_group ordering')
        if dist.get_process_group_ranks(direct_buffer.group) != list(group_ranks):
            raise ValueError('direct communicator membership/order differs from group_ranks')
        lanes = [PipelineLane(direct_buffer, direct_handle, torch.npu.current_stream())]
        groups = []
        options = dict(pg_kwargs or {})
        options.setdefault('use_local_synchronization', True)
        for name in ('first', 'second'):
            group = dist.new_group(ranks=list(group_ranks), backend='hccl', **options)
            groups.append(group)
            buffer = type(direct_buffer)(group, num_nvl_bytes=0, num_rdma_bytes=0)
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                handle = buffer.prepare_ccu_urma_peer_plan_alltoallv('ep8-relay-' + name, str(manifest))
            lanes.append(PipelineLane(buffer, handle, stream))
        context = cls(lanes)
        context.owned_groups = groups
        return context

    def close(self, destroy_handle):
        """Explicit teardown after completion; destroy_handle must check errors."""
        import torch.distributed as dist
        if self.closed:
            return
        self.lanes[0].stream.synchronize()  # also waits for both relay joins
        for lane in self.lanes[1:]:
            destroy_handle(lane.handle)
        for group in reversed(self.owned_groups):
            dist.destroy_process_group(group)
        self.owned_groups.clear()
        self.closed = True


class PreparedPipelinedAllToAllV:
    """Per-matrix layout/workspace; reuses persistent context and its communicators.

    execute is asynchronous on the context's caller stream. Source and destination
    may change per invocation but must have matching shape/dtype/device. Reuse on
    that stream is ordered; do not free this object before completion.
    """
    def __init__(self, context, plan, example_send, *, chunk_bytes=2 << 20,
                 max_chunks=128, max_scratch_bytes=1 << 30):
        import torch
        self.context, self.plan, self.rank = context, plan, context.rank
        self.runtime = context.runtime
        if context.closed or self.runtime.current() != context.lanes[0].stream:
            raise ValueError('prepare requires an open context on its caller stream')
        if example_send.dim() < 1 or not example_send.is_contiguous():
            raise ValueError('invalid send shape/strides')
        if example_send.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.int32):
            raise ValueError('unsupported payload dtype')
        self.tail, self.dtype, self.device = tuple(example_send.shape[1:]), example_send.dtype, example_send.device
        if prod(self.tail) * example_send.element_size() != plan.row_bytes:
            raise ValueError('row_bytes does not match payload')
        self.send_rows = sum(plan.counts[self.rank * N:(self.rank + 1) * N])
        self.recv_rows = sum(plan.counts[self.rank::N])
        if example_send.shape[0] != self.send_rows:
            raise ValueError('send rows differ from plan')
        self.schedule = build_pipeline_schedule(plan, chunk_bytes=chunk_bytes, max_chunks=max_chunks)
        limit = _uint(max_scratch_bytes, 'max_scratch_bytes')
        # Same global check on ALL ranks, before any collective is enqueued.
        if any(self.schedule.scratch_rows(r) * plan.row_bytes > limit for r in range(N)):
            raise ValueError('pipeline workspace exceeds max_scratch_bytes on a participant')
        self.uses_relay = plan.uses_relay
        self.decision = dict(policy='force', proposed_relay=plan.uses_relay, executed_relay=plan.uses_relay,
                             reason='pipelined_relay' if plan.uses_relay else 'no_relay_candidate',
                             **self.schedule.report())
        self.counts = list(plan.counts)
        self.start = self.runtime.event()
        self.ready = [self.runtime.event() for _ in self.schedule.chunks]
        self.done = [self.runtime.event() for _ in self.schedule.chunks]
        if not self.uses_relay:
            self.workspace = []
            return
        def alloc(n): return torch.empty((n, *self.tail), dtype=self.dtype, device=self.device)
        dc = self.schedule.direct_counts
        self.direct_send = alloc(sum(dc[self.rank * N:(self.rank + 1) * N]))
        self.direct_recv = alloc(sum(dc[self.rank::N]))
        self.pool = [[alloc(n) for n in self.schedule.pool_sizes(self.rank)] for _ in range(2)]
        self.workspace = [self.direct_send, self.direct_recv] + [v for slot in self.pool for v in slot]
        self.direct_counts = list(dc)
        self.pack_direct = _coalesce_spans([(src, ps, n) for s, d, n, src, dst, ps, pr
                                           in self.schedule.direct_pieces if s == self.rank])
        self.unpack_direct = _coalesce_spans([(pr, dst, n) for s, d, n, src, dst, ps, pr
                                             in self.schedule.direct_pieces if d == self.rank])
        self.chunks = []
        for j, chunk in enumerate(self.schedule.chunks):
            tensors = tuple(v.narrow(0, 0, n) for v, n in zip(self.pool[j % 2], chunk.sizes(self.rank)))
            pack = _coalesce_spans([(p.source, p.first_send, p.rows) for p in chunk.pieces if p.src == self.rank])
            forward = _coalesce_spans([(p.first_recv, p.second_send, p.rows) for p in chunk.pieces if p.relay == self.rank])
            unpack = _coalesce_spans([(p.second_recv, p.final, p.rows) for p in chunk.pieces if p.dst == self.rank])
            self.chunks.append((tensors, list(chunk.first_counts), list(chunk.second_counts), pack, forward, unpack))

    def execute(self, send, recv):
        rt = self.runtime
        direct, first, second = self.context.lanes
        if self.context.closed or rt.current() != direct.stream:
            raise ValueError('pipeline must execute on the context caller stream')
        for tensor, rows in ((send, self.send_rows), (recv, self.recv_rows)):
            if (tuple(tensor.shape) != (rows, *self.tail) or tensor.dtype != self.dtype
                    or tensor.device != self.device or not tensor.is_contiguous()):
                raise ValueError('payload shape/dtype/device differs from prepared pipeline')
        if send.numel() and recv.numel():
            a, b = send.data_ptr(), recv.data_ptr()
            if a < b + recv.numel() * recv.element_size() and b < a + send.numel() * send.element_size():
                raise ValueError('send and receive buffers overlap')
        if not self.uses_relay:
            rt.transfer(direct, send, recv, self.counts)
            return recv
        # Record BEFORE direct packing/communication. Both relay streams wait
        # only for the input producer (and previous invocation's final joins).
        rt.record(self.start, direct.stream)
        rt.wait(first.stream, self.start)
        rt.wait(second.stream, self.start)
        rt.retain(send, first.stream); rt.retain(recv, second.stream)
        for slot in self.pool:
            for tensor in slot:
                rt.retain(tensor, first.stream); rt.retain(tensor, second.stream)
        for src, dst, n in self.pack_direct:
            rt.copy(self.direct_send, dst, send, src, n)
        rt.transfer(direct, self.direct_send, self.direct_recv, self.direct_counts)
        for src, dst, n in self.unpack_direct:
            rt.copy(recv, dst, self.direct_recv, src, n)
        for j, (tensors, fc, sc, pack, forward, unpack) in enumerate(self.chunks):
            fs, fr, ss, sr = tensors
            with rt.stream(first.stream):
                # Slot j%2 is not reusable until its previous second hop AND
                # unpack complete. A different slot permits the overlap.
                if j >= 2:
                    rt.wait(first.stream, self.done[j - 2])
                for src, dst, n in pack:
                    rt.copy(fs, dst, send, src, n)
                rt.transfer(first, fs, fr, fc)
                rt.record(self.ready[j], first.stream)
            with rt.stream(second.stream):
                rt.wait(second.stream, self.ready[j])
                for src, dst, n in forward:
                    rt.copy(ss, dst, fr, src, n)
                rt.transfer(second, ss, sr, sc)
                for src, dst, n in unpack:
                    rt.copy(recv, dst, sr, src, n)
                rt.record(self.done[j], second.stream)
        # Only the consumer joins the two legs. No device-wide synchronize and
        # no dependency from direct completion to relay first/second hop.
        rt.wait(direct.stream, self.done[-1])
        return recv
