"""EP8 direct + two independently scheduled relay legs, with bounded double buffering.

A chunk is a round of <= chunk_rows rows PER original relayed flow. The existing
CCU collective has a per-round all-peer done handshake, not per-packet readiness.
No direct completion or full-message first-hop barrier gates second-hop launch.
"""
from dataclasses import dataclass, field
from math import prod
import weakref
from typing import NamedTuple
try:
    from .heuristic_alltoallv import N, _uint, _offsets, _coalesce_spans, validate_plan
except ImportError:  # repository test workers also import this module directly
    from heuristic_alltoallv import N, _uint, _offsets, _coalesce_spans, validate_plan

try:
    from .ccu_payload import payload_dtypes, copy_payload_
except ImportError:
    from ccu_payload import payload_dtypes, copy_payload_


class RelayPiece(NamedTuple):
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

    rank_sizes: tuple = field(default=None, repr=False)
    copy_spans: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.rank_sizes is not None:
            return
        object.__setattr__(self, 'rank_sizes', tuple(
            (sum(self.first_counts[r * N:(r + 1) * N]), sum(self.first_counts[r::N]),
             sum(self.second_counts[r * N:(r + 1) * N]), sum(self.second_counts[r::N])) for r in range(N)))

    def sizes(self, rank):
        return self.rank_sizes[rank]


@dataclass(frozen=True)
class PipelineSchedule:
    plan: object
    chunk_rows: int
    direct_counts: tuple
    direct_pieces: tuple  # src, dst, rows, original source/final, packed source/final
    chunks: tuple

    rank_pool_sizes: tuple = field(default=None, repr=False)

    def __post_init__(self):
        if self.rank_pool_sizes is not None:
            return
        object.__setattr__(self, 'rank_pool_sizes', tuple(tuple(
            max((c.sizes(r)[i] for c in self.chunks), default=0) for i in range(4)) for r in range(N)))

    def pool_sizes(self, rank):
        return self.rank_pool_sizes[rank]

    def scratch_rows(self, rank):
        if not self.plan.uses_relay:
            return 0
        return 2 * sum(self.pool_sizes(rank))

    def report(self):
        return dict(executor='participant_hbm_pipelined_ccu', chunks=len(self.chunks),
                    chunk_rows_per_flow=self.chunk_rows,
                    chunk_bytes_per_flow=self.chunk_rows * self.plan.row_bytes,
                    ccu_calls_per_execute=1 + 2 * len(self.chunks),
                    buffer_slots=2, direct_layout='original_peer_prefix_no_pack', readiness='first_hop_chunk_collective_completion',
                    scratch_bytes_per_rank=[self.scratch_rows(r) * self.plan.row_bytes for r in range(N)])


def build_pipeline_schedule(plan, *, chunk_bytes=8 << 20, max_chunks=128, rank=None):
    if rank is not None and _uint(rank, 'rank') >= N:
        raise ValueError('rank must be an EP8 participant')
    if plan._native_layout is not None:
        rows, direct, pieces, chunks, pools = plan._native_layout(
            plan.counts, plan.relay_rank, plan.relay_rows,
            _uint(chunk_bytes, 'chunk_bytes', 1), _uint(max_chunks, 'max_chunks', 1),
            -1 if rank is None else rank)
        return PipelineSchedule(plan, rows, direct, pieces,
                                tuple(Chunk(fc, sc, tuple(RelayPiece(*p) for p in ps), sizes, copies)
                                      for fc, sc, ps, sizes, copies in chunks), pools)
    # Reuse the serial planner's validation, but NOT its two-stage layout.
    validate_plan(plan)
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
        import sys
        self.torch = torch
        module = sys.modules.get('deep_ep_cpp') or sys.modules.get('deep_ep.deep_ep_cpp')
        self.native_copy = getattr(module, 'ep8_copy_spans_out', None)
        self.bind_workspace_views = getattr(module, 'ep8_bind_workspace_views', None)

    def current(self): return self.torch.npu.current_stream()
    def synchronize(self, stream): stream.synchronize()
    def stream(self, stream): return self.torch.npu.stream(stream)
    def event(self): return self.torch.npu.Event()
    def record(self, event, stream): event.record(stream)
    def wait(self, stream, event): stream.wait_event(event)
    def retain(self, tensor, stream):
        if tensor.numel(): tensor.record_stream(stream)
    def copy(self, dst, di, src, si, n): copy_payload_(dst.narrow(0, di, n), src.narrow(0, si, n))
    def copy_spans(self, dst, src, spans): self.native_copy(src, dst, spans)
    def transfer(self, lane, send, recv, counts, storage_counts=None):
        if storage_counts is None:
            lane.buffer.ccu_urma_peer_plan_alltoallv_out(send, recv, lane.handle, counts)
        else:
            lane.buffer.ccu_urma_peer_plan_alltoallv_layout_out(send, recv, lane.handle, counts, storage_counts)


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
        self._prepared = weakref.WeakSet()
        self._arena, self._capacities, self._signature = [], None, None
        self._start, self._ready, self._done = None, [], []
        self.generation = self.allocations = self.event_allocations = 0

    def reserve(self, schedule, example, limit):
        """Bounded high-water workspace/event pool, independent of route-matrix cache.

        Counts/offsets are rebuilt every batch. Reallocation happens only on
        capacity growth, after prior work completes; live prepared plans drop
        old views and rebind lazily. One dtype/row shape per context.
        """
        import torch
        signature = (example.device, example.dtype, tuple(example.shape[1:]))
        if self._signature is not None and self._signature != signature:
            raise ValueError('pipeline context payload dtype/row shape changed; create a new context')
        rb = schedule.plan.row_bytes
        # Most dynamic calls fit the high-water pool. Avoid recreating 8x8
        # capacity lists and rounding each element when nothing must grow.
        if (self._capacities is not None and len(self._ready) >= len(schedule.chunks)
                and all(max(1, n) <= self._capacities[r][i]
                        for r in range(N) for i, n in enumerate(schedule.pool_sizes(r)))):
            if any(sum(row) * rb > limit for row in self._capacities):
                raise ValueError('pipeline context high-water workspace exceeds max_scratch_bytes')
            return
        needed = []
        for r in range(N):
            # Allocate real backing even for zero-row ranks. The bridge uses
            # this stable storage for tokens instead of new empty sentinels.
            row = [*schedule.pool_sizes(r), *schedule.pool_sizes(r)]
            needed.append([max(1, n) for n in row])
        previous = self._capacities or [[0] * 8 for _ in range(N)]
        exact = [[max(a, b) for a, b in zip(old, new)] for old, new in zip(previous, needed)]
        capacities = [[1 << (n - 1).bit_length() for n in row] for row in exact]
        if any(sum(row) * rb > limit for row in capacities):
            capacities = exact
        if any(sum(row) * rb > limit for row in capacities):
            raise ValueError('pipeline context high-water workspace exceeds max_scratch_bytes')
        if capacities != self._capacities:
            if self._arena:
                self.runtime.synchronize(self.lanes[0].stream)
                for prepared in self._prepared:
                    prepared._invalidate_views()
                self._arena = []
            self._arena = [torch.empty((n, *signature[2]), dtype=signature[1], device=signature[0])
                           for n in capacities[self.rank]]
            self.allocations += len(self._arena)
            self.generation += 1
            self._capacities, self._signature = capacities, signature
        if self._start is None:
            self._start = self.runtime.event(); self.event_allocations += 1
        while len(self._ready) < len(schedule.chunks):
            self._ready.append(self.runtime.event()); self._done.append(self.runtime.event())
            self.event_allocations += 2

    def workspace_stats(self):
        return dict(generation=self.generation, tensor_allocations=self.allocations,
                    event_allocations=self.event_allocations,
                    reserved_bytes=sum(t.numel() * t.element_size() for t in self._arena),
                    addresses=[t.data_ptr() for t in self._arena])

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
        for prepared in self._prepared:
            prepared._invalidate_views()
        self._arena = []
        self._start, self._ready, self._done = None, [], []
        self.closed = True


class PreparedPipelinedAllToAllV:
    """Per-matrix layout/workspace; reuses persistent context and its communicators.

    execute is asynchronous on the context's caller stream. Source and destination
    may change per invocation but must have matching shape/dtype/device. Reuse on
    that stream is ordered; do not free this object before completion.
    """
    def __init__(self, context, plan, example_send, *, chunk_bytes=8 << 20,
                 max_chunks=128, max_scratch_bytes=1 << 30):
        import torch
        self.context, self.plan, self.rank = context, plan, context.rank
        self.runtime = context.runtime
        self._native_copies = plan._native_layout is not None
        if self._native_copies and isinstance(self.runtime, _NpuRuntime) and (
                self.runtime.native_copy is None or self.runtime.bind_workspace_views is None):
            raise RuntimeError('rebuild/install DeepEP: native span copies and workspace views required')
        if context.closed or self.runtime.current() != context.lanes[0].stream:
            raise ValueError('prepare requires an open context on its caller stream')
        if example_send.dim() < 1 or not example_send.is_contiguous():
            raise ValueError('invalid send shape/strides')
        if example_send.dtype not in payload_dtypes():
            raise ValueError('unsupported payload dtype')
        self.tail, self.dtype, self.device = tuple(example_send.shape[1:]), example_send.dtype, example_send.device
        if prod(self.tail) * example_send.element_size() != plan.row_bytes:
            raise ValueError('row_bytes does not match payload')
        self.send_rows = sum(plan.counts[self.rank * N:(self.rank + 1) * N])
        self.recv_rows = sum(plan.counts[self.rank::N])
        if example_send.shape[0] != self.send_rows:
            raise ValueError('send rows differ from plan')
        self.schedule = build_pipeline_schedule(plan, chunk_bytes=chunk_bytes, max_chunks=max_chunks, rank=self.rank)
        limit = _uint(max_scratch_bytes, 'max_scratch_bytes')
        # Same global check on ALL ranks, before any collective is enqueued.
        if any(self.schedule.scratch_rows(r) * plan.row_bytes > limit for r in range(N)):
            raise ValueError('pipeline workspace exceeds max_scratch_bytes on a participant')
        self.uses_relay = plan.uses_relay
        self.decision = dict(policy='force', proposed_relay=plan.uses_relay, executed_relay=plan.uses_relay,
                             reason='pipelined_relay' if plan.uses_relay else 'no_relay_candidate',
                             **self.schedule.report())
        self.counts = list(plan.counts)
        self._bound_generation = -1
        if not self.uses_relay:
            self.workspace = []
            return
        context.reserve(self.schedule, example_send, limit)
        context._prepared.add(self)
        self.start = context._start
        self.ready = context._ready[:len(self.schedule.chunks)]
        self.done = context._done[:len(self.schedule.chunks)]
        dc = self.schedule.direct_counts
        self.direct_counts = list(dc)
        self._chunk_specs = []
        for chunk in self.schedule.chunks:
            if chunk.copy_spans is not None:
                pack, forward, unpack = chunk.copy_spans
            else:
                pack = _coalesce_spans([(p.source, p.first_send, p.rows) for p in chunk.pieces if p.src == self.rank])
                forward = _coalesce_spans([(p.first_recv, p.second_send, p.rows) for p in chunk.pieces if p.relay == self.rank])
                unpack = _coalesce_spans([(p.second_recv, p.final, p.rows) for p in chunk.pieces if p.dst == self.rank])
            self._chunk_specs.append((chunk, list(chunk.first_counts), list(chunk.second_counts), pack, forward, unpack))
        self._bind_views()

    def _invalidate_views(self):
        self.pool = self.workspace = self.chunks = []
        self._bound_generation = -1

    def _bind_views(self):
        arena = self.context._arena
        sizes = self.schedule.pool_sizes(self.rank)
        native_bind = getattr(self.runtime, 'bind_workspace_views', None)
        if self._native_copies and native_bind is not None:
            self.workspace, views = native_bind(arena, sizes, [c.sizes(self.rank) for c, *_ in self._chunk_specs])
            self.pool = [self.workspace[:4], self.workspace[4:]]
            self.chunks = [(tuple(tensors), fc, sc, pack, forward, unpack)
                           for tensors, (_, fc, sc, pack, forward, unpack) in zip(views, self._chunk_specs)]
            self._bound_generation = self.context.generation
            return
        self.pool = [[arena[slot * 4 + i].narrow(0, 0, n) for i, n in enumerate(sizes)] for slot in range(2)]
        self.workspace = [v for slot in self.pool for v in slot]
        self.chunks = []
        for j, (chunk, fc, sc, pack, forward, unpack) in enumerate(self._chunk_specs):
            tensors = tuple(v.narrow(0, 0, n) for v, n in zip(self.pool[j % 2], chunk.sizes(self.rank)))
            self.chunks.append((tensors, fc, sc, pack, forward, unpack))
        self._bound_generation = self.context.generation

    def _transfer(self, lane, send, recv, counts, stage, storage_counts=None):
        try:
            self.runtime.transfer(lane, send, recv, counts, storage_counts)
        except RuntimeError as exc:
            raise RuntimeError(f'PIPELINE_TRANSFER_FAILED stage={stage} rank={self.rank} '
                               f'handle={lane.handle} generation={self.context.generation} '
                               f'send_rows={len(send)} recv_rows={len(recv)}') from exc

    def _copy_spans(self, dst, src, spans):
        if not spans:
            return
        if self._native_copies and hasattr(self.runtime, 'copy_spans'):
            self.runtime.copy_spans(dst, src, spans)
        else:  # Python reference and the CPU dependency simulator.
            for si, di, n in spans:
                self.runtime.copy(dst, di, src, si, n)

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
        if self._bound_generation != self.context.generation:
            self._bind_views()
        # Record BEFORE direct packing/communication. Both relay streams wait
        # only for the input producer (and previous invocation's final joins).
        rt.record(self.start, direct.stream)
        rt.wait(first.stream, self.start)
        rt.wait(second.stream, self.start)
        rt.retain(send, first.stream); rt.retain(recv, second.stream)
        # Context owns all pool storage until close or synchronized capacity
        # growth. Do not retire/re-record every internal tensor on both lanes.
        # Send original peer prefixes straight into final output segments.
        # Relay writes disjoint suffixes; no full direct pack/unpack or workspace.
        self._transfer(direct, send, recv, self.direct_counts, "direct", self.counts)
        for j, (tensors, fc, sc, pack, forward, unpack) in enumerate(self.chunks):
            fs, fr, ss, sr = tensors
            with rt.stream(first.stream):
                # Slot j%2 is not reusable until its previous second hop AND
                # unpack complete. A different slot permits the overlap.
                if j >= 2:
                    rt.wait(first.stream, self.done[j - 2])
                self._copy_spans(fs, send, pack)
                self._transfer(first, fs, fr, fc, f"first_chunk_{j}")
                rt.record(self.ready[j], first.stream)
            with rt.stream(second.stream):
                rt.wait(second.stream, self.ready[j])
                self._copy_spans(ss, fr, forward)
                self._transfer(second, ss, sr, sc, f"second_chunk_{j}")
                self._copy_spans(recv, sr, unpack)
                rt.record(self.done[j], second.stream)
        # Only the consumer joins the two legs. No device-wide synchronize and
        # no dependency from direct completion to relay first/second hop.
        rt.wait(direct.stream, self.done[-1])
        return recv
