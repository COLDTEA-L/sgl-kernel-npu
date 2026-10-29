"""EP8 one-relay greedy planner and eager two-stage CCU executor.

This module is import-safe without Torch. Counts are *packed rows*, not expert
hit counts. Every rank supplies the same 8x8 matrix in communicator rank order.
The executor forwards through participant HBM; it is not UB source routing.
"""
from dataclasses import dataclass, field
import hashlib
import json
from math import prod
from numbers import Integral

N = 8
MAX_I64 = (1 << 63) - 1


def _uint(value, name, minimum=0):
    # Device counters converted to host lists contain plain ints. Keep the
    # Integral fallback for NumPy scalars, without its ABC check on every entry.
    if type(value) is int and minimum <= value <= MAX_I64:
        return value
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= int(value) <= MAX_I64:
        raise ValueError(f'{name} must be an integer in [{minimum}, {MAX_I64}]')
    return int(value)


def normalize_counts(counts):
    """Host integers only; GPU-to-host synchronization must be explicit upstream."""
    if not isinstance(counts, (list, tuple)):
        raise ValueError('counts must be a host list/tuple, flat 64 or nested 8x8')
    if len(counts) == 8 and all(isinstance(row, (list, tuple)) and len(row) == 8 for row in counts):
        counts = [v for row in counts for v in row]
    if len(counts) != 64:
        raise ValueError('counts must have exactly 64 entries')
    values = tuple(_uint(v, 'count') for v in counts)
    if sum(values) > MAX_I64:
        raise ValueError('total row count overflows int64')
    return values


def normalize_relays(relays):
    if len(relays) != 64:
        raise ValueError('allowed_relays requires 64 entries')
    result = []
    for f, candidates in enumerate(relays):
        s, d = divmod(f, N)
        if not isinstance(candidates, (list, tuple)):
            raise ValueError('relay candidates must be lists of communicator ranks')
        row = tuple(sorted(_uint(k, 'relay') for k in candidates))
        if len(set(row)) != len(row) or any(k >= N or k in (s, d) for k in row) or (s == d and row):
            raise ValueError('relays must be distinct participant ranks, excluding both endpoints')
        result.append(row)
    return tuple(result)


def cut_lower_bound_rows(counts):
    counts = normalize_counts(counts)
    best, group = 0, 0
    for mask in range(1, 255):
        members = [s for s in range(N) if mask & (1 << s)]
        other = [d for d in range(N) if not mask & (1 << d)]
        rows = sum(counts[s * N + d] for s in members for d in other)
        edges = len(members) * len(other)
        bound = (rows + edges - 1) // edges
        if bound > best:
            best, group = bound, mask
    return best, group


@dataclass(frozen=True)
class RoutePlan:
    counts: tuple
    row_bytes: int
    relay_rank: tuple             # -1 means all direct
    relay_rows: tuple
    link_rows: tuple              # direct + both forwarding legs, excluding self
    actions: tuple                # (src, dst, relay, moved_rows, local_gain_rows)
    candidate_evaluations: int
    stop_reason: str
    _native_layout: object = field(default=None, repr=False, compare=False)

    @property
    def uses_relay(self):
        return any(self.relay_rows)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps((self.counts, self.row_bytes, self.relay_rank,
                                         self.relay_rows)).encode()).hexdigest()

    def report(self):
        original = [v if f // N != f % N else 0 for f, v in enumerate(self.counts)]
        lower, mask = cut_lower_bound_rows(self.counts)
        return dict(fingerprint=self.fingerprint, counts=list(self.counts), row_bytes=self.row_bytes,
                    relay_rank=list(self.relay_rank), relay_rows=list(self.relay_rows),
                    actions=list(self.actions), stop_reason=self.stop_reason,
                    candidate_evaluations=self.candidate_evaluations,
                    direct_peak_bytes=max(original) * self.row_bytes,
                    planned_link_peak_bytes=max(self.link_rows) * self.row_bytes,
                    original_network_bytes=sum(original) * self.row_bytes,
                    relay_bytes=sum(self.relay_rows) * self.row_bytes,
                    transmitted_network_bytes=sum(self.link_rows) * self.row_bytes,
                    cut_lower_bound_bytes=lower * self.row_bytes,
                    bottleneck_cut=[r for r in range(N) if mask & (1 << r)])


class RelayPlanner:
    """Persistent host planner; actual counts may change on EVERY invocation.

    native_module is the loaded deep_ep_cpp extension (or the CPU test module).
    None selects the original Python oracle explicitly, never a silent fallback.
    This object owns only static topology/options, no counts or plan cache.
    """
    def __init__(self, row_bytes, allowed_relays, *, min_chunk_bytes=1, min_gain_bytes=0,
                 max_actions=56, relay_capacity_bytes=None, native_module=None):
        self.row_bytes = _uint(row_bytes, 'row_bytes', 1)
        self.relays = normalize_relays(allowed_relays)
        self.options = dict(min_chunk_bytes=_uint(min_chunk_bytes, 'min_chunk_bytes', 1),
                            min_gain_bytes=_uint(min_gain_bytes, 'min_gain_bytes'),
                            max_actions=_uint(max_actions, 'max_actions'))
        if self.options['max_actions'] > 56:
            raise ValueError('max_actions must not exceed 56')
        if relay_capacity_bytes is not None:
            if len(relay_capacity_bytes) != N:
                raise ValueError('relay_capacity_bytes needs one value per rank')
            relay_capacity_bytes = tuple(_uint(v, 'relay capacity') for v in relay_capacity_bytes)
        self.options['relay_capacity_bytes'] = relay_capacity_bytes
        self._native = None
        if native_module is not None:
            if getattr(native_module, 'ep8_relay_planner_abi', None) != 1:
                raise RuntimeError('rebuild/install DeepEP: ep8_relay_planner_abi=1 required')
            self._native = native_module.Ep8RelayPlanner(
                self.relays, self.row_bytes, min_chunk_bytes, min_gain_bytes, max_actions,
                () if relay_capacity_bytes is None else relay_capacity_bytes)
        self.backend = 'native' if self._native is not None else 'python'

    def plan(self, counts):
        if self._native is None:
            return make_plan(counts, self.row_bytes, self.relays, **self.options)
        c = normalize_counts(counts)
        c, chosen, moved, loads, actions, evaluations, reason = self._native.plan(c)
        return RoutePlan(c, self.row_bytes, chosen, moved, loads, actions, evaluations, reason,
                         self._native.layout)


def make_plan(counts, row_bytes, allowed_relays, *, min_chunk_bytes=1,
              min_gain_bytes=0, max_actions=56, relay_capacity_bytes=None):
    """Greedy O(N^5), at most one distinct relay/action per original flow.

    Priority: highest current direct-edge load, then local peak reduction,
    then lowest (src,dst,relay). Loads include transit from earlier actions.
    relay_capacity_bytes bounds forwarded payload per relay (not all scratch).
    """
    c = normalize_counts(counts)
    row_bytes = _uint(row_bytes, 'row_bytes', 1)
    if sum(c) * row_bytes > MAX_I64:
        raise ValueError('total payload overflows int64 bytes')
    relays = normalize_relays(allowed_relays)
    min_chunk_bytes = _uint(min_chunk_bytes, 'min_chunk_bytes', 1)
    min_gain_bytes = _uint(min_gain_bytes, 'min_gain_bytes')
    max_actions = _uint(max_actions, 'max_actions')
    if max_actions > 56:
        raise ValueError('max_actions must not exceed 56')
    if relay_capacity_bytes is None:
        capacity = [sum(c)] * N
    else:
        if len(relay_capacity_bytes) != N:
            raise ValueError('relay_capacity_bytes needs one value per rank')
        capacity = [_uint(v, 'relay capacity') // row_bytes for v in relay_capacity_bytes]
    load = [v if f // N != f % N else 0 for f, v in enumerate(c)]
    chosen, moved, used = [-1] * 64, [0] * 64, [0] * N
    actions, evaluations, reason = [], 0, 'action_limit'
    chunk_rows = (min_chunk_bytes + row_bytes - 1) // row_bytes
    for _ in range(max_actions):
        best = None
        eligible = sorted((f for f in range(64) if f // N != f % N and c[f] and chosen[f] < 0),
                          key=lambda f: (-load[f], f))
        for f in eligible:
            a = load[f]
            if best is not None and a < best[0][0]:
                break
            s, d = divmod(f, N)
            for k in relays[f]:
                evaluations += 1
                h = max(load[s * N + k], load[k * N + d])
                x = min(c[f], max(0, (a - h) // 2), capacity[k] - used[k])
                gain = a - max(a - x, h + x)
                if x < chunk_rows or gain <= 0 or gain * row_bytes < min_gain_bytes:
                    continue
                key = (a, gain, -s, -d, -k)
                if best is None or key > best[0]:
                    best = (key, s, d, k, x, gain)
        if best is None:
            reason = 'no_profitable_candidate'
            break
        _, s, d, k, x, gain = best
        f = s * N + d
        chosen[f], moved[f] = k, x
        used[k] += x
        load[f] -= x
        load[s * N + k] += x
        load[k * N + d] += x
        actions.append((s, d, k, x, gain))
    return RoutePlan(c, row_bytes, tuple(chosen), tuple(moved), tuple(load),
                     tuple(actions), evaluations, reason)


@dataclass(frozen=True)
class Piece:
    src: int
    dst: int
    target: int                   # destination of first hop
    rows: int
    source: int                   # original send row offset
    final: int                    # final receive row offset
    first_send: int
    first_recv: int
    second_send: int = 0
    second_recv: int = 0


@dataclass(frozen=True)
class Schedule:
    plan: RoutePlan
    first_counts: tuple
    second_counts: tuple
    pieces: tuple
    self_offsets: tuple

    def scratch_rows(self, rank):
        return (sum(self.first_counts[rank * N:(rank + 1) * N]),
                sum(self.first_counts[rank::N]),
                sum(self.second_counts[rank * N:(rank + 1) * N]),
                sum(self.second_counts[rank::N]))

    def report(self):
        return dict(first_counts=list(self.first_counts), second_counts=list(self.second_counts),
                    phase_peak_sum_bytes=(max(self.first_counts) + max(self.second_counts)) * self.plan.row_bytes,
                    ccu_calls_per_execute=2 if self.plan.uses_relay else 1,
                    scratch_bytes_per_rank=[sum(self.scratch_rows(r)) * self.plan.row_bytes
                                            if self.plan.uses_relay else 0 for r in range(N)])


def _offsets(counts):
    src, dst, sends, recvs = [0] * 64, [0] * 64, [0] * N, [0] * N
    for s in range(N):
        for d in range(N):
            f = s * N + d
            src[f], dst[f] = sends[s], recvs[d]
            sends[s] += counts[f]
            recvs[d] += counts[f]
    return src, dst


def validate_plan(plan):
    c = normalize_counts(plan.counts)
    if len(plan.relay_rank) != 64 or len(plan.relay_rows) != 64:
        raise ValueError('invalid plan shape')
    for f, n in enumerate(c):
        s, d = divmod(f, N)
        x, k = plan.relay_rows[f], plan.relay_rank[f]
        if not 0 <= x <= n or (x and (k not in range(N) or k in (s, d))) or (not x and k != -1):
            raise ValueError('invalid relay assignment')
        if s == d and x:
            raise ValueError('self traffic cannot use a relay')
    return c


def build_schedule(plan):
    """Pack into two contiguous AllToAllV stages; preserve final source-major order."""
    c = validate_plan(plan)
    first, second, raw = [0] * 64, [0] * 64, []
    source, final = _offsets(c)
    for f, n in enumerate(c):
        s, d = divmod(f, N)
        x, k = plan.relay_rows[f], plan.relay_rank[f]
        if not 0 <= x <= n or (x and (k not in range(N) or k in (s, d))) or (not x and k != -1):
            raise ValueError('invalid relay assignment')
        if s == d:
            if x: raise ValueError('self traffic cannot use a relay')
            continue
        if n - x:
            raw.append((s, d, d, n - x, source[f], final[f]))
            first[s * N + d] += n - x
        if x:
            raw.append((s, d, k, x, source[f] + n - x, final[f] + n - x))
            first[s * N + k] += x
            second[k * N + d] += x
    fs, fr = _offsets(first)
    ss, sr = _offsets(second)
    first_used, second_used, pieces = [0] * 64, [0] * 64, []
    for s, d, k, n, original, dest in raw:
        f = s * N + k
        p = [s, d, k, n, original, dest, fs[f] + first_used[f], fr[f] + first_used[f]]
        first_used[f] += n
        if k != d:
            g = k * N + d
            p.extend((ss[g] + second_used[g], sr[g] + second_used[g]))
            second_used[g] += n
        pieces.append(Piece(*p))
    return Schedule(plan, tuple(first), tuple(second), tuple(pieces),
                    tuple((source[r * 8 + r], final[r * 8 + r], c[r * 8 + r]) for r in range(N)))


def execution_decision(plan, policy='force', *, schedule=None):
    """Separate a routing proposal from the decision to execute it.

    This is a conservative equal-link-bandwidth model, NOT a hardware latency
    prediction. With globally serial stages, peak(stage1)+peak(stage2) >=
    max(original edge): the largest original flow has portions n-x and x in
    those stages. Thus this executor cannot promise a network-only speedup.
    `force` is for measuring the actual relay path; `auto` rejects that known
    structural penalty. A future pipelined executor needs a different model.
    """
    if policy not in ('force', 'auto'):
        raise ValueError('policy must be force/auto')
    schedule = build_schedule(plan) if schedule is None else schedule
    direct = max((n for f, n in enumerate(plan.counts) if f // N != f % N), default=0)
    serial = max(schedule.first_counts) + max(schedule.second_counts)
    use = plan.uses_relay and (policy == 'force' or serial < direct)
    return dict(policy=policy, proposed_relay=plan.uses_relay, executed_relay=use,
                ccu_calls_per_execute=2 if use else 1,
                scratch_bytes_per_rank=[sum(schedule.scratch_rows(r)) * plan.row_bytes if use else 0
                                        for r in range(N)],
                reason=('forced_relay' if use else 'serial_phase_cost_not_better_than_direct'
                        if plan.uses_relay else 'no_relay_candidate'),
                direct_peak_bytes=direct * plan.row_bytes,
                serial_phase_peak_sum_bytes=serial * plan.row_bytes,
                model='equal_bandwidth_serial_stages_excludes_copy_launch_and_shared_resources')


def _coalesce_spans(spans):
    """Merge only copies adjacent in BOTH source and destination, never gaps."""
    result = []
    for src, dst, n in sorted(spans):
        if result and (result[-1][0] + result[-1][2], result[-1][1] + result[-1][2]) == (src, dst):
            a, b, old = result[-1]
            result[-1] = (a, b, old + n)
        else:
            result.append((src, dst, n))
    return result


class PreparedAllToAllV:
    """Reusable eager workspace. Exclusive to one rank/stream; ordered reuse is allowed.

    prepare/apply route matrices explicitly per batch. No hidden counts cache,
    expert histogram, device-to-host copy, graph capture, or background planner.
    """
    def __init__(self, buffer, handle, rank, plan, example_send, *, max_scratch_bytes=1 << 30,
                 sync_mode="stream", policy="force"):
        if sync_mode not in ("stream", "device") or policy not in ("force", "auto"):
            raise ValueError("sync_mode must be stream/device; policy must be force/auto")
        self.sync_mode, self.policy = sync_mode, policy
        import torch
        self.rank = _uint(rank, 'rank')
        if self.rank >= N or example_send.dim() < 1 or not example_send.is_contiguous():
            raise ValueError('invalid rank or noncontiguous send tensor')
        if example_send.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.int32):
            raise ValueError('unsupported payload dtype')
        if getattr(buffer, 'rank', rank) != rank or getattr(buffer, 'group_size', N) != N:
            raise ValueError('prepared rank/group differs from EP8 Buffer')
        self.schedule = build_schedule(plan)
        self.decision = execution_decision(plan, policy, schedule=self.schedule)
        self.uses_relay = self.decision["executed_relay"]
        self.plan, self.buffer, self.handle = plan, buffer, handle
        self.tail, self.dtype, self.device = tuple(example_send.shape[1:]), example_send.dtype, example_send.device
        if example_send.element_size() * prod(self.tail) != plan.row_bytes:
            raise ValueError('row_bytes does not match payload')
        self.send_rows = sum(plan.counts[rank * 8:(rank + 1) * 8])
        self.recv_rows = sum(plan.counts[rank::8])
        if example_send.shape[0] != self.send_rows:
            raise ValueError('send rows do not match routing matrix')
        sizes = self.schedule.scratch_rows(rank) if self.uses_relay else (0, 0, 0, 0)
        if sum(sizes) * plan.row_bytes > _uint(max_scratch_bytes, 'max_scratch_bytes'):
            raise ValueError('multipath workspace exceeds max_scratch_bytes')
        self.workspace = [torch.empty((n, *self.tail), dtype=self.dtype, device=self.device) for n in sizes]
        self.first_counts = list(self.schedule.first_counts)
        self.second_counts = list(self.schedule.second_counts)
        self.counts = list(plan.counts)
        self.outgoing = [p for p in self.schedule.pieces if p.src == rank]
        self.forwarding = [p for p in self.schedule.pieces if p.target == rank and p.target != p.dst]
        self.incoming = [p for p in self.schedule.pieces if p.dst == rank]
        self.pack_spans = _coalesce_spans([(p.source, p.first_send, p.rows) for p in self.outgoing])
        self.forward_spans = _coalesce_spans([(p.first_recv, p.second_send, p.rows) for p in self.forwarding])
        self.direct_spans = _coalesce_spans([(p.first_recv, p.final, p.rows) for p in self.incoming if p.target == p.dst])
        self.relay_spans = _coalesce_spans([(p.second_recv, p.final, p.rows) for p in self.incoming if p.target != p.dst])

    def _phase_fence(self):
        # Legacy reference only. The bridge's OrderedCcuTensorStream drains the
        # torch_npu host queue before direct ACL/HCOMM submission; the CCU launch
        # orders completion back onto that same stream (including die-1 slave).
        # Subsequent Torch copies therefore do not need a device-wide fence.
        if self.sync_mode == 'device' and self.device.type != 'cpu':
            import torch
            torch.npu.synchronize(self.device)

    def execute(self, send, recv):
        for tensor, rows in ((send, self.send_rows), (recv, self.recv_rows)):
            if (tuple(tensor.shape) != (rows, *self.tail) or tensor.dtype != self.dtype
                    or tensor.device != self.device or not tensor.is_contiguous()):
                raise ValueError('payload shape/dtype/device differs from prepared plan')
        # Reject overlapping views, not just identical tensor objects.
        if send.numel() and recv.numel():
            a, b = send.data_ptr(), recv.data_ptr()
            if a < b + recv.numel() * recv.element_size() and b < a + send.numel() * send.element_size():
                raise ValueError('send and receive buffers overlap')
        op = self.buffer.ccu_urma_peer_plan_alltoallv_out
        if not self.uses_relay:
            return op(send, recv, self.handle, self.counts)
        first_send, first_recv, second_send, second_recv = self.workspace
        for src, dst, n in self.pack_spans:
            first_send.narrow(0, dst, n).copy_(send.narrow(0, src, n))
        # Existing CCU kernel publishes all peers before waiting, including zero
        # directions, and completes remote writes before its done handshake.
        self._phase_fence()
        op(first_send, first_recv, self.handle, self.first_counts)
        self._phase_fence()
        for src, dst, n in self.forward_spans:
            second_send.narrow(0, dst, n).copy_(first_recv.narrow(0, src, n))
        self._phase_fence()
        op(second_send, second_recv, self.handle, self.second_counts)
        self._phase_fence()
        for tensor, spans in ((first_recv, self.direct_spans), (second_recv, self.relay_spans)):
            for src, dst, n in spans:
                recv.narrow(0, dst, n).copy_(tensor.narrow(0, src, n))
        src, dst, n = self.schedule.self_offsets[self.rank]
        if n:
            recv.narrow(0, dst, n).copy_(send.narrow(0, src, n))
        return recv


def gather_routing_counts(local_counts, group=None):
    """Explicit synchronous adapter for a model's device-resident packed row counts.

    local_counts: int64 tensor [8], on the device required by the process group.
    Returns a host tuple[64]. Includes the collective and GPU->CPU synchronization;
    caller must account for them in end-to-end measurements.
    """
    import torch
    import torch.distributed as dist
    if dist.get_world_size(group) != N or local_counts.shape != (N,) or local_counts.dtype != torch.int64:
        raise ValueError('requires an EP8 group and int64 local_counts[8]')
    parts = [torch.empty_like(local_counts) for _ in range(N)]
    dist.all_gather(parts, local_counts.contiguous(), group=group)
    return normalize_counts(torch.stack(parts).cpu().tolist())
