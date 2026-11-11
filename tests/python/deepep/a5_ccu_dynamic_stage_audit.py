"""Opt-in host diagnostics. Spans include waits; none are device execution times."""
import time

STAGE_METRICS = ('audit_counts_only', 'audit_dynamic_prepared', 'audit_dynamic',
                 'audit_dynamic_serial', 'audit_counts_prepare_only')


class StageRecorder:
    def __init__(self, clock=time.perf_counter, cpu_clock=None):
        self.clock = clock
        self.cpu_clock = cpu_clock or getattr(time, "thread_time", None)
        if self.cpu_clock is None:
            raise RuntimeError("dynamic stage audit requires time.thread_time on the target host")
        self.values = {}

    def reset(self):
        self.values.clear()

    def measure(self, name, fn, *args, **kwargs):
        wall, cpu = self.clock(), self.cpu_clock()
        result = fn(*args, **kwargs)
        self.values[name + '_cpu_us'] = (self.cpu_clock() - cpu) * 1e6
        self.values[name + '_wall_us'] = (self.clock() - wall) * 1e6
        return result

    def cpp_run(self, result):
        # EidHost::Run returns {Select/Build/Validate, reserved_zero, Submit}.
        self.values['cpp_plan_validate_us'] = float(result[0])
        self.values['cpp_submit_us'] = float(result[2])

    def snapshot(self):
        return dict(self.values)


def read_legacy_counts(local_counts, recorder, normalize, group=None, *, torch=None, dist=None):
    """Same operations/order as gather_routing_counts; no added sync/event/collective."""
    if torch is None:
        import torch
    if dist is None:
        import torch.distributed as dist
    if dist.get_world_size(group) != 8 or local_counts.shape != (8,) or local_counts.dtype != torch.int64:
        raise ValueError('requires an EP8 group and int64 local_counts[8]')
    parts = recorder.measure('counts_alloc', lambda: [torch.empty_like(local_counts) for _ in range(8)])
    source = recorder.measure('counts_contiguous', local_counts.contiguous)
    recorder.measure('counts_collective', dist.all_gather, parts, source, group=group)
    packed = recorder.measure('counts_stack', torch.stack, parts)
    host = recorder.measure('counts_d2h', packed.cpu)
    return recorder.measure('counts_convert', lambda: normalize(host.tolist()))


def measurement_schedule(metrics, modes, rounds, *, interleave=False):
    """Keep legacy order by default; diagnostic rotates metrics inside each round."""
    def ordered_modes(turn, offset=0):
        shift = (turn + offset) % len(modes)
        order = tuple(modes[shift:]) + tuple(modes[:shift])
        return order[::-1] if turn % 2 else order
    if not interleave:
        return [(metric, turn, mode) for metric in metrics for turn in range(rounds)
                for mode in ordered_modes(turn)]
    result = []
    for turn in range(rounds):
        shift = turn % len(metrics)
        base = tuple(metrics)[::-1] if turn % 2 else tuple(metrics)
        order = base[shift:] + base[:shift]
        for index, metric in enumerate(order):
            result.extend((metric, turn, mode) for mode in ordered_modes(turn, index))
    return result
