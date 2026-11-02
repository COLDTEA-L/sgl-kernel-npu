"""Host-boundary diagnostics; never interpret these samples as device kernel time."""
import time

AUDIT_METRICS = ('audit_dynamic', 'audit_dynamic_serial',
                 'audit_dynamic_prepared', 'audit_counts_prepare_only')


class CombinationAudit:
    def __init__(self, read_counts, submit, execute_prepared, prepare_only, synchronize,
                 clock=time.perf_counter):
        self.read_counts = read_counts
        self.submit = submit
        self.execute_prepared = execute_prepared
        self.prepare_only = prepare_only
        self.synchronize = synchronize
        self.clock = clock
        self.samples = []

    def reset(self):
        self.samples.clear()

    def call(self, metric, mode):
        if metric not in AUDIT_METRICS:
            raise ValueError('unknown combination audit metric: ' + metric)
        t0 = self.clock()
        counts = self.read_counts()
        t1 = self.clock()
        if metric == 'audit_counts_prepare_only':
            # Control only: host plan construction, no payload launch.
            self.prepare_only(mode, counts)
        elif metric == 'audit_dynamic_prepared':
            # Fixed-matrix microbenchmark only. Caller validates gathered counts
            # against the prepared matrix outside timing. Not a model fast path.
            self.execute_prepared(mode)
        else:
            self.submit(mode, counts)
        t2 = self.clock()
        if metric == 'audit_dynamic_serial':
            self.synchronize()
        t3 = self.clock()
        self.samples.append(dict(counts_host_us=(t1-t0)*1e6,
                                 action_host_us=(t2-t1)*1e6,
                                 fence_host_us=(t3-t2)*1e6,
                                 call_host_us=(t3-t0)*1e6))
