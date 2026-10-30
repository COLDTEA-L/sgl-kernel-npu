"""CPU-only control rendezvous for single-node tests; no process group required."""
from datetime import timedelta
import hashlib
from pathlib import Path


class CpuControl:
    def __init__(self, path, rank, world_size, timeout_seconds=180):
        from torch.distributed import FileStore
        if not 0 <= rank < world_size or timeout_seconds <= 0:
            raise ValueError('invalid CPU control rank/world size/timeout')
        self.rank, self.world_size = rank, world_size
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Caller supplies a fresh path for each launch/restart. Keep this store
        # alive through all control phases so FileStore cleanup cannot race them.
        self.store = FileStore(str(path), world_size)
        self.store.set_timeout(timedelta(seconds=timeout_seconds))

    def exchange(self, phase, value):
        keys = [f'{phase}/{r}' for r in range(self.world_size)]
        self.store.set(keys[self.rank], value)
        try:
            self.store.wait(keys)
            return [self.store.get(key) for key in keys]
        except RuntimeError as error:
            raise RuntimeError(f'CPU_CONTROL_FAILED phase={phase} rank={self.rank}: {error}') from error

    def check_config(self, identity):
        digest = hashlib.sha256(identity.encode()).hexdigest()
        values = self.exchange('config', digest)
        if any(value != values[0] for value in values):
            hashes = {r: value.decode() for r, value in enumerate(values)}
            raise ValueError(f'rank configuration mismatch: sha256={hashes}')
        return digest

    def barrier(self, phase):
        self.exchange(phase, 'ready')


def launch_store_path(run_dir, environment):
    run_id = environment.get('TORCHELASTIC_RUN_ID')
    if not run_id or run_id == 'none':
        raise ValueError('CPU control requires a unique torchrun rendezvous id; use --standalone')
    identity = run_id + ':' + environment.get('TORCHELASTIC_RESTART_COUNT', '0')
    return Path(run_dir) / ('cpu_control_' + hashlib.sha256(identity.encode()).hexdigest() + '.store')
