#!/usr/bin/env python3
"""Single-node eight-card resource domain, multiple two-rank graph workers.

Unix socket messages are local control-plane traffic, never sent by ACL callbacks.
This runner is a benchmark adapter, not an eight-rank AllToAll implementation.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import selectors
import signal
import socket
import statistics
import hashlib
import subprocess
import tempfile
import time

from a5_path_scheduler import JobSpec, Scheduler
from a5_physical_resources import PhysicalInventory, conflicts


def load_config(path):
    config = json.loads(path.read_text())
    if config.get('resource_mode') != 'physical_link' or config.get('expectation') not in ('shared', 'disjoint'):
        raise ValueError('require resource_mode=physical_link and expectation=shared|disjoint; card-sharing configs are obsolete')
    def resolve(name):
        value = Path(config[name])
        return value if value.is_absolute() else path.parent / value
    inventory = PhysicalInventory(resolve('topology_json'), resolve('provenance_json'))
    managed = config.get('managed_devices')
    if (not isinstance(managed, list) or len(managed) != 8 or
            any(type(d) is not int or d < 0 for d in managed) or len(set(managed)) != 8):
        raise ValueError('managed_devices must contain eight distinct physical card IDs')
    jobs = config.get('jobs', [])
    if not 2 <= len(jobs) <= 16:
        raise ValueError('configure 2..16 jobs to compare conflicting/disjoint collectives')
    names, endpoints = set(), set()
    result = []
    for entry in jobs:
        j = dict(entry)
        name = j.get('name', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', name) or name in names:
            raise ValueError('job names must be unique simple identifiers')
        names.add(name)
        devices, relays = j.get('devices', []), j.get('relay_phys', [])
        if (len(devices) != 2 or len(relays) not in (1, 2) or
                any(type(d) is not int for d in devices + relays) or
                len(set(devices + relays)) != len(devices + relays) or
                not set(devices + relays) <= set(managed)):
            raise ValueError(f'{name}: invalid endpoint/relay cards')
        # A card may be another job's relay; this is required to share a two-hop
        # physical link while keeping the worker endpoint sets disjoint.
        if endpoints & set(devices):
            raise ValueError('worker endpoint cards must be disjoint across jobs in v1')
        endpoints.update(devices)
        manifest = Path(j['relay_manifest'])
        if not manifest.is_absolute():
            manifest = path.parent / manifest
        manifest = manifest.resolve()
        with manifest.open() as f:
            rows = list(csv.DictReader((s for s in f if s.strip() and not s.startswith('#')), delimiter='\t'))
        if ([int(r['relay_phy']) for r in rows] != relays or
                any([int(r['src_phy']), int(r['dst_phy'])] != devices for r in rows)):
            raise ValueError(f'{name}: manifest endpoints/relay order differ from configuration')
        j['relay_manifest'] = str(manifest)
        policy = j.get('policy', 'mixed')
        if policy not in ['mixed'] + [f'relay{i}' for i in range(len(relays))]:
            raise ValueError(f'{name}: physical benchmark uses fixed relay-only policy')
        j['policy'] = policy
        if j.get('extra_resources'):
            raise ValueError('physical benchmark does not accept manually invented contention resources')
        j['path_links'] = [inventory.validate(row) for row in rows]
        if len({int(r['src_die']) for r in rows}) != 1 or len({int(r['dst_die']) for r in rows}) != 1:
            raise ValueError('relay catalog spans endpoint dies')
        # Every catalog path exchanges metadata, even when its payload is zero.
        j['resources'] = sorted({link for links in j['path_links'] for link in links})
        indices = range(len(relays)) if policy == 'mixed' else [int(policy[5:])]
        j['payload_resources'] = sorted({link for i in indices for link in j['path_links'][i]})
        j['relay_only'] = True
        result.append(j)
    shared = conflicts(result)
    if config['expectation'] == 'shared' and not shared:
        raise ValueError('no shared physical PAYLOAD link; sharing a relay card is insufficient')
    if config['expectation'] == 'disjoint' and any(set(a['resources']) & set(b['resources'])
                                                  for i, a in enumerate(result) for b in result[i + 1:]):
        raise ValueError('disjoint config has overlapping physical catalog links')
    return result


class ControlServer:
    def __init__(self, path, scheduler):
        self.scheduler, self.selector = scheduler, selectors.DefaultSelector()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(path))
        os.chmod(path, 0o600)
        self.listener.listen(64)
        self.listener.setblocking(False)
        self.selector.register(self.listener, selectors.EVENT_READ, None)
        self.clients, self.identities, self.registered = {}, {}, set()

    def close(self):
        for conn in list(self.clients):
            self._close(conn)
        self.selector.unregister(self.listener)
        self.listener.close()
        self.selector.close()

    def _close(self, conn):
        self.selector.unregister(conn)
        conn.close()
        self.clients.pop(conn)

    def queue(self, conn, line):
        client = self.clients[conn]
        client['out'] += (line + '\n').encode('ascii')
        self.selector.modify(conn, selectors.EVENT_READ | selectors.EVENT_WRITE, client)

    def grants(self, tasks):
        for task in tasks:
            job, it, op = task.key
            spec = self.scheduler.jobs[job]
            split = spec.split(it, op)
            for rank in range(spec.ranks):
                conn = self.identities[job, rank]
                self.queue(conn, f'GO {task.ticket} {len(split)} ' + ' '.join(map(str, split)))

    def message(self, conn, line):
        client = self.clients[conn]
        fields = line.split()
        if not client['identity']:
            if len(fields) != 3 or fields[0] != 'HELLO':
                raise ValueError('expected HELLO job rank')
            name, rank = fields[1], int(fields[2])
            if (name not in self.scheduler.jobs or not 0 <= rank < self.scheduler.jobs[name].ranks or
                    (name, rank) in self.registered):
                raise ValueError('invalid/duplicate worker identity')
            client['identity'] = name, rank
            self.identities[name, rank] = conn
            self.registered.add((name, rank))
            self.queue(conn, 'OK')
            return
        job, rank = client['identity']
        if len(fields) == 3 and fields[0] == 'ARRIVED':
            self.grants(self.scheduler.arrive(job, rank, int(fields[1]), int(fields[2])))
        elif len(fields) == 5 and fields[0] == 'DONE':
            tasks = self.scheduler.done(job, rank, *map(int, fields[1:]))
            self.queue(conn, 'ACK')
            self.grants(tasks)
        else:
            raise ValueError('unexpected scheduler message')

    def poll(self, timeout=0.02):
        for key, mask in self.selector.select(timeout):
            conn, client = key.fileobj, key.data
            if client is None:
                child, _ = conn.accept()
                child.setblocking(False)
                data = {'identity': None, 'in': b'', 'out': b''}
                self.clients[child] = data
                self.selector.register(child, selectors.EVENT_READ, data)
                continue
            if mask & selectors.EVENT_READ:
                block = conn.recv(4096)
                if not block:
                    identity = client['identity']
                    if not identity or not self.scheduler.rank_finished(*identity) or client['out']:
                        raise RuntimeError('worker disconnected before acknowledged completion; reservations retained')
                    self._close(conn)
                    continue
                client['in'] += block
                if len(client['in']) > 4096:
                    raise ValueError('oversized scheduler message')
                while b'\n' in client['in']:
                    line, client['in'] = client['in'].split(b'\n', 1)
                    self.message(conn, line.decode('ascii'))
            if mask & selectors.EVENT_WRITE and client['out']:
                count = conn.send(client['out'])
                client['out'] = client['out'][count:]
                if not client['out']:
                    self.selector.modify(conn, selectors.EVENT_READ, client)


def summary(scheduler, warmup):
    rows = [r for r in scheduler.metrics() if r['iteration'] >= warmup]
    result = {'mode': scheduler.mode, 'measured_collectives': len(rows),
              'blocked_collectives': sum(r['blocked'] for r in rows)}
    for field in ['arrival_skew_ns', 'cohort_wait_ns', 'resource_wait_ns', 'reservation_ns', 'ready_to_done_ns']:
        values = sorted(r[field] / 1e6 for r in rows)
        result[field.replace('_ns', '_ms')] = {
            'mean': statistics.mean(values), 'p50': statistics.median(values),
            'p95': values[max(0, (95 * len(values) + 99) // 100 - 1)]}
    spans = []
    for it in range(warmup, scheduler.iterations):
        tasks = [t for t in scheduler.tasks.values() if t.key[1] == it]
        spans.append((max(t.end_ns for t in tasks) - min(t.first_arrival_ns for t in tasks)) / 1e6)
    result['workload_iteration_span_ms'] = {'mean': statistics.mean(spans), 'samples': spans}
    result['notes'] = 'Host-observed synthetic workload timing; includes IPC/polling/validation between nodes, not device-only latency or model iteration throughput.'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--only-job', help='run one job from the validated physical plan for isolated timing/HCCN')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--schedule', choices=['serialize', 'uncontrolled'], default='serialize')
    p.add_argument('--mode', choices=['GRAPH', 'EAGER'], default='GRAPH')
    p.add_argument('--notify', choices=['CALLBACK', 'D2H'], default='CALLBACK')
    p.add_argument('--bytes', type=int, default=4 * 1024 * 1024)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--warmup-iters', type=int, default=2)
    p.add_argument('--poll-us', type=int, default=50)
    p.add_argument('--grant-delay-ms', type=int, default=0, help='debug only: hold reservation before READY; use 0 for performance')
    p.add_argument('--cohort-barrier', action='store_true', help='synthetic load alignment at each op across all jobs')
    p.add_argument('--require-blocked', action='store_true', help='fail if serialize did not exercise resource waiting')
    p.add_argument('--timeout', type=float, default=300)
    a = p.parse_args()
    try:
        jobs = load_config(a.config.resolve())
        if a.only_job:
            jobs = [j for j in jobs if j['name'] == a.only_job]
            if not jobs:
                raise ValueError('--only-job not found in configuration')
            if a.require_blocked:
                raise ValueError('--only-job cannot require a conflict')
        if not a.binary.is_file():
            raise ValueError('binary must exist')
        if (a.bytes < 1024 or a.bytes > 1 << 28 or a.bytes % 1024 or a.iters < 4 or
                not 0 <= a.warmup_iters < a.iters or not 1 <= a.poll_us <= 1000000 or
                not 0 <= a.grant_delay_ms <= 10000 or not 0 < a.timeout < 86400):
            raise ValueError('invalid size/iteration/timing arguments')
        if a.require_blocked and a.schedule != 'serialize':
            raise ValueError('--require-blocked requires serialize')
    except (ValueError, KeyError, TypeError, OSError) as error:
        p.error(str(error))
    a.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='a5-serial-scheduler.', dir=a.output_root.resolve()))
    (out / 'config.json').write_text(json.dumps(jobs, indent=2))
    (out / 'arguments.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=2))
    specs = [JobSpec(j['name'], 2, frozenset(j['resources']), len(j['relay_phys']), a.bytes, j['policy'], True) for j in jobs]
    physical = dict(resource_mode='physical_link', direction='both (two-rank bidirectional exchange)',
                    shared_payload_links=conflicts(jobs),
                    paths=[dict(job=j['name'], devices=j['devices'], relay_phys=j['relay_phys'],
                                path_links=j['path_links'], payload_resources=j['payload_resources']) for j in jobs])
    source = json.loads(a.config.read_text())
    (out / 'source_config.json').write_text(json.dumps(source, indent=2))
    for field, name in [('topology_json', 'topology.json'), ('provenance_json', 'provenance.json')]:
        path = Path(source[field])
        if not path.is_absolute():
            path = a.config.resolve().parent / path
        content = path.read_bytes()
        (out / name).write_bytes(content)
        physical[field + '_sha256'] = hashlib.sha256(content).hexdigest()
    (out / 'physical_plan.json').write_text(json.dumps(physical, indent=2))
    print(json.dumps(physical, indent=2), flush=True)
    print(f'RESULT_DIR={out}', flush=True)
    children, logs = [], []
    # Short AF_UNIX pathname, user-private directory, separate from potentially long log paths.
    with tempfile.TemporaryDirectory(prefix='a5sched-') as temp, (out / 'events.jsonl').open('w') as events:
        def emit(event):
            events.write(json.dumps(event) + '\n')
            events.flush()
        scheduler = Scheduler(specs, a.iters, a.schedule, a.cohort_barrier, emit=emit)
        endpoint = Path(temp) / 'control.sock'
        server = ControlServer(endpoint, scheduler)
        started = time.monotonic()
        try:
            for j in jobs:
                jobout = out / j['name']
                jobout.mkdir()
                manifest = jobout / 'relay_manifest.tsv'
                manifest.write_bytes(Path(j['relay_manifest']).read_bytes())
                env = os.environ.copy()
                for name in list(env):
                    if name.startswith(('A5_CCU_SYNTHETIC_', 'A5_CCU_DESC_', 'A5_CCU_PATH_', 'A5_CCU_SOURCE_ROUTE_', 'A5_GATE_')) or name in ('LD_PRELOAD', 'HCCL_ALGO', 'A5_CCU_CHANNEL_SYNC_MODE'):
                        env.pop(name)
                env.update(ASCEND_RT_VISIBLE_DEVICES=','.join(map(str, j['devices'])),
                           HCCL_OP_EXPANSION_MODE='CCU_SCHED', A5_GATE_SCHED_SOCKET=str(endpoint),
                           A5_GATE_SCHED_JOB=j['name'], A5_GATE_POLL_US=str(a.poll_us),
                           A5_GATE_RELAY_ONLY='1',
                           A5_GATE_SCHED_TIMEOUT_S=str(max(1, int(a.timeout) + 1)),
                           A5_GATE_GRANT_DELAY_MS=str(a.grant_delay_ms))
                for rank in range(2):
                    log = (jobout / f'rank{rank}.log').open('w')
                    logs.append(log)
                    command = [str(a.binary.resolve()), str(rank), str(rank), str(jobout / 'bootstrap'),
                               str(a.bytes), str(a.iters), str(manifest), a.notify, a.mode, '0']
                    children.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None, 0) for c in children):
                    raise RuntimeError('worker failed; no forced release of reservations')
                if time.monotonic() - started > a.timeout:
                    raise TimeoutError('scheduler/worker timeout; no forced release of reservations')
                server.poll()
            if any(c.returncode != 0 for c in children):
                raise RuntimeError('worker exited with failure after control-plane completion')
            if not scheduler.complete():
                raise RuntimeError('workers exited without all collective DONE messages')
            for j in jobs:
                for rank in range(2):
                    text = (out / j['name'] / f'rank{rank}.log').read_text()
                    if (f'GRAPH_GATE_RANK_PASS rank={rank} ' not in text or
                            text.count(f'GATE_OP_PASS rank={rank} ') != 2 * a.iters):
                        raise RuntimeError('missing payload/policy/worker completion checks')
            if a.require_blocked and not any(t.blocked for t in scheduler.tasks.values()):
                raise RuntimeError('no resource conflict observed')
            report = summary(scheduler, a.warmup_iters)
            report['wall_seconds_including_setup_validation'] = time.monotonic() - started
            (out / 'metrics.json').write_text(json.dumps(scheduler.metrics(), indent=2))
            (out / 'summary.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2), flush=True)
            print('SERIAL_SCHEDULER_ALL_WORKERS_PASS', flush=True)
        except BaseException as error:
            (out / 'failure.json').write_text(json.dumps(dict(error=str(error), active=[list(k) for k in scheduler.active], pending=[list(k) for k in scheduler.pending]), indent=2))
            raise
        finally:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            for child in children:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            server.close()
            for log in logs:
                log.close()


if __name__ == '__main__':
    main()
