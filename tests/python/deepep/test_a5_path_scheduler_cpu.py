"""Admission invariants and real Unix-socket launcher tests, without a device."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from a5_path_scheduler import JobSpec, Scheduler
from run_a5_serial_scheduler import ControlServer, load_config


class Clock:
    def __init__(self):
        self.ns = 100

    def __call__(self):
        self.ns += 1
        return self.ns


def job(name, resources, ranks=2):
    return JobSpec(name, ranks, frozenset(resources), 3, 4096)


class AdmissionTests(unittest.TestCase):
    def scheduler(self, jobs, **kwargs):
        return Scheduler(jobs, 4, clock=Clock(), **kwargs)

    def ready(self, s, name, it=0, op=0):
        grants = []
        for rank in range(s.jobs[name].ranks):
            grants += s.arrive(name, rank, it, op)
        return grants

    def finish(self, s, name, it=0, op=0):
        ticket = s.tasks[name, it, op].ticket
        grants = []
        for rank in range(s.jobs[name].ranks):
            grants += s.done(name, rank, it, op, ticket, 100)
        return grants

    def test_reserve_after_all_arrived_release_after_all_done(self):
        s = self.scheduler([job('a', ['relay0']), job('b', ['relay0'])])
        self.assertEqual(s.arrive('a', 0, 0, 0), [])
        self.assertEqual(s.active, {})
        self.assertEqual(len(s.arrive('a', 1, 0, 0)), 1)
        self.assertEqual(self.ready(s, 'b'), [])
        ticket = s.tasks['a', 0, 0].ticket
        self.assertEqual(s.done('a', 0, 0, 0, ticket, 100), [])
        self.assertIn(('a', 0, 0), s.active)
        grants = s.done('a', 1, 0, 0, ticket, 101)
        self.assertEqual([t.key[0] for t in grants], ['b'])

    def test_disjoint_work_bypasses_blocked_waiter(self):
        s = self.scheduler([job('a', ['x']), job('b', ['x']), job('c', ['z'])])
        self.ready(s, 'a')
        self.assertEqual(self.ready(s, 'b'), [])
        self.assertEqual([t.key[0] for t in self.ready(s, 'c')], ['c'])

    def test_per_resource_fifo_and_atomic_multi_resource_reservation(self):
        s = self.scheduler([job('a', ['x']), job('b', ['x', 'y']), job('c', ['y'])])
        self.ready(s, 'a')
        self.ready(s, 'b')
        self.assertEqual(self.ready(s, 'c'), [])
        self.assertEqual(list(s.active), [('a', 0, 0)])
        self.assertEqual([t.key[0] for t in self.finish(s, 'a')], ['b'])
        self.assertEqual([t.key[0] for t in self.finish(s, 'b')], ['c'])

    def test_uncontrolled_mode_records_overlapping_reservations(self):
        events = []
        s = self.scheduler([job('a', ['x']), job('b', ['x'])], mode='uncontrolled', emit=events.append)
        self.ready(s, 'a')
        self.assertEqual(len(self.ready(s, 'b')), 1)
        self.assertEqual(len(s.active), 2)
        self.assertEqual([e for e in events if e['event'] == 'GRANT'][-1]['overlap_tickets'], [1])

    def test_eight_rank_aggregation(self):
        s = self.scheduler([job('a', ['x'], 8)])
        for rank in range(7):
            self.assertEqual(s.arrive('a', rank, 0, 0), [])
        ticket = s.arrive('a', 7, 0, 0)[0].ticket
        for rank in range(7):
            s.done('a', rank, 0, 0, ticket, 0)
            self.assertTrue(s.active)
        s.done('a', 7, 0, 0, ticket, 0)
        self.assertFalse(s.active)

    def test_duplicate_stale_or_out_of_order_messages_rejected(self):
        s = self.scheduler([job('a', ['x'])])
        with self.assertRaisesRegex(ValueError, 'out-of-order'):
            s.arrive('a', 0, 0, 1)
        s.arrive('a', 0, 0, 0)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            s.arrive('a', 0, 0, 0)
        with self.assertRaisesRegex(ValueError, 'ungranted'):
            s.done('a', 0, 0, 0, 1, 0)
        ticket = s.arrive('a', 1, 0, 0)[0].ticket
        with self.assertRaisesRegex(ValueError, 'stale'):
            s.done('a', 0, 0, 0, ticket + 1, 0)
        s.done('a', 0, 0, 0, ticket, 0)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            s.done('a', 0, 0, 0, ticket, 0)

    def test_cohort_barrier_is_separate_from_resource_waiting(self):
        s = self.scheduler([job('a', ['x']), job('b', ['x'])], cohort_barrier=True)
        self.assertEqual(self.ready(s, 'a'), [])
        self.assertFalse(s.active)
        self.assertEqual([t.key[0] for t in self.ready(s, 'b')], ['a'])
        self.finish(s, 'a')
        self.finish(s, 'b')
        metrics = s.metrics()
        self.assertGreater(metrics[0]['cohort_wait_ns'], 0)
        self.assertTrue(metrics[1]['blocked'])
        self.assertGreater(metrics[1]['resource_wait_ns'], 0)

    def test_multiple_replays_have_distinct_tickets(self):
        s = self.scheduler([job('a', ['x'])])
        tickets = []
        for it in range(4):
            for op in range(2):
                tickets.append(self.ready(s, 'a', it, op)[0].ticket)
                self.finish(s, 'a', it, op)
        self.assertTrue(s.complete())
        self.assertEqual(len(set(tickets)), 8)
        with self.assertRaises(ValueError):
            s.done('a', 0, 0, 0, tickets[0], 0)

    def test_missing_done_never_expires_or_releases(self):
        s = self.scheduler([job('a', ['x']), job('b', ['x'])])
        self.ready(s, 'a')
        self.ready(s, 'b')
        s.clock.ns += 10**15
        self.assertEqual(s.schedule(), [])
        self.assertEqual(list(s.active), [('a', 0, 0)])


FAKE_WORKER = '''#!/usr/bin/env python3
import os, socket, sys, time
rank, iters = int(sys.argv[1]), int(sys.argv[5])
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(os.environ['A5_GATE_SCHED_SOCKET'])
f = s.makefile('rwb', buffering=0)
def exchange(message):
    f.write((message + '\\n').encode())
    reply = f.readline().decode().strip()
    if not reply: raise RuntimeError('server disconnected')
    return reply
assert exchange('HELLO %s %s' % (os.environ['A5_GATE_SCHED_JOB'], rank)) == 'OK'
for it in range(iters):
    for op in range(2):
        reply = exchange('ARRIVED %d %d' % (it, op)).split()
        assert reply[0] == 'GO' and sum(map(int, reply[3:])) == int(sys.argv[4])
        time.sleep(0.001 if rank == 0 else 0.005)
        assert exchange('DONE %d %d %s 1000000' % (it, op, reply[1])) == 'ACK'
        print('GATE_OP_PASS rank=%d iteration=%d op=%d' % (rank, it, op), flush=True)
print('GRAPH_GATE_RANK_PASS rank=%d iterations=%d' % (rank, iters), flush=True)
f.close()
s.close()
'''


class AdapterTests(unittest.TestCase):
    def config(self, root, shared=True):
        jobs = []
        for name, devices, relays in [('a', [2, 3], [0, 1]),
                                     ('b', [4, 5], [0, 1] if shared else [6, 7])]:
            manifest = root / (name + '.tsv')
            manifest.write_text('relay_phy\tsrc_phy\tdst_phy\n' + ''.join(f'{r}\t{devices[0]}\t{devices[1]}\n' for r in relays))
            jobs.append(dict(name=name, devices=devices, relay_phys=relays, relay_manifest=manifest.name))
        path = root / 'config.json'
        path.write_text(json.dumps(dict(managed_devices=list(range(8)), jobs=jobs)))
        return path

    def test_manifest_and_scope_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.config(root)
            jobs = load_config(path)
            self.assertEqual(set(jobs[0]['resources']) & set(jobs[1]['resources']), {'card:0', 'card:1'})
            config = json.loads(path.read_text())
            config['jobs'][0]['relay_phys'] = [1, 0]
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'manifest'):
                load_config(path)

    def test_socket_early_disconnect_is_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            s = Scheduler([job('a', ['x'])], 4)
            server = ControlServer(Path(temp) / 's', s)
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                client.connect(str(Path(temp) / 's'))
                server.poll(0.1)
                client.sendall(b'HELLO a 0\n')
                server.poll(0.1)
                server.poll(0.1)
                self.assertEqual(client.recv(16), b'OK\n')
                client.close()
                with self.assertRaisesRegex(RuntimeError, 'disconnected'):
                    server.poll(0.1)
            finally:
                client.close()
                server.close()

    def test_end_to_end_unix_socket_scheduler(self):
        for mode, shared in [('serialize', True), ('serialize', False), ('uncontrolled', True)]:
            with self.subTest(mode=mode, shared=shared), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = self.config(root, shared)
                binary = root / 'worker'
                binary.write_text(FAKE_WORKER)
                binary.chmod(0o755)
                if os.environ.get('A5_SCHEDULER_CPU_CLIENT'):
                    binary = Path(os.environ['A5_SCHEDULER_CPU_CLIENT'])
                command = [sys.executable, str(ROOT / 'scripts/run_a5_serial_scheduler.py'),
                           '--binary', str(binary), '--config', str(config), '--output-root', str(root / 'out'),
                           '--iters', '4', '--warmup-iters', '1', '--cohort-barrier', '--schedule', mode]
                if mode == 'serialize' and shared:
                    command.append('--require-blocked')
                result = subprocess.run(command, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('SERIAL_SCHEDULER_ALL_WORKERS_PASS', result.stdout)
                out = next((root / 'out').iterdir())
                report = json.loads((out / 'summary.json').read_text())
                self.assertEqual(report['measured_collectives'], 12)
                self.assertEqual(report['blocked_collectives'] > 0, mode == 'serialize' and shared)
                active = {}
                for event in map(json.loads, (out / 'events.jsonl').read_text().splitlines()):
                    if event['event'] == 'GRANT':
                        if mode == 'serialize':
                            for resources in active.values():
                                self.assertFalse(set(event['resources']) & resources)
                        active[event['ticket']] = set(event['resources'])
                    elif event['event'] == 'RELEASE':
                        active.pop(event['ticket'])
                self.assertFalse(active)

    def test_zero_exit_without_protocol_is_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / 'empty_worker'
            binary.write_text('#!/usr/bin/env python3\n')
            binary.chmod(0o755)
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_a5_serial_scheduler.py'),
                                     '--binary', str(binary), '--config', str(self.config(root)),
                                     '--output-root', str(root / 'out'), '--iters', '4'],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('without all collective DONE', result.stderr)

    def test_nonzero_exit_after_pass_markers_is_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / 'late_failure_worker'
            binary.write_text(FAKE_WORKER + '\nsys.exit(7)\n')
            binary.chmod(0o755)
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_a5_serial_scheduler.py'),
                                     '--binary', str(binary), '--config', str(self.config(root)),
                                     '--output-root', str(root / 'out'), '--iters', '4'],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('SERIAL_SCHEDULER_ALL_WORKERS_PASS', result.stdout)


if __name__ == '__main__':
    unittest.main()
