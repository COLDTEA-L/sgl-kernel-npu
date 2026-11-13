"""Admission invariants and real Unix-socket launcher tests, without a device."""
import json
import os
import hashlib
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
    def config(self, root, shared=True, same_relay=False, custom_b=None):
        topology = dict(version='2.0', peer_count=8, peer_list=[dict(local_id=i) for i in range(8)], edge_list=[])
        bindings = []
        def eid(device, peer):
            return '2001:0db8:0000:0000:0000:0000:0000:%04x' % (device * 16 + peer + 1)
        for a in range(8):
            for b in range(8):
                if a != b:
                    bindings.append(dict(device=a, die=0, port=b, eid=eid(a, b), net_layer=0,
                                         net_instance_id='fixture', verified=True,
                                         plane_id='0', name='udmac0', index=b))
                if a < b:
                    topology['edge_list'].append(dict(net_layer=0, protocols=['UB_CTP'], link_type='PEER2PEER',
                        local_a=a, local_b=b, local_a_ports=[f'0/{b}'], local_b_ports=[f'0/{a}']))
        topology['edge_count'] = len(topology['edge_list'])
        (root / 'topology.json').write_text(json.dumps(topology))
        (root / 'provenance.json').write_text(json.dumps(dict(source='rootinfo+atlas+live_hccn',
            topology_json_sha256=hashlib.sha256((root / 'topology.json').read_bytes()).hexdigest(), bindings=bindings)))
        jobs = []
        b_devices, b_relays = custom_b or ([0, 1] if shared and not same_relay else [4, 5],
                                         [0, 1] if same_relay else ([2, 3] if shared else [6, 7]))
        for name, devices, relays in [('a', [2, 3], [0, 1]), ('b', b_devices, b_relays)]:
            manifest = root / (name + '.tsv')
            from resolve_a5_explicit_multirelay_eids import FIELDS
            rows = []
            for relay in relays:
                rows.append(dict(route_id=1000 + relay, src_phy=devices[0], dst_phy=devices[1], relay_phy=relay,
                    src_die=0, dst_die=0, relay_die=0, src_port=relay, dst_port=relay,
                    relay_port_from_src=devices[0], relay_port_to_dst=devices[1],
                    src_eid=eid(devices[0], relay), dst_eid=eid(devices[1], relay)))
            manifest.write_text('\t'.join(FIELDS) + '\n' + ''.join('\t'.join(str(row.get(k, 0)) for k in FIELDS) + '\n' for row in rows))
            jobs.append(dict(name=name, devices=devices, relay_phys=relays, relay_manifest=manifest.name))
        path = root / 'config.json'
        path.write_text(json.dumps(dict(resource_mode='physical_link', expectation='shared' if shared else 'disjoint',
            topology_json='topology.json', provenance_json='provenance.json', managed_devices=list(range(8)), jobs=jobs)))
        return path

    def test_manifest_and_scope_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.config(root)
            jobs = load_config(path)
            self.assertEqual(len(set(jobs[0]['resources']) & set(jobs[1]['resources'])), 4)
            self.assertTrue(all(r.startswith('link:') for r in jobs[0]['resources']))
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

    def test_shared_relay_card_is_not_a_physical_link_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.config(Path(temp), same_relay=True)
            with self.assertRaisesRegex(ValueError, 'no shared physical PAYLOAD link'):
                load_config(path)
            config = json.loads(path.read_text())
            config['expectation'] = 'disjoint'
            path.write_text(json.dumps(config))
            jobs = load_config(path)
            self.assertFalse(set(jobs[0]['resources']) & set(jobs[1]['resources']))

    def test_manifest_port_not_bound_to_eid_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.config(root)
            import csv
            with (root / 'a.tsv').open() as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows, fields = list(reader), reader.fieldnames
            rows[0]['src_port'] = '7'
            with (root / 'a.tsv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'uniquely select'):
                load_config(path)

    def test_ambiguous_eid_port_and_topology_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.config(root)
            provenance = json.loads((root / 'provenance.json').read_text())
            clone = dict(next(b for b in provenance['bindings'] if b['device'] == 2 and b['port'] == 0))
            clone['port'] = 7
            provenance['bindings'].append(clone)
            (root / 'provenance.json').write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, 'uniquely select'):
                load_config(path)
            (root / 'topology.json').write_text((root / 'topology.json').read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                load_config(path)

    def test_metadata_overlap_does_not_count_as_payload_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.config(Path(temp), custom_b=([0, 4], [2, 5]))
            config = json.loads(path.read_text())
            # Catalogs share 2-0, but payloads via relay1 use disjoint hops.
            for job in config['jobs']:
                job['policy'] = 'relay1'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'no shared physical PAYLOAD link'):
                load_config(path)

    def test_manual_resource_name_cannot_fabricate_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.config(Path(temp))
            config = json.loads(path.read_text())
            config['jobs'][0]['extra_resources'] = ['invented-conflict']
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'invented'):
                load_config(path)

    def test_topology_based_catalog_selection(self):
        from a5_physical_resources import PhysicalInventory
        from prepare_a5_physical_conflict import catalogs, choose
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.config(root)
            inv = PhysicalInventory(root / 'topology.json', root / 'provenance.json')
            a = catalogs(inv.provenance['bindings'], inv.edges, [2, 3], [0, 1], 0, inv)
            b = catalogs(inv.provenance['bindings'], inv.edges, [0, 1], [2, 3], 0, inv)
            self.assertEqual(len(choose(a, b, True)[2]), 4)
            c = catalogs(inv.provenance['bindings'], inv.edges, [4, 5], [0, 1], 0, inv)
            with self.assertRaisesRegex(ValueError, 'no requested physical-link'):
                choose(a, c, True)

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

    def test_isolated_job_retains_same_relay_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / 'worker'
            binary.write_text(FAKE_WORKER)
            binary.chmod(0o755)
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_a5_serial_scheduler.py'),
                '--binary', str(binary), '--config', str(self.config(root)), '--only-job', 'a',
                '--output-root', str(root / 'out'), '--iters', '4', '--warmup-iters', '1'],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            out = next((root / 'out').iterdir())
            report = json.loads((out / 'summary.json').read_text())
            self.assertEqual(report['measured_collectives'], 6)
            grants = [e for e in map(json.loads, (out / 'events.jsonl').read_text().splitlines()) if e['event'] == 'GRANT']
            self.assertTrue(all(e['split'] == [2097152, 2097152] for e in grants))

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
