#!/usr/bin/env python3
"""Planner/layout tests, independent payload oracle, optional Torch CPU executor test."""
import importlib.util
import json
import tempfile
import random
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python/deep_ep/deep_ep'))
sys.path.insert(0, str(ROOT / 'scripts'))
from heuristic_alltoallv import make_plan, build_schedule, normalize_counts, cut_lower_bound_rows, PreparedAllToAllV, execution_decision, _coalesce_spans
from prepare_a5_ccu_ep8_topology import build_topology
from summarize_a5_ccu_heuristic import summarize, compare_runs

ALL = [[k for k in range(8) if k not in (s, d)] if s != d else []
       for s in range(8) for d in range(8)]


def collective(send, counts):
    """Independent source-major AllToAllV reference on Python row objects."""
    out = [[] for _ in range(8)]
    for s in range(8):
        offset = 0
        for d in range(8):
            n = counts[s * 8 + d]
            out[d].extend(send[s][offset:offset + n])
            offset += n
        assert offset == len(send[s])
    return out


def simulate(plan):
    sch = build_schedule(plan)
    send = [[(s, r) for r in range(sum(plan.counts[s * 8:s * 8 + 8]))] for s in range(8)]
    expected = collective(send, plan.counts)
    first = [[None] * sch.scratch_rows(r)[0] for r in range(8)]
    for p in sch.pieces:
        first[p.src][p.first_send:p.first_send + p.rows] = send[p.src][p.source:p.source + p.rows]
    first_recv = collective(first, sch.first_counts)
    second = [[None] * sch.scratch_rows(r)[2] for r in range(8)]
    for p in sch.pieces:
        if p.target != p.dst:
            second[p.target][p.second_send:p.second_send + p.rows] = first_recv[p.target][p.first_recv:p.first_recv + p.rows]
    second_recv = collective(second, sch.second_counts)
    actual = [[None] * len(v) for v in expected]
    for p in sch.pieces:
        buf, off = ((first_recv[p.dst], p.first_recv) if p.target == p.dst
                    else (second_recv[p.dst], p.second_recv))
        actual[p.dst][p.final:p.final + p.rows] = buf[off:off + p.rows]
    for r, (src, dst, n) in enumerate(sch.self_offsets):
        actual[r][dst:dst + n] = send[r][src:src + n]
    assert actual == expected
    assert all(v is not None for rank in first + second for v in rank)


class PlannerTests(unittest.TestCase):
    def test_single_flow_exact_split_and_group_relays_only(self):
        c = [0] * 64; c[4] = 30
        p = make_plan(c, 8192, ALL)
        self.assertEqual(p.actions, ((0, 4, 1, 15, 15),))
        self.assertEqual(max(p.link_rows), 15)
        self.assertEqual(p.stop_reason, 'no_profitable_candidate')
        simulate(p)

    def test_equal_load_no_relay_and_cut_example(self):
        p = make_plan([7] * 64, 2, ALL)
        self.assertFalse(p.uses_relay)
        c = [7 if s < 4 and d >= 4 else 0 for s in range(8) for d in range(8)]
        self.assertEqual(cut_lower_bound_rows(c)[0], 7)
        self.assertFalse(make_plan(c, 2, ALL).uses_relay)

    def test_background_transit_is_counted(self):
        c = [0] * 64; c[1] = 40; c[2] = 30
        allowed = [[] for _ in range(64)]
        allowed[1] = [2]; allowed[2] = [3]
        p = make_plan(c, 1, allowed)
        self.assertEqual(p.actions[0], (0, 1, 2, 5, 5))
        self.assertEqual(p.actions[1], (0, 2, 3, 17, 17))
        self.assertEqual(p.link_rows[2], 18)  # remaining 13 + previous transit 5
        simulate(p)

    def test_threshold_budget_capacity_zero_self(self):
        c = [0] * 64; c[4] = 31; c[0] = 100
        self.assertFalse(make_plan(c, 2, ALL, max_actions=0).uses_relay)
        self.assertFalse(make_plan(c, 2, ALL, min_chunk_bytes=31).uses_relay)
        self.assertFalse(make_plan(c, 2, ALL, min_gain_bytes=31).uses_relay)
        self.assertFalse(make_plan(c, 2, ALL, relay_capacity_bytes=[0] * 8).uses_relay)
        p = make_plan(c, 2, ALL, relay_capacity_bytes=[6] * 8)
        self.assertEqual(p.relay_rows[4], 3)
        for c in ([0] * 64, [r + 1 if r == d else 0 for r in range(8) for d in range(8)]):
            simulate(make_plan(c, 1, ALL))

    def test_invalid_inputs(self):
        for c in ([0] * 63, [-1] + [0] * 63, [True] * 64, [1.5] * 64, [2**63] * 64):
            with self.assertRaises(ValueError): make_plan(c, 1, ALL)
        bad = [list(v) for v in ALL]; bad[1] = [8]
        with self.assertRaises(ValueError): make_plan([1] * 64, 1, bad)
        with self.assertRaises(ValueError): make_plan([1] * 64, 0, ALL)
        with self.assertRaises(ValueError): make_plan([1] * 64, 2**62, ALL)
        with self.assertRaises(ValueError): make_plan([1] * 64, 1, ALL, max_actions=57)

    def test_random_conservation_determinism_and_schedule(self):
        rng = random.Random(20261028)
        for trial in range(150):
            c = [rng.randrange(30) if rng.random() < 0.4 else 0 for _ in range(64)]
            if trial % 3 == 0:
                c[56:64] = [0] * 8
                c[6::8] = [0] * 8
            allowed = [[k for k in ks if rng.random() < .6] for ks in ALL]
            p = make_plan(c, 514, allowed)
            self.assertEqual(p, make_plan(c, 514, allowed))
            self.assertLessEqual(len(p.actions), 56)
            self.assertLessEqual(p.candidate_evaluations, 9576)
            self.assertEqual(len({(s, d) for s, d, *_ in p.actions}), len(p.actions))
            load = [c[f] if f // 8 != f % 8 else 0 for f in range(64)]
            for s, d, k, x, gain in p.actions:
                self.assertIn(k, allowed[s * 8 + d]); self.assertGreater(gain, 0)
                old = max(load)
                load[s * 8 + d] -= x; load[s * 8 + k] += x; load[k * 8 + d] += x
                self.assertLessEqual(max(load), old)
            self.assertEqual(tuple(load), p.link_rows)
            self.assertEqual(sum(load), sum(c[f] for f in range(64) if f // 8 != f % 8) + sum(p.relay_rows))
            simulate(p)

    def test_serial_cost_gate_does_not_claim_link_peak_speedup(self):
        rng = random.Random(73)
        matrices = [[rng.randrange(100) for _ in range(64)] for _ in range(100)]
        hot = [0] * 64; hot[4] = 4096
        matrices.append(hot)
        for c in matrices:
            plan = make_plan(c, 8192, ALL)
            decision = execution_decision(plan, 'auto')
            self.assertGreaterEqual(decision['serial_phase_peak_sum_bytes'], decision['direct_peak_bytes'])
            self.assertFalse(decision['executed_relay'])
            self.assertEqual(execution_decision(plan, 'force')['executed_relay'], plan.uses_relay)
        decision = execution_decision(make_plan(hot, 8192, ALL), 'auto')
        self.assertEqual(decision['direct_peak_bytes'], 32 << 20)
        self.assertEqual(decision['serial_phase_peak_sum_bytes'], 32 << 20)
        with self.assertRaises(ValueError): execution_decision(plan, 'bad')

    def test_coalescing_keeps_gaps_and_destination_order(self):
        self.assertEqual(_coalesce_spans([(3, 13, 2), (0, 10, 3), (5, 20, 1), (7, 21, 2)]),
                         [(0, 10, 5), (5, 20, 1), (7, 21, 2)])

    def test_v2_dynamic_per_round_slowest_rank_and_no_relay_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'results').mkdir()
            (root / 'exit.status').write_text('0')
            for rank in range(8):
                case = dict(case='cut', correctness='PASS', fingerprint='same',
                            plan={'fingerprint': 'same'}, schedule={}, planner_us=1,
                            prepared_round_us={'direct': [10, 10], 'multipath': [10, 10]},
                            dynamic_mean_us={'direct': 20, 'multipath': 10},
                            dynamic_round_us={'direct': [20 + rank, 50 - rank], 'multipath': [10, 10]},
                            dynamic_components_round_us={m: [dict(counts=rank + 1), dict(counts=rank + 2)]
                                                         for m in ('direct', 'multipath')},
                            execution={'executed_relay': False}, executor_sync='stream')
                result = dict(rank=rank, ranks=8, correctness='PASS', mode='both', dtype='float16',
                              width=8, backend='test', timing_scope='test', dynamic_scope='test',
                              benchmark={'protocol': 2}, cases=[case])
                (root / 'results' / f'rank{rank}.json').write_text(json.dumps(result))
            summary = summarize(root)['cases'][0]
            self.assertEqual(summary['dynamic']['direct']['slowest_rank_round_us'], [27, 50])
            self.assertEqual(summary['dynamic']['direct']['mean_us'], 38.5)
            self.assertEqual(summary['dynamic']['direct']['slowest_rank_components_round_us'],
                             [dict(counts=8), dict(counts=2)])
            self.assertIn('NOT multipath speedup', summary['analysis'])
            result['cases'][0]['dynamic_round_us']['direct'] = [1]
            (root / 'results' / 'rank7.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError): summarize(root)

    def test_topology_join_and_missing_live_eid(self):
        devices = list(reversed(range(8))); entries, edges = [], []
        for s in range(8):
            for d in range(s + 1, 8):
                edges.append(dict(net_layer=0, link_type='PEER2PEER', protocols=['UB_CTP'],
                                  local_a=devices[s], local_b=devices[d],
                                  local_a_ports=[f'0/{d}'], local_b_ports=[f'0/{s}']))
                for a, b in ((s, d), (d, s)):
                    entries.append(dict(device=devices[a], die=0, port=b, net_layer=0,
                                        net_instance_id='test', eid=f'{a}-{b}', verified=True))
        rows, t = build_topology(devices, entries, edges)
        self.assertEqual(t['allowed_relays'], ALL)
        self.assertEqual(rows[0]['src_phy'], 7)
        self.assertEqual(len(rows), 28)
        entries[0]['verified'] = False
        with self.assertRaisesRegex(ValueError, 'verified'): build_topology(devices, entries, edges)

    def test_summary_slowest_rank_comparison_and_missing_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'results').mkdir()
            (root / 'exit.status').write_text('0')
            for rank in range(8):
                case = dict(case='test', correctness='PASS', fingerprint='same',
                            plan={'fingerprint': 'same'}, schedule={}, planner_us=rank + 1,
                            prepared_round_us={'direct': [10 + rank, 30 - rank], 'multipath': [10, 10]},
                            dynamic_mean_us={'direct': 40 + rank, 'multipath': 50})
                result = dict(rank=rank, ranks=8, correctness='PASS', mode='both', dtype='float16',
                              width=8, backend='test', timing_scope='test', dynamic_scope='test',
                              benchmark={}, cases=[case])
                (root / 'results' / f'rank{rank}.json').write_text(json.dumps(result))
            summary = summarize(root)['cases'][0]
            self.assertEqual(summary['prepared']['direct']['slowest_rank_round_us'], [17, 30])
            self.assertLess(summary['dynamic']['direct_over_multipath'], 1)
            self.assertEqual(compare_runs(root, root)[0]['dynamic']['direct_us'], 47)
            (root / 'results' / 'rank7.json').unlink()
            with self.assertRaises(FileNotFoundError): summarize(root)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch not installed')
    def test_actual_executor_cpu_threads_repeated_and_changed_matrix(self):
        import torch
        torch.set_num_threads(1)
        for trial in range(12):
            c = [((s * 17 + d * 3 + trial) % 13) if d != 6 and s != 7 else 0
                 for s in range(8) for d in range(8)]
            if trial % 3 == 2: c = [2] * 64  # exercise direct fallback too
            p = make_plan(c, 14, ALL)
            rendezvous = threading.Barrier(8, timeout=20)
            calls = [None] * 8
            errors = []
            class Buffer:
                def __init__(self, rank): self.rank = rank
                def ccu_urma_peer_plan_alltoallv_out(self, send, recv, handle, counts):
                    calls[self.rank] = (send.clone(), recv, counts)
                    rendezvous.wait()
                    if self.rank == 0:
                        assert all(x[2] == counts for x in calls)
                        output = collective([x[0].tolist() for x in calls], counts)
                        for r in range(8):
                            calls[r][1].copy_(torch.tensor(output[r], dtype=torch.float16).reshape(-1, 7))
                    rendezvous.wait()
                    return recv
            def worker(r):
                try:
                    n = sum(c[r * 8:r * 8 + 8]); m = sum(c[r::8])
                    source = (torch.arange(n * 7).reshape(n, 7) % 97 + r).to(torch.float16)
                    input_storage = torch.full((n * 7 + 74,), -19., dtype=torch.float16)
                    send = input_storage[37:-37].reshape(n, 7); send.copy_(source)
                    storage = torch.full((m * 7 + 74,), -17., dtype=torch.float16)
                    recv = storage[37:-37].reshape(m, 7)
                    prepared = PreparedAllToAllV(Buffer(r), 1, r, p, send,
                                                sync_mode='device' if trial // 3 % 2 else 'stream',
                                                policy='auto' if trial // 6 else 'force')
                    if trial // 6:
                        self.assertFalse(prepared.uses_relay)
                        self.assertEqual(sum(v.numel() for v in prepared.workspace), 0)
                    for epoch in range(3):
                        send.copy_(source + epoch); recv.fill_(-17)
                        before = input_storage.clone()
                        prepared.execute(send, recv)
                        expected = []
                        for s in range(8):
                            start = sum(c[s * 8:s * 8 + r])
                            for row in range(start, start + c[s * 8 + r]):
                                expected.append([(row * 7 + col) % 97 + s + epoch for col in range(7)])
                        self.assertTrue(torch.equal(recv, torch.tensor(expected).reshape(m, 7)))
                        self.assertTrue(torch.equal(before, input_storage))
                        self.assertTrue(bool((storage[:37] == -17).all() and (storage[-37:] == -17).all()))
                except BaseException as exc:
                    errors.append(exc); rendezvous.abort()
            threads = [threading.Thread(target=worker, args=(r,)) for r in range(8)]
            for t in threads: t.start()
            for t in threads: t.join(30)
            self.assertFalse(any(t.is_alive() for t in threads))
            if errors: raise errors[0]


if __name__ == '__main__':
    unittest.main()
