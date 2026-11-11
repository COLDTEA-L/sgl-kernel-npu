import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from a5_ccu_dynamic_stage_audit import StageRecorder, read_legacy_counts, measurement_schedule, STAGE_METRICS
from a5_ccu_combination_audit import CombinationAudit
from summarize_a5_ccu_combination_audit import summarize_records, summarize_stage_records


class DynamicStageTests(unittest.TestCase):
    def test_legacy_operations_preserve_order_without_added_sync(self):
        calls = []
        class Tensor:
            shape, dtype = (8,), 'i64'
            def contiguous(self): calls.append('contiguous'); return self
            def cpu(self): calls.append('cpu'); return self
            def tolist(self): calls.append('tolist'); return [[r]*8 for r in range(8)]
        def empty(t): calls.append('alloc'); return Tensor()
        def gather(parts, source, group=None):
            self.assertEqual(len(parts), 8); calls.append('gather')
        def stack(parts): calls.append('stack'); return Tensor()
        torch = SimpleNamespace(int64='i64', empty_like=empty, stack=stack)
        dist = SimpleNamespace(get_world_size=lambda g: 8, all_gather=gather)
        recorder = StageRecorder(cpu_clock=lambda: 0.)
        result = read_legacy_counts(Tensor(), recorder, lambda rows: tuple(x for row in rows for x in row),
                                    torch=torch, dist=dist)
        self.assertEqual(result, tuple(r for r in range(8) for _ in range(8)))
        self.assertEqual(calls, ['alloc']*8 + ['contiguous', 'gather', 'stack', 'cpu', 'tolist'])
        self.assertEqual(len(recorder.snapshot()), 12)
        self.assertTrue(all(v >= 0 for v in recorder.snapshot().values()))

    def test_counts_control_never_submits_or_synchronizes(self):
        def forbidden(*a): self.fail('counts-only must not execute action/sync')
        recorder = StageRecorder(cpu_clock=lambda: 0.)
        audit = CombinationAudit(lambda: recorder.measure('read', lambda: (1,)*64),
                                 forbidden, forbidden, forbidden, forbidden, stages=recorder)
        audit.call('audit_counts_only', 'multipath')
        first = audit.samples[0]['stages']
        self.assertIn('read_wall_us', first)
        recorder.values['stale'] = 1
        audit.call('audit_counts_only', 'native')
        self.assertNotIn('stale', audit.samples[1]['stages'])
        self.assertNotIn('stale', first)

    def test_schedule_legacy_unchanged_and_diagnostic_complete(self):
        modes = ('native', 'direct', 'multipath')
        metrics = ('prepared', 'dynamic')
        expected = []
        for metric in metrics:
            for turn in range(6):
                order = modes[turn % 3:] + modes[:turn % 3]
                if turn % 2: order = order[::-1]
                expected.extend((metric, turn, mode) for mode in order)
        self.assertEqual(measurement_schedule(metrics, modes, 6), expected)
        schedule = measurement_schedule(STAGE_METRICS, modes, 6, interleave=True)
        wanted = {(m, r, x) for m in STAGE_METRICS for r in range(6) for x in modes}
        self.assertEqual(set(schedule), wanted)
        self.assertEqual(len(schedule), len(wanted))
        self.assertEqual({r for _, r, _ in schedule[:15]}, {0})
        self.assertNotEqual(schedule[0][0], schedule[15][0])

    def test_cpp_timers_not_confused_with_device_duration(self):
        recorder = StageRecorder(cpu_clock=lambda: 0.)
        recorder.cpp_run([12., 0., 30.])
        self.assertEqual(recorder.snapshot(), dict(cpp_plan_validate_us=12., cpp_submit_us=30.))

    def test_paired_stage_summary_keeps_rank_and_outliers(self):
        records = []
        for rank in range(8):
            samples = [dict(counts_host_us=10, action_host_us=20, fence_host_us=0,
                            call_host_us=30 if i == 0 else 300,
                            start_host_s=100+i*.01+rank*.00001,
                            stages={'counts_d2h_wall_us': rank+1.}) for i in range(2)]
            records.append((rank, dict(metric='audit_dynamic', mode='direct', round=0,
                samples=samples, iterations=2, elapsed_host_us=400 if rank == 2 else 350,
                final_drain_host_us=10)))
        groups = summarize_records(records)
        result = summarize_stage_records(records, groups)[0]
        self.assertEqual(result['rank'], 2)
        self.assertEqual(result['stages_mean_us']['counts_d2h_wall_us'], 3.)
        self.assertEqual(result['call_max_us'], 300)
        self.assertAlmostEqual(result['start_spread_max_us'], 70, places=4)
        with self.assertRaisesRegex(ValueError, 'eight ranks'):
            summarize_stage_records(records[:-1], groups)
        records[0][1]['samples'][0]['stages'] = {}
        with self.assertRaisesRegex(ValueError, 'missing stage'):
            summarize_stage_records(records, groups)

    def test_stage_cli_exports_paired_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'results').mkdir(); (root/'plans').mkdir()
            (root/'exit.status').write_text('0')
            (root/'plans/topology.json').write_text('{}')
            (root/'run.log').write_text(''.join(f'EP8_EID_RANK_PASS rank={r}\n' for r in range(8)))
            for rank in range(8):
                sample = dict(counts_host_us=10, action_host_us=0, fence_host_us=0,
                              call_host_us=10, start_host_s=rank*.000001,
                              stages={'counts_d2h_wall_us': float(rank+1)})
                row = dict(metric='audit_counts_only', mode='native', round=0, iterations=1,
                           samples=[sample], elapsed_host_us=20+rank, final_drain_host_us=2)
                result = dict(rank=rank, backend='participant_eid_ccu_urma',
                    options=dict(suite='benchmark', combination_audit=True, dynamic_stage_audit=True,
                                 rounds=1, dtype='float8_e4m3fn'),
                    reports=[dict(case='external', counts=[0]*64, row_bytes=5120, relay_rows=0,
                        samples_us={'audit_counts_only': {'native': [27]}},
                        median_us={'audit_counts_only': {'native': 27}}, rank_local_audit=[row])])
                (root/'results'/f'rank{rank}.json').write_text(json.dumps(result))
            script = Path(__file__).resolve().parents[3]/'scripts/summarize_a5_ccu_combination_audit.py'
            result = subprocess.run([sys.executable, str(script), str(root), '--output-json', str(root/'summary.json')],
                                    capture_output=True, text=True, check=True)
            self.assertIn('EP8_COMBINATION_AUDIT_SUMMARY_PASS', result.stdout)
            exported = json.loads((root/'summary.json').read_text())['reports'][0]['rounds'][0]
            self.assertEqual(exported['rank'], 7)
            self.assertEqual(exported['stages_mean_us']['counts_d2h_wall_us'], 8.)


if __name__ == '__main__': unittest.main()
