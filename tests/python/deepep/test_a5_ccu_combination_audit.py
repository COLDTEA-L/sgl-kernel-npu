import json
import subprocess
import tempfile
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
from a5_ccu_combination_audit import CombinationAudit, AUDIT_METRICS
from summarize_a5_ccu_combination_audit import summarize_records


class AuditTests(unittest.TestCase):
    def test_boundaries_actions_and_reset(self):
        for metric in AUDIT_METRICS:
            calls=[]
            now=[0.]
            def read():
                calls.append('counts');now[0]+=0.000010
                return (1,)*64
            def action(name, mode, counts=None):
                calls.append((name,mode));now[0]+=0.000020
                if counts is not None:self.assertEqual(counts,(1,)*64)
            def sync():calls.append('sync');now[0]+=0.000030
            a=CombinationAudit(read,lambda m,c:action('run',m,c),lambda m:action('execute',m),
                               lambda m,c:action('prepare',m,c),sync,clock=lambda:now[0])
            a.call(metric,'direct')
            row=a.samples[0]
            self.assertAlmostEqual(row['counts_host_us'],10)
            self.assertAlmostEqual(row['action_host_us'],20)
            self.assertAlmostEqual(row['fence_host_us'],30 if metric=='audit_dynamic_serial' else 0)
            self.assertAlmostEqual(row['call_host_us'],60 if metric=='audit_dynamic_serial' else 30)
            expected='prepare' if metric=='audit_counts_prepare_only' else 'execute' if metric=='audit_dynamic_prepared' else 'run'
            self.assertEqual(calls[:2],['counts',(expected,'direct')])
            self.assertEqual('sync' in calls,metric=='audit_dynamic_serial')
            a.reset();self.assertEqual(a.samples,[])
    def test_failure_does_not_fabricate_complete_sample(self):
        def fail(*a):raise RuntimeError('injected failure')
        for count_failure in (True,False):
            a=CombinationAudit(fail if count_failure else lambda:(), fail,fail,fail,fail)
            with self.assertRaisesRegex(RuntimeError,'injected failure'):a.call('audit_dynamic','native')
            self.assertEqual(a.samples,[])
    def test_slow_rank_components_stay_paired(self):
        def row(total, counts, action):
            return dict(metric='audit_dynamic',mode='direct',round=0,iterations=2,
                elapsed_host_us=total,final_drain_host_us=2,
                samples=[dict(counts_host_us=counts,action_host_us=action,fence_host_us=0,
                              call_host_us=counts+action)]*2)
        rows=summarize_records([(0,row(20,5,12)),(1,row(15,10,3))])
        result=rows['audit_dynamic','direct',0]
        self.assertEqual(result['rank'],0)
        self.assertEqual(result['counts_us'],5)  # not independently maximal rank1=10
        self.assertEqual(result['action_us'],12)
        self.assertEqual(result['drain_per_call_us'],1)
        self.assertEqual(result['loop_other_us'],2)
    def test_summary_cli_validates_complete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'results').mkdir();(root/'plans').mkdir()
            (root/'exit.status').write_text('0')
            (root/'plans/topology.json').write_text('{}')
            (root/'run.log').write_text(''.join(f'EP8_EID_RANK_PASS rank={r}\n' for r in range(8)))
            for rank in range(8):
                row=dict(metric='audit_dynamic',mode='direct',round=0,iterations=2,
                         elapsed_host_us=20 if rank==0 else 15,final_drain_host_us=2,
                         samples=[dict(counts_host_us=5,action_host_us=8,fence_host_us=0,call_host_us=13)]*2)
                result=dict(rank=rank,backend='participant_eid_ccu_urma',
                    options=dict(suite='benchmark',combination_audit=True,rounds=1,dtype='bfloat16'),
                    reports=[dict(case='external',counts=[0]*64,row_bytes=10240,relay_rows=0,
                                  median_us={'audit_dynamic':{'direct':20}},
                                  samples_us={'audit_dynamic':{'direct':[20]}},rank_local_audit=[row])])
                (root/'results'/f'rank{rank}.json').write_text(json.dumps(result))
            script=Path(__file__).resolve().parents[3]/'scripts/summarize_a5_ccu_combination_audit.py'
            result=subprocess.run([sys.executable,str(script),str(root)],capture_output=True,text=True,check=True)
            self.assertIn('EP8_COMBINATION_AUDIT_SUMMARY_PASS',result.stdout)
            self.assertIn('audit_dynamic direct 20.00 5.00 8.00 0.00 1.00 6.00',result.stdout)
            (root/'exit.status').write_text('1')
            result=subprocess.run([sys.executable,str(script),str(root)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertNotIn('SUMMARY_PASS',result.stdout)
    def test_incomplete_samples_rejected(self):
        with self.assertRaisesRegex(ValueError,'iteration'):
            summarize_records([(0,dict(samples=[],iterations=1))])

if __name__ == '__main__': unittest.main()
