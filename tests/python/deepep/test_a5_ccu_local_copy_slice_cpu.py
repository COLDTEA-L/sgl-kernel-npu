"""Geometry/result policy regression, not device local-copy performance."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_a5_ccu_local_copy_probe import analyze as proof
from analyze_a5_ccu_local_copy_perf import analyze as perf

class SliceTests(unittest.TestCase):
    def test_geometry_is_required_and_exact_in_both_results(self):
        for slice_bytes in (32768,65536,131072):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                lanes=262144//slice_bytes
                case=dict(name=f'group_l{lanes}_s{slice_bytes//1024}k_4194304',copy_mode='group',bytes=4194304,
                    group_slice_bytes=slice_bytes,group_lanes=lanes,group_ms_budget=64)
                settings=dict(devices=[2],cases=[case],changed_calls=3,queued_calls=20,
                    warmup=100,iterations=20,benchmark=True,profile=False,builder_only=False)
                (root/'run_settings.json').write_text(json.dumps(settings))
                directory=root/'cases'/case['name']; directory.mkdir(parents=True)
                (directory/'status.txt').write_text('0'); (directory/'rank0.status').write_text('0')
                result=dict(rank=0,copy_mode='group',bytes=4194304,changed_calls=3,queued_calls=20,
                    guards='PASS',status='PASS',engine='CCU',channel_count=0,acl_d2d=False,
                    group_slice_bytes=slice_bytes,group_lanes=lanes,group_ms_budget=64)
                timing=dict(rank=0,pid=100,copy_mode='group',bytes=4194304,warmup=100,iterations=20,
                    host_batch_total_us=600.,host_batch_avg_us=30.,profile_requested=False,
                    formal_h2d_d2h=False,remote_write=False,engine='CCU',post_check='PASS',
                    group_slice_bytes=slice_bytes,group_lanes=lanes,group_ms_budget=64)
                def write(r,t):
                    (directory/'rank0.log').write_text('LOCAL_COPY_RESULT_JSON '+json.dumps(r)+'\n'+
                        'LOCAL_COPY_PERF_JSON '+json.dumps(t)+'\nLOCAL_COPY_WORKER_PASS rank=0 cleanup=PASS\n')
                write(result,timing)
                self.assertTrue(proof(root)['complete']); self.assertTrue(perf(root)['complete'])
                for bad in (32769,True,None):
                    write(dict(result,group_slice_bytes=bad),timing)
                    self.assertFalse(proof(root)['complete'])
                    write(result,dict(timing,group_slice_bytes=bad))
                    self.assertTrue(proof(root)['complete']); self.assertFalse(perf(root)['complete'])
                stale=dict(result); stale.pop('group_slice_bytes')
                write(stale,timing); self.assertFalse(proof(root)['complete'])
                for key in ('group_lanes','group_ms_budget'):
                    write(dict(result,**{key:lanes+1}),timing)
                    self.assertFalse(proof(root)['complete'])
                    write(result,dict(timing,**{key:lanes+1}))
                    self.assertTrue(proof(root)['complete']); self.assertFalse(perf(root)['complete'])

    def test_bad_slices_stop_before_runtime(self):
        for mode,slices in (('group','16'),('group','64,64'),('group','256'),('direct','64')):
            p=subprocess.run(['bash',str(ROOT/'scripts/run_a5_ccu_local_copy_probe.sh'),
                '--copy-mode',mode,'--group-slices-kib',slices,'--cann-root','/not-present'],
                text=True,capture_output=True,timeout=10)
            self.assertEqual(p.returncode,2,p.stderr)
            self.assertNotIn('Result directory:',p.stdout)
            self.assertNotIn('missing CANN',p.stderr)

if __name__=='__main__': unittest.main()
