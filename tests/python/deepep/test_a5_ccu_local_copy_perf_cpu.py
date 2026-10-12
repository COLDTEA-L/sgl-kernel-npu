"""Synthetic profiling/phase regression only, not an NPU performance test."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_a5_ccu_local_copy_perf import analyze, parse_device


class PerfTests(unittest.TestCase):
    def setup_run(self, root, profile=True):
        settings = dict(devices=[2,3], benchmark=True, builder_only=False, profile=profile,
            warmup=100, iterations=20, cases=[dict(name='direct_4194304', copy_mode='direct', bytes=4194304)])
        (root/'run_settings.json').write_text(json.dumps(settings))
        (root/'local_copy_summary.json').write_text('{"complete":true}')
        directory=root/'cases/direct_4194304'; directory.mkdir(parents=True)
        for rank in (0,1):
            event=dict(rank=rank,pid=100+rank,copy_mode='direct',bytes=4194304,warmup=100,iterations=20,
                host_batch_avg_us=90.0,host_batch_total_us=1800.0,profile_requested=profile,
                formal_h2d_d2h=False,remote_write=False,engine='CCU',post_check='PASS')
            (directory/f'rank{rank}.log').write_text('LOCAL_COPY_PERF_JSON '+json.dumps(event)+'\n')
        return settings,directory

    def traces(self, directory, rank, count=20, memcpy=False):
        prof=directory/'profiling'/f'rank{rank}'/'PROF_FAKE'/'mindstudio_profiler_output'
        prof.mkdir(parents=True,exist_ok=True)
        events=[dict(name='Ccu',ph='X',pid=rank,tid=0,ts=i*100,dur=80+i%2,
            args=dict(taskId=i,pid=100+rank,size=4294967295)) for i in range(count)]
        # These host/API records MUST NOT enter CCU device statistics.
        events += [dict(name='HcclCcuKernelLaunch',ph='X',ts=9999,dur=1),
                   dict(name='Ccu',ph='X',cat='CANN',ts=9990,dur=3)]
        if memcpy: events.append(dict(name='MemcpyAsync',ph='X',cat='Ascend Hardware',ts=9999,dur=4))
        file=prof/'msprof_fake.json'; file.write_text(json.dumps(dict(traceEvents=events)))
        return file

    def test_exact_device_samples_and_no_host_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,directory=self.setup_run(root)
            self.assertFalse(analyze(root)['complete'])
            for rank in (0,1): self.traces(directory,rank)
            result=analyze(root)
            self.assertTrue(result['complete'])
            self.assertAlmostEqual(result['cases'][0]['ccu_mean_us'],80.5)
            self.assertAlmostEqual(result['cases'][0]['payload_GB_s'],4194304/80500)
            self.traces(directory,1,count=19); self.assertFalse(analyze(root)['complete'])
            self.traces(directory,1,count=120); self.assertFalse(analyze(root)['complete'])
            self.traces(directory,1,memcpy=True); self.assertFalse(analyze(root)['complete'])

    def test_duplicate_exports_dedup_and_csv_unit_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,directory=self.setup_run(root)
            for rank in (0,1):
                file=self.traces(directory,rank)
                file.with_name('msprof_duplicate.json').write_text(file.read_text())
            self.assertTrue(analyze(root)['complete'])
            prof=directory/'profiling/rank0'
            file=prof/'task_time_test.csv'
            with file.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['Task Type','Task Duration(us)','Start Time(us)','Task Id','PID'])
                writer.writeheader()
                for i in range(20): writer.writerow({'Task Type':'CCU','Task Duration(us)':10+i,
                    'Start Time(us)':i*100,'Task Id':i,'PID':100})
            records,_,_=parse_device(prof,100)
            self.assertEqual(len(records),20); self.assertEqual(records[0]['duration_us'],10)
            file.write_text(file.read_text().replace('Duration(us)','Duration(ns)'))
            records,_,_=parse_device(prof,100)
            self.assertEqual(records[0]['format'],'chrome_us')  # never read ns as us

    def test_bad_timing_or_failed_correctness_never_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,directory=self.setup_run(root)
            for rank in (0,1): self.traces(directory,rank)
            original=(directory/'rank0.log').read_text()
            for text in (original.replace('"iterations": 20','"iterations": true'),
                         original.replace('"host_batch_avg_us": 90.0','"host_batch_avg_us": NaN'),
                         original.replace('"formal_h2d_d2h": false','"formal_h2d_d2h": true'),
                         original+original):
                (directory/'rank0.log').write_text(text); self.assertFalse(analyze(root)['complete'])
            (directory/'rank0.log').write_text(original)
            (root/'local_copy_summary.json').write_text('{"complete":false}')
            self.assertFalse(analyze(root)['complete'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.setup_run(root,profile=False)
            self.assertEqual(analyze(root)['conclusion'],'FUNCTIONAL_HOST_ONLY')

    def test_benchmark_has_only_launches_inside_timing(self):
        source=(ROOT/'examples/a5_ccu_urma_route_probe/testcase/local_copy_probe.cc').read_text()
        section=source[source.index('void Benchmark('):source.index('void Run()')]
        formal=section[section.index('const auto begin ='):section.index('profile.Stop();')]
        self.assertIn('HcclCcuKernelLaunch',formal)
        self.assertNotIn('aclrtMemcpy',formal)
        self.assertEqual(formal.count('aclrtSynchronizeStream('),1)
        self.assertLess(section.index('profile.Stop();'),section.index('"post-profile output"'))

    def test_perf_bad_arguments_stop_before_runtime(self):
        script=ROOT/'scripts/run_a5_ccu_local_copy_perf.sh'
        for args in (['--warmup','-1'],['--iters','0'],['--devices','2,2']):
            result=subprocess.run(['bash',str(script),'--cann-root','/not-present']+args,
                text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertNotIn('Result directory:',result.stdout)


if __name__=='__main__': unittest.main()
