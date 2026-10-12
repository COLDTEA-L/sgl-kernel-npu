"""CPU-only checks for local-copy proof gates; never loads an NPU runtime."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_a5_ccu_local_copy_probe import analyze


class LocalCopyProofTests(unittest.TestCase):
    def populate(self, root):
        settings = dict(devices=[2, 3], changed_calls=3, queued_calls=20, builder_only=False,
                        cases=[dict(name='direct_4096', copy_mode='direct', bytes=4096)])
        (root / 'run_settings.json').write_text(json.dumps(settings))
        case = root / 'cases/direct_4096'; case.mkdir(parents=True)
        (case / 'status.txt').write_text('0')
        for rank in (0, 1):
            event = dict(rank=rank, copy_mode='direct', bytes=4096, changed_calls=3, queued_calls=20,
                         guards='PASS', status='PASS', engine='CCU', channel_count=0, acl_d2d=False)
            (case / f'rank{rank}.log').write_text('LOCAL_COPY_RESULT_JSON ' + json.dumps(event) +
                f'\nLOCAL_COPY_WORKER_PASS rank={rank} data+guards+cleanup=PASS\n')
            (case / f'rank{rank}.status').write_text('0')
        return settings, case

    def test_pass_requires_data_and_cleanup_from_every_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, case = self.populate(root)
            self.assertTrue(analyze(root)['complete'])
            (case / 'rank1.status').write_text('137')
            self.assertFalse(analyze(root)['complete'])
            (case / 'rank1.status').write_text('0')
            (case / 'rank1.log').write_text('LOCAL_COPY_BUILDER_PASS\n')
            self.assertFalse(analyze(root)['complete'])

    def test_results_fail_closed_for_corruption_or_partial_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); settings, case = self.populate(root)
            original = (case / 'rank1.log').read_text()
            for replacement in ('LOCAL_COPY_RESULT_JSON {bad\n', original + 'LOCAL_COPY_FAILURE late error\n',
                                original.replace('"queued_calls": 20', '"queued_calls": 0'),
                                original.replace('"acl_d2d": false', '"acl_d2d": 0'),
                                original.replace('"channel_count": 0', '"channel_count": false'),
                                original + original):
                (case / 'rank1.log').write_text(replacement)
                self.assertFalse(analyze(root)['complete'])
            (case / 'rank1.log').write_text(original)
            settings['builder_only'] = True
            (root / 'run_settings.json').write_text(json.dumps(settings))
            self.assertFalse(analyze(root)['complete'])
            settings['builder_only'] = False
            settings['cases'].append(dict(name='direct_4194304', copy_mode='direct', bytes=4194304))
            (root / 'run_settings.json').write_text(json.dumps(settings))
            result = analyze(root)
            self.assertFalse(result['complete'])
            self.assertEqual(result['hardware_data_pass_cases'], ['direct_4096'])

    def test_bad_arguments_stop_before_runtime_or_file_creation(self):
        script = ROOT / 'scripts/run_a5_ccu_local_copy_probe.sh'
        for args in (['--devices', '2,2'], ['--sizes', '0'], ['--queued-iters', '0'],
                     ['--sizes', '256', '--copy-mode', 'both'], ['--copy-mode', 'unknown']):
            result = subprocess.run(['bash', str(script), '--cann-root', '/not-present'] + args,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertNotIn('Result directory:', result.stdout)

    def test_local_kernel_has_no_remote_or_acl_d2d_path(self):
        source = (ROOT / 'examples/a5_ccu_urma_route_probe/testcase/local_copy_probe.cc').read_text()
        self.assertIn('LocalCopyNb(destination, source, bytes, done)', source)
        self.assertIn('LocalCopyNb(destination, source, chunk, done)', source)
        self.assertIn('WaitEvent(done)', source)
        self.assertNotIn('WriteNb(', source)
        self.assertNotIn('HcclChannelAcquire(', source)
        self.assertNotIn('ACL_MEMCPY_DEVICE_TO_DEVICE', source)
        self.assertNotIn('aclrtResetDevice(', source)
        queued = source[source.index('for (uint32_t index = 0; index < options.queued; ++index)\n        enqueue'):]
        self.assertLess(queued.index('enqueue('), queued.index('aclrtSynchronizeStream('))


if __name__ == '__main__':
    unittest.main()
