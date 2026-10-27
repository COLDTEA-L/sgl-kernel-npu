"""Failure summaries must preserve native errors and isolate this run's workers."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/collect_a5_ccu_failure_context.py"
spec = importlib.util.spec_from_file_location("ccu_failure_context", SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class FailureContextTests(unittest.TestCase):
    def test_own_pids_only_and_channel_error_survives_endpoint_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "case.log"
            log.write_text('PEER_CASE_PHASE rank=0 pid=123 phase=prepare_plan\n'
                           'CCU_PG_MODE raw_mode=0 diagnostic_only=1\n')
            plog = root / "plog-123_20261027.log"
            plog.write_text('[ERROR] HcclChannelAcquire opExpansionMode[2] rejected\n' +
                            '[ERROR] GetEndpointInfo failed\n' * 100)
            (root / "plog-1234_20261027.log").write_text('[ERROR] unrelated process\n')
            result = collector.collect(log, [root, root])
            self.assertIn('raw_mode=0', result)
            self.assertIn('opExpansionMode[2] rejected', result)
            self.assertNotIn('unrelated process', result)
            self.assertEqual(result.count('opExpansionMode[2] rejected'), 1)
            self.assertLess(len(result.splitlines()), 20)

    def test_stale_pid_reuse_log_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / 'case.log'
            log.write_text('PEER_CASE_PHASE rank=0 pid=123 phase=prepare_plan\n')
            stale = root / 'plog-123_old.log'
            stale.write_text('[ERROR] old attempt\n')
            os.utime(stale, (1, 1))
            result = collector.collect(log, [root], since=2)
            self.assertNotIn('old attempt', result)
            self.assertIn('CCU_PLOG_UNAVAILABLE', result)

    def test_no_worker_pid_does_not_read_other_plogs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / 'case.log'
            log.write_text('torchrun launcher failed\n')
            (root / 'plog-123_other.log').write_text('[ERROR] other task\n')
            result = collector.collect(log, [root])
            self.assertIn('worker_pids=[]', result)
            self.assertNotIn('other task', result)


if __name__ == '__main__':
    unittest.main()
