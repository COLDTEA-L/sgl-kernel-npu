"""Process-start L1 gate: no NPU, no L1 inventory reads when disabled."""
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'python/deep_ep/deep_ep'))
sys.path.insert(0, str(ROOT/'scripts'))
from ccu_l1_config import l1_enabled, select_l1_config, planner_l1_options


class L1SwitchTests(unittest.TestCase):
    def test_explicit_and_legacy_modes(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(l1_enabled())
            self.assertTrue(l1_enabled(True))
            self.assertEqual(planner_l1_options('l1_bottleneck', ['ports'], None),
                             ('l1_bottleneck', ['ports'], None))
            # Preserve legacy validation rather than silently deleting bad input.
            self.assertEqual(planner_l1_options('greedy', ['ports'], None), ('greedy', ['ports'], None))
        with patch.dict(os.environ, {'A5_ENABLE_L1':'0'}, clear=True):
            self.assertFalse(l1_enabled(True))
            self.assertEqual(planner_l1_options('l1_bottleneck', ['ports'], {'bad':1}),
                             ('bottleneck', None, None))
            self.assertEqual(planner_l1_options('greedy', None, None), ('greedy', None, None))
            with redirect_stdout(io.StringIO()):
                self.assertIsNone(select_l1_config(Path('/does/not/exist')))
        with patch.dict(os.environ, {'A5_ENABLE_L1':'1'}, clear=True):
            with self.assertRaises(ValueError): select_l1_config(None)
            self.assertTrue(l1_enabled())
        for value in ('', 'auto', 'true', '2'):
            with patch.dict(os.environ, {'A5_ENABLE_L1':value}, clear=True):
                with self.assertRaises(ValueError): l1_enabled(True)

    def test_peer_validate_off_never_reads_l1_file(self):
        command=[sys.executable,str(ROOT/'scripts/prepare_a5_ccu_peer_plan.py'),
                 '--devices','0,1','--validate-only','--l1-config','/does/not/exist']
        result=subprocess.run(command,env=dict(os.environ,A5_ENABLE_L1='0'),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('L1_SWITCH enabled=0 ignored_config=1',result.stdout)
        self.assertIn('PEER_POLICY_VALIDATED',result.stdout)

    def test_ep8_off_generates_l0_only_without_l1_validation(self):
        import prepare_a5_ccu_ep8_topology as generator
        with tempfile.TemporaryDirectory() as tmp:
            argv=['prepare','--output-dir',tmp,'--l1-config','/does/not/exist']
            topology=dict(version=2,executor='participant_eid_ccu_urma',devices=list(range(8)))
            with patch.dict(os.environ,{'A5_ENABLE_L1':'0'}), patch.object(sys,'argv',argv), \
                 patch.object(generator,'load_machine',return_value=([],[],{})), \
                 patch.object(generator,'build_eid_topology',return_value=([],topology)), \
                 patch('a5_ccu_l1_policy.apply_l1',side_effect=AssertionError('must not read L1')) as apply, \
                 redirect_stdout(io.StringIO()):
                generator.main()
            apply.assert_not_called()
            result=json.loads((Path(tmp)/'topology.json').read_text())
            self.assertNotIn('l1_resources',result)
            self.assertNotIn('l1_single',(Path(tmp)/'peer_plan.tsv').read_text())


if __name__=='__main__': unittest.main()
