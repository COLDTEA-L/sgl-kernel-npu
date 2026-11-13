"""Launcher validation only; never claims device/graph execution."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parents[3] / 'scripts/run_a5_relay_graph_gate.py'


class LauncherValidation(unittest.TestCase):
    def run_case(self, **changes):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / 'relay.tsv'
            manifest.write_text('relay_phy\tsrc_phy\tdst_phy\n0\t2\t3\n1\t2\t3\n')
            args = {'binary': '/bin/true', 'devices': '2,3', 'relay-phys': '0,1',
                    'relay-manifest': str(manifest), 'notify': 'CALLBACK', 'mode': 'GRAPH',
                    'iters': '4', 'output-root': temp}
            args.update(changes)
            command = [sys.executable, str(RUNNER)]
            for key, value in args.items():
                command.extend(['--' + key, value])
            return subprocess.run(command, capture_output=True, text=True, timeout=10)

    def test_relay_cannot_be_endpoint(self):
        result = self.run_case(**{'relay-phys': '2,1'})
        self.assertEqual(result.returncode, 2)
        self.assertIn('distinct endpoints', result.stderr)

    def test_manifest_order_must_match(self):
        result = self.run_case(**{'relay-phys': '1,0'})
        self.assertEqual(result.returncode, 2)
        self.assertIn('relay order differs', result.stderr)

    def test_manifest_endpoints_must_match(self):
        result = self.run_case(devices='3,2')
        self.assertEqual(result.returncode, 2)
        self.assertIn('endpoint order differs', result.stderr)

    def test_zero_exit_without_rank_markers_is_failure(self):
        result = self.run_case()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing completion accounting', result.stderr)


if __name__ == '__main__':
    unittest.main()
