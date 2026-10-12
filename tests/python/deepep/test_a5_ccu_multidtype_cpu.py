"""Multi-dtype helpers and analysis; no NPU initialization."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

from a5_ccu_peer_test_support import PAYLOAD_DTYPES, payload_code

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analyze_a5_ccu_peer_plan import valid_result


class MultiDtypeTests(unittest.TestCase):
    def test_payload_is_exact_small_integer_and_changes_each_epoch(self):
        for src in range(4):
            for dst in range(4):
                signatures = []
                for epoch in range(50):
                    values = tuple(payload_code(i, src, dst, epoch) for i in range(31))
                    self.assertTrue(all(-63 <= value <= 63 for value in values))
                    signatures.append(values)
                self.assertEqual(len(set(signatures)), len(signatures))

    def test_result_type_metadata_is_validated(self):
        base = dict(rank=0, ranks=2, correctness="PASS", iterations=1,
                    host_call_us=[10], host_avg_us=10)
        for dtype, size in PAYLOAD_DTYPES.items():
            self.assertTrue(valid_result(dict(base, dtype=dtype, element_bytes=size), 2))
            self.assertFalse(valid_result(dict(base, dtype=dtype, element_bytes=3), 2))
        self.assertFalse(valid_result(dict(base, dtype="float64"), 2))

    def test_cli_rejects_unsupported_type_or_partial_element_before_runtime(self):
        script = str(Path(__file__).with_name("test_a5_ccu_peer_plan_alltoall.py"))
        for args in (("--dtype", "float64"), ("--dtype", "float16", "--bytes", "3")):
            result = subprocess.run([sys.executable, script, "--runtime-check", *args],
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("runtime_bootstrap", result.stdout)

    def test_no_float_only_pattern_or_byte_count_remains_in_peer_worker(self):
        source = Path(__file__).with_name("test_a5_ccu_peer_plan_alltoall.py").read_text()
        ast.parse(source)
        self.assertNotIn("args.bytes // 4", source)
        self.assertNotIn("1000000", source)
        self.assertNotIn("fill_(-999)", source)
        self.assertIn("input_for(value) for value in epochs", source)
        self.assertIn("check(snapshots[index], epochs[index])", source)


if __name__ == "__main__":
    unittest.main()
