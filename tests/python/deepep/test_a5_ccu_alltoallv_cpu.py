#!/usr/bin/env python3
"""Compile and schedule the real EP8 CCU graph with CPU stand-ins."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_a5_ccu_peer_plan import build_rows, validate_policy


class AllToAllVTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "g++ required")
    def test_real_graph_asymmetric_zero_lengths_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = str(Path(tmp) / "model")
            subprocess.run([
                "g++", "-std=c++14", "-Wall", "-Wextra", "-Werror",
                "-I" + str(ROOT / "tests/cpp/fake_ccu"),
                str(ROOT / "tests/cpp/test_a5_ccu_alltoallv.cpp"),
                str(ROOT / "examples/a5_ccu_urma_route_probe/op_kernel_ccu/alltoallv_kernel.cc"),
                str(ROOT / "tests/cpp/group_copy_model_stub.cc"), "-o", binary,
            ], check=True, timeout=60)
            result = subprocess.run([binary], check=True, capture_output=True, text=True, timeout=30)
            self.assertIn("ALLTOALLV_CPU_MODEL_PASS launches=800 epochs=100", result.stdout)

    def test_eight_direct_peers_and_no_relays(self):
        for devices in (list(range(8)), list(reversed(range(8)))):
            rows, info = build_rows(devices, devices, {}, [], [])
            self.assertEqual(len(rows), 28)
            self.assertEqual(info["channels_per_rank"], [7] * 8)
            self.assertTrue(all(row["kind"] == "direct" for row in rows))
        with self.assertRaises(ValueError):
            validate_policy(list(range(8)), list(range(16)), {"0,1": [8]})


if __name__ == "__main__":
    unittest.main()
