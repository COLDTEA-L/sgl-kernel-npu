#!/usr/bin/env python3
"""No NPU: production graph model and fail-closed CCU result analysis."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_a5_ccu_peer_plan import analyze


class CombinedTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "g++ required for representation model")
    def test_production_graph_byte_layouts_and_completion_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = str(Path(tmp) / "model")
            subprocess.run(["g++", "-std=c++14", "-Wall", "-Wextra", "-Werror",
                "-I" + str(ROOT / "tests/cpp/fake_ccu"),
                str(ROOT / "tests/cpp/test_a5_ccu_local_remote_kernel.cpp"),
                str(ROOT / "examples/a5_ccu_urma_route_probe/op_kernel_ccu/route_kernel.cc"),
                "-o", binary], check=True, timeout=60)
            result = subprocess.run([binary], check=True, text=True, capture_output=True, timeout=30)
            self.assertIn("CPU_MODEL_PASS", result.stdout)

    def test_combined_analysis_requires_every_rank_and_real_mode_marker(self):
        for k in (2, 4):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "plans").mkdir(); (root / "cases").mkdir()
                (root / "plans/peer_plan.json").write_text(json.dumps(dict(devices=list(range(k)), relay_limit_per_peer=1)))
                (root / "run_settings.json").write_text(json.dumps(dict(repeats=1, self_copy_mode="ccu")))
                log = root / "cases/peer_plan_r1.log"
                status = log.with_suffix(".status"); status.write_text("0")
                results = [dict(rank=r, ranks=k, correctness="PASS", iterations=2,
                    host_call_us=[], host_avg_us=10, host_batch_total_us=20, sync_mode="batch",
                    self_copy_mode="ccu", batch_validation=dict(status="PASS", rounds=2, calls_per_round=2))
                    for r in range(k)]
                lines = ["RESULT_JSON " + json.dumps(r) for r in results]
                markers = [f"CCU_LOCAL_REMOTE_PLAN handle=1 rank={r} enabled=1 self_copy=LocalCopyNb" for r in range(k)]
                log.write_text("\n".join(lines))
                self.assertFalse(analyze(root)["complete"])
                log.write_text("\n".join(lines + markers[:-1]))
                self.assertFalse(analyze(root)["complete"])
                log.write_text("\n".join(lines + markers))
                self.assertTrue(analyze(root)["complete"])
                status.write_text("137")
                self.assertFalse(analyze(root)["complete"])

    def test_ccu_cli_help_and_invalid_policy_stays_outside_runtime(self):
        launcher = str(ROOT / "scripts/run_a5_ccu_peer_plan_alltoall.sh")
        help_run = subprocess.run(["bash", launcher, "--help"], text=True, capture_output=True, timeout=10)
        self.assertEqual(help_run.returncode, 0)
        self.assertIn("one CCU local+remote graph", help_run.stdout)
        invalid = subprocess.run(["bash", launcher, "--self-copy-mode", "ccu",
            "--devices", "2,3", "--relay-phys", "2", "--cann-root", "/not-present/cann"],
            text=True, capture_output=True, timeout=10)
        self.assertEqual(invalid.returncode, 2)
        self.assertNotIn("runtime_bootstrap", invalid.stdout)


if __name__ == "__main__":
    unittest.main()
