"""Import/CLI/lifecycle checks; no NPU is loaded by these tests."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_a5_ccu_peer_plan import valid_result


class CopyOverlapTests(unittest.TestCase):
    def test_cli_rejects_bad_copy_mode_before_runtime(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name(
            "test_a5_ccu_peer_plan_alltoall.py")), "--self-copy-mode", "unsafe"],
            text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("runtime_bootstrap", result.stdout)

    def test_mode_metadata_rejects_unknown_values(self):
        base = dict(rank=0, ranks=2, correctness="PASS", host_call_us=[10],
                    iterations=1, host_avg_us=10)
        for mode in ("serial", "overlap"):
            self.assertTrue(valid_result(dict(base, self_copy_mode=mode), 2))
        self.assertFalse(valid_result(dict(base, self_copy_mode="unsafe"), 2))

    def test_worker_releases_graph_before_copy_resources_and_process_group(self):
        source = Path(__file__).with_name("test_a5_ccu_peer_plan_alltoall.py").read_text()
        self.assertLess(source.index('phase("final_synchronize_done")'), source.index("graph.reset()"))
        self.assertLess(source.index("graph.reset()"), source.index("status = destroy_plan(handle)"))
        self.assertLess(source.index("status = destroy_plan(handle)"), source.index("dist.destroy_process_group()"))
        self.assertIn("graph_output.fill_(-128)", source)

    def test_overlap_does_not_restore_combined_ccu_kernel(self):
        runtime = (ROOT / "examples/a5_ccu_urma_route_probe/op_host/peer_plan_runtime.cc").read_text()
        self.assertIn("SelfCopyGuard copy(r.selfCopy)", runtime)
        self.assertIn("copy.Finish(status)", runtime)
        self.assertNotIn("ALLTOALL_CONCURRENT", runtime)
        lane = (ROOT / "examples/a5_ccu_urma_route_probe/op_host/self_copy_lane.cc").read_text()
        self.assertIn('"aclrtCreateEventExWithFlag"', lane)
        self.assertNotIn("aclrtResetEvent(", lane)
        self.assertIn("lane poisoned", lane)


if __name__ == "__main__":
    unittest.main()
