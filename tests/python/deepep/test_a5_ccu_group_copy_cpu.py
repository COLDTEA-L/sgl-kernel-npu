#!/usr/bin/env python3
"""GroupCopy host geometry checks; these are not hardware throughput tests."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]

class GroupCopyTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "g++ required")
    def test_bulk_tail_and_max_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / "plan")
            subprocess.run(["g++", "-std=c++14", "-Wall", "-Wextra", "-Werror",
                str(ROOT / "tests/cpp/test_a5_ccu_group_copy_plan.cpp"), "-o", exe],
                check=True, timeout=60)
            out = subprocess.run([exe], text=True, capture_output=True, check=True, timeout=20)
            self.assertIn("GROUP_PLAN_CPU_PASS", out.stdout)

    def test_contiguous_resources_and_read_write_waits(self):
        text = (ROOT / "examples/a5_ccu_urma_route_probe/op_kernel_ccu/grouped_local_copy_kernel.cc").read_text()
        for name in ("CreateBlockCcuBuf", "CreateBlockCompletedEvent", "CreateBlockExecutor", "LoopGroupCall"):
            self.assertIn(name, text)
        self.assertEqual(text.count("GROUP_CHECK(WaitEvent(events[i]))"), 2)
        self.assertNotIn("aclrtMemcpy", text)

    def test_standalone_group_cli(self):
        p = subprocess.run(["bash", str(ROOT / "scripts/run_a5_ccu_local_copy_probe.sh"), "--help"],
            text=True, capture_output=True, timeout=10)
        self.assertEqual(p.returncode, 0)
        self.assertIn("group", p.stdout)

if __name__ == "__main__":
    unittest.main()
