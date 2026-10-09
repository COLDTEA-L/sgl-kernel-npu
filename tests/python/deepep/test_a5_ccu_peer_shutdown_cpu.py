"""Test the shutdown observer's exact process scope without NPU or ptrace."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from watch_a5_ccu_peer_shutdown import matching_processes, log_state, snapshot, watch


class ShutdownObserverTests(unittest.TestCase):
    @unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux /proc required")
    def test_live_observer_on_owned_cpu_only_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp).resolve() / "r1"; case.mkdir()
            log = Path(tmp) / "case.log"
            log.write_text("\n".join("RESULT_JSON " + json.dumps(dict(
                rank=rank, ranks=4, correctness="PASS")) for rank in range(4)))
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)",
                "test_a5_ccu_peer_plan_alltoall.py", "--run-dir", str(case)])
            try:
                watch(case, log, 4, os.getpid(), grace=0, max_seconds=5)
                scope = json.loads((case / "shutdown_trace/scope.json").read_text())
                self.assertEqual([p["pid"] for p in scope["processes"]], [child.pid])
                snapshots = json.loads((case / "shutdown_trace/proc_snapshot0.json").read_text())
                self.assertEqual(snapshots[0]["pid"], child.pid)
                self.assertIn("threads", snapshots[0])
                self.assertIsNone(child.poll())  # read-only observer did not terminate it
            finally:
                child.terminate()
                child.wait(timeout=5)

    def test_scope_excludes_other_jobs_and_same_path_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = (root / "run/r1").resolve()
            proc = root / "proc"; proc.mkdir()
            for pid, path in ((900001, str(case)), (900002, str(case) + "_other")):
                base = proc / str(pid); base.mkdir()
                (base / "cmdline").write_bytes(
                    ("python3\0/repo/test_a5_ccu_peer_plan_alltoall.py\0--run-dir\0" + path + "\0").encode())
                (base / "stat").write_text(str(pid) + " (python) S " + " ".join(["0"] * 18 + ["12345"]))
            targets = matching_processes(case, proc)
            self.assertEqual([target["pid"] for target in targets], [900001])
            self.assertEqual(targets[0]["starttime"], "12345")
            # Never examine a recycled PID with a different process start time.
            (proc / "900001/stat").write_text("900001 (python) S " + " ".join(["0"] * 18 + ["98765"]))
            self.assertEqual(snapshot(targets[0], proc)["state"], "EXITED_OR_IDENTITY_CHANGED")

    def test_observer_requires_results_from_every_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "case.log"
            entries = ["RESULT_JSON " + json.dumps(dict(rank=rank, ranks=4, correctness="PASS"))
                       for rank in range(4)]
            log.write_text("\n".join(entries[:3]))
            self.assertFalse(log_state(log, 4)[0])
            log.write_text("\n".join(entries) + "\nPEER_CASE_PHASE rank=2 pid=101 phase=python_atexit\n")
            complete, phases = log_state(log, 4)
            self.assertTrue(complete)
            self.assertEqual(phases["2"], "python_atexit")


if __name__ == "__main__":
    unittest.main()
