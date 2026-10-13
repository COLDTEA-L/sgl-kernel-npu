#!/usr/bin/env python3
"""CPU-only protocol-analysis gates; no NPU imports or performance claims."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_a5_ccu_channel_sync import MODES, analyze


def fixture(root):
    entries = []
    for mode in MODES:
        run = root / mode
        (run / "plans").mkdir(parents=True)
        (run / "cases").mkdir()
        (run / "r1/results").mkdir(parents=True)
        (run / "plans/peer_plan.json").write_text(json.dumps(dict(devices=[6, 7], relay_limit_per_peer=6,
            channels_per_rank=[7,7])))
        settings = dict(repeats=1, self_copy_mode="ccu", sync_mode="batch", graph_backend="none",
            warmup=100, iterations=20, bytes_per_peer=4194304, dtype="float32", channel_sync_mode=mode)
        (run / "run_settings.json").write_text(json.dumps(settings))
        lines = []
        for rank in range(2):
            result = dict(rank=rank, ranks=2, pid=100+rank, correctness="PASS", iterations=20,
                host_call_us=[], host_avg_us=10, host_batch_total_us=200, sync_mode="batch",
                self_copy_mode="ccu", channel_sync_mode=mode, graph_backend="none", warmup=100,
                bytes_per_peer=4194304, dtype="float32", element_bytes=4,
                batch_validation=dict(status="PASS", rounds=2, calls_per_round=20))
            (run / f"r1/results/rank{rank}.json").write_text(json.dumps(result))
            lines += [f"CCU_LOCAL_REMOTE_PLAN handle=1 rank={rank} enabled=1 self_copy=GroupCopy8x32K",
                      f"CHANNEL_SYNC_PLAN rank={rank} mode={mode} channels=7"]
            profile = run / f"r1/profiling/rank{rank}/test_ascend_pt/ASCEND_PROFILER_OUTPUT"
            profile.mkdir(parents=True)
            events = [dict(name="Ccu", ph="X", cat="Ascend Hardware", pid=0, tid=1,
                ts=i*100, dur=40 if mode=="baseline" else 30, args=dict(pid=100+rank, taskId=i))
                for i in range(20)]
            # A host launch API must never be counted as a hardware Ccu.
            events += [dict(name="HcclCcuKernelLaunch", ph="X", cat="CANN", dur=100)]
            (profile / "trace_view.json").write_text(json.dumps(events))
        (run / "cases/peer_plan_r1.log").write_text("\n".join(lines))
        (run / "cases/peer_plan_r1.status").write_text("0")
        entries.append(dict(mode=mode, status="0", run_dir=str(run)))
    with (root / "runs.tsv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=("mode", "status", "run_dir"), delimiter="\t")
        writer.writeheader(); writer.writerows(entries)


class ProtocolTests(unittest.TestCase):
    def test_complete_formal_profile_and_separate_host_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            result = analyze(root)
            self.assertTrue(result["functional_complete"])
            self.assertTrue(result["device_profile_complete"])
            self.assertAlmostEqual(result["baseline_speedup"]["packed"], 4/3)
            self.assertEqual(result["rows"][0]["device_samples"], 20)
            self.assertEqual(result["rows"][0]["host_avg_us"], 10)

    def test_extra_task_rejects_speedup_but_keeps_correctness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            trace = next((root / "packed/r1/profiling/rank1").rglob("trace_view.json"))
            data = json.loads(trace.read_text()); data.append(dict(name="Ccu", ph="X",
                cat="Ascend Hardware", ts=9999, dur=30, args=dict(pid=101, taskId=30)))
            trace.write_text(json.dumps(data))
            result = analyze(root)
            self.assertTrue(result["functional_complete"])
            self.assertFalse(result["device_profile_complete"])
            self.assertEqual(result["baseline_speedup"], {})

    def test_missing_protocol_marker_and_changed_settings_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            log = root / "packed/cases/peer_plan_r1.log"
            log.write_text(log.read_text().replace("CHANNEL_SYNC_PLAN rank=1", "NOT_A_PROTOCOL rank=1"))
            result = analyze(root)
            self.assertFalse(result["functional_complete"])
            self.assertEqual(result["baseline_speedup"], {})
            settings = root / "packed-ready/run_settings.json"
            data = json.loads(settings.read_text()); data["bytes_per_peer"] *= 2
            settings.write_text(json.dumps(data))
            self.assertFalse(analyze(root)["configuration_consistent"])

    def test_invalid_mode_rejected_before_runtime(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/run_a5_ccu_peer_plan_alltoall.sh"),
            "--channel-sync-mode", "typo"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("runtime_bootstrap", result.stdout)
        result = subprocess.run(["bash", str(ROOT / "scripts/run_a5_ccu_channel_sync_perf.sh"), "--help"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn("compute 6,7", result.stdout)

    def test_actual_channel_count_must_match_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            log = root / "packed/cases/peer_plan_r1.log"
            log.write_text(log.read_text().replace("channels=7", "channels=5"))
            result = analyze(root)
            self.assertFalse(result["functional_complete"])
            self.assertEqual(result["baseline_speedup"], {})


if __name__ == "__main__":
    unittest.main()
