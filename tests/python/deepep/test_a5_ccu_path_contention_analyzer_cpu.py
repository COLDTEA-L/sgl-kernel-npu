#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


FIELDS = (
    "route_id", "src_phy", "dst_phy", "relay_phy", "weight",
    "src_die", "src_port", "relay_die", "relay_port_from_src", "src_edge",
)


def write_manifest(path, src, dst, relay):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        edge = f"edge-{min(src, relay)}-{max(src, relay)}"
        if src < relay:
            physical = dict(src_die=0, src_port=1, relay_die=0,
                            relay_port_from_src=2, src_edge=edge)
        else:
            physical = dict(src_die=0, src_port=2, relay_die=0,
                            relay_port_from_src=1, src_edge=edge)
        writer.writerow(dict(route_id=1000 + relay, src_phy=src, dst_phy=dst,
                             relay_phy=relay, weight=1, **physical))


def write_trace(path, job, durations, starts, launch_only=False):
    path.mkdir(parents=True, exist_ok=True)
    for rank in (0, 1):
        records = []
        for index, (duration, start) in enumerate(zip(durations, starts), 1):
            records.append(dict(iteration=index, started_ns=start + rank,
                                ended_ns=start + int(duration * 1000) + rank,
                                duration_us=duration,
                                timing_semantics="host_submission_only"))
        (path / f"rank{rank}.json").write_text(json.dumps(dict(
            job=job, rank=rank, completion_mode="batch",
            iteration_timing_semantics="batch_only" if launch_only else "host_submission_only",
            iterations=len(durations), loop_body="launch_only", pid=100 + rank,
            batch_started_ns=min(starts) + rank,
            batch_ended_ns=max(
                start + int(duration * 1000)
                for duration, start in zip(durations, starts)
            ) + rank,
            batch_total_us=sum(durations),
            batch_average_us=sum(durations) / len(durations),
            records=[] if launch_only else records)))


class PathContentionAnalyzerTest(unittest.TestCase):
    def test_baseline_resource_comparison_and_missing_device_samples(self):
        repo = Path(__file__).resolve().parents[3]
        analyzer = repo / "scripts/analyze_a5_ccu_multipath_contention.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = dict(sizes=[65536], repeats=1, iterations=2, warmup=100,
                            completion_mode="batch", experiment_mode="baseline-check",
                            mindstudio_profile=True,
                            jobs={"P": {"endpoints": [2, 3]}, "Q": {"endpoints": [6, 7]}})
            (root / "run_settings.json").write_text(json.dumps(settings))
            for job in ("P", "Q"):
                for mode, prefix, duration in (("solo", "solo_baseline", 6), ("resident", "baseline", 9)):
                    phase = f"{prefix}_{job}"
                    write_trace(root / "traces/size_64KB/repeat_1" / phase, job,
                                [duration, duration], [1_000_000, 2_000_000], launch_only=True)
                    for rank in (0, 1):
                        profile = root / "profiles/size_64KB/repeat_1" / phase / f"rank{rank}"
                        profile.mkdir(parents=True)
                        with (profile / "task_time.csv").open("w", newline="") as handle:
                            fields = ("Task Type", "Task Duration(us)", "Start Time(us)", "Task ID", "PID")
                            writer = csv.DictWriter(handle, fieldnames=fields)
                            writer.writeheader()
                            for index in range(2):
                                writer.writerow(dict(zip(fields, ("Ccu", duration + rank, 100 + index, index, 100 + rank))))
            subprocess.run([sys.executable, str(analyzer), "--run-dir", str(root)], check=True)
            summary = json.loads((root / "baseline_resource_summary.json").read_text())
            self.assertTrue(summary["device_profile_complete"])
            self.assertEqual(len(summary["aggregate"]), 4)
            rank0 = summary["aggregate"][0]
            self.assertEqual(rank0["solo_ccu_median_us"], 6)
            self.assertEqual(rank0["resident_ccu_median_us"], 9)
            self.assertEqual(rank0["resident_over_solo_ccu_ratio"], 1.5)
            self.assertTrue((root / "results/size_64KB/baseline_resource_summary.csv").is_file())
            # A missing rank profile must not silently use its host duration.
            (root / "profiles/size_64KB/repeat_1/baseline_P/rank0/task_time.csv").unlink()
            result = subprocess.run([sys.executable, str(analyzer), "--run-dir", str(root)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            summary = json.loads((root / "baseline_resource_summary.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertIsNone(summary["aggregate"][0]["resident_ccu_median_us"])

    def test_synthetic_collision_report(self):
        repo = Path(__file__).resolve().parents[3]
        analyzer = repo / "scripts/analyze_a5_ccu_multipath_contention.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifests").mkdir()
            write_manifest(root / "manifests/P.tsv", 2, 3, 6)
            write_manifest(root / "manifests/Q.tsv", 6, 7, 2)
            (root / "run_settings.json").write_text(json.dumps(dict(
                sizes=[16384], repeats=1, iterations=2, warmup=1,
                completion_mode="batch",
                jobs={
                    "P": {"endpoints": [2, 3], "relay": 6},
                    "Q": {"endpoints": [6, 7], "relay": 2},
                })))
            trace = root / "traces/size_16KB/repeat_1"
            base = 1_000_000_000
            write_trace(trace / "baseline_P", "P", [10, 10], [base, base + 20_000])
            write_trace(trace / "baseline_Q", "Q", [20, 20], [base, base + 30_000])
            write_trace(trace / "concurrent_PQ/P", "P", [15, 15], [base, base + 20_000])
            write_trace(trace / "concurrent_PQ/Q", "Q", [25, 25], [base + 5_000, base + 31_000])
            subprocess.run([sys.executable, str(analyzer), "--run-dir", str(root)], check=True)
            summary = json.loads((root / "path_contention_summary.json").read_text())
            self.assertEqual(summary["shared_physical_directions"], ["2->6", "6->2"])
            self.assertEqual(summary["shared_physical_link"]["p_source_phy"], 2)
            self.assertEqual(summary["shared_physical_link"]["p_source_port"], 1)
            self.assertEqual(summary["shared_physical_link"]["p_relay_phy"], 6)
            self.assertEqual(summary["shared_physical_link"]["p_relay_port"], 2)
            by_job = {row["job"]: row for row in summary["aggregate"]}
            self.assertEqual(by_job["P"]["time_inflation_median"], 1.5)
            self.assertEqual(by_job["Q"]["time_inflation_median"], 1.25)
            self.assertEqual(
                by_job["P"]["conflict_probability"],
                "MINDSTUDIO_TIMELINE_REQUIRED",
            )
            self.assertTrue((root / "path_contention_batches.csv").is_file())
            self.assertTrue((root / "path_contention_submissions.csv").is_file())
            self.assertTrue((root / "path_contention_report.md").is_file())
            size_root = root / "results/size_16KB"
            self.assertTrue((size_root / "path_contention_batches.csv").is_file())
            self.assertTrue((size_root / "path_contention_submissions.csv").is_file())
            self.assertTrue((size_root / "path_contention_summary.csv").is_file())
            self.assertTrue((size_root / "path_contention_report.md").is_file())
            size_summary = json.loads(
                (size_root / "path_contention_summary.json").read_text()
            )
            self.assertEqual(size_summary["bytes_per_peer"], 16384)
            self.assertEqual(len(size_summary["aggregate"]), 2)
            # New launch-only runs retain batch times/counts, not fabricated
            # per-call timestamps. Keep legacy trace support as well.
            for trace_path in trace.rglob("rank*.json"):
                document = json.loads(trace_path.read_text())
                document.update(records=[], iterations=2, loop_body="launch_only",
                                iteration_timing_semantics="batch_only")
                trace_path.write_text(json.dumps(document))
            subprocess.run([sys.executable, str(analyzer), "--run-dir", str(root)], check=True)
            with (root / "path_contention_submissions.csv").open() as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            summary = json.loads((root / "path_contention_summary.json").read_text())
            self.assertEqual(summary["aggregate"][0]["time_inflation_median"], 1.5)


if __name__ == "__main__":
    unittest.main()
