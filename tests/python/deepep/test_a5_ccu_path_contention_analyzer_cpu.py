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
        if src == 2:
            physical = dict(src_die=0, src_port=1, relay_die=0,
                            relay_port_from_src=2, src_edge="edge-2-4")
        else:
            physical = dict(src_die=0, src_port=2, relay_die=0,
                            relay_port_from_src=1, src_edge="edge-2-4")
        writer.writerow(dict(route_id=1000 + relay, src_phy=src, dst_phy=dst,
                             relay_phy=relay, weight=1, **physical))


def write_trace(path, job, durations, starts):
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
            iteration_timing_semantics="host_submission_only",
            batch_started_ns=min(starts) + rank,
            batch_ended_ns=max(
                start + int(duration * 1000)
                for duration, start in zip(durations, starts)
            ) + rank,
            batch_total_us=sum(durations),
            batch_average_us=sum(durations) / len(durations),
            records=records)))


class PathContentionAnalyzerTest(unittest.TestCase):
    def test_synthetic_collision_report(self):
        repo = Path(__file__).resolve().parents[3]
        analyzer = repo / "scripts/analyze_a5_ccu_multipath_contention.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifests").mkdir()
            write_manifest(root / "manifests/P.tsv", 2, 3, 4)
            write_manifest(root / "manifests/Q.tsv", 4, 5, 2)
            (root / "run_settings.json").write_text(json.dumps(dict(
                sizes=[16384], repeats=1, iterations=2, warmup=1,
                completion_mode="batch",
                jobs={
                    "P": {"endpoints": [2, 3], "relay": 4},
                    "Q": {"endpoints": [4, 5], "relay": 2},
                })))
            trace = root / "traces/size_16KB/repeat_1"
            base = 1_000_000_000
            write_trace(trace / "baseline_P", "P", [10, 10], [base, base + 20_000])
            write_trace(trace / "baseline_Q", "Q", [20, 20], [base, base + 30_000])
            write_trace(trace / "concurrent_PQ/P", "P", [15, 15], [base, base + 20_000])
            write_trace(trace / "concurrent_PQ/Q", "Q", [25, 25], [base + 5_000, base + 31_000])
            subprocess.run([sys.executable, str(analyzer), "--run-dir", str(root)], check=True)
            summary = json.loads((root / "path_contention_summary.json").read_text())
            self.assertEqual(summary["shared_physical_directions"], ["2->4", "4->2"])
            self.assertEqual(summary["shared_physical_link"]["card2_port"], 1)
            self.assertEqual(summary["shared_physical_link"]["card4_port"], 2)
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


if __name__ == "__main__":
    unittest.main()
