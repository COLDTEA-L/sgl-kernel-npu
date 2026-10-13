#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class H2dIoAnalyzerTest(unittest.TestCase):
    def test_synthetic_complete_run(self):
        repo = Path(__file__).resolve().parents[3]
        analyzer = repo / "scripts/analyze_a5_h2d_io_contention.py"
        cases = ["h2d_solo_rank0", "h2d_solo_rank1", "h2d_both"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            settings = dict(devices=[2, 3], sizes=[1024], cases=cases,
                            warmup=1, iterations=2, repeats=1)
            (root / "run_settings.json").write_text(json.dumps(settings))
            for rank in (0, 1):
                records = []
                for case in cases:
                    active = case == "h2d_both" or case == f"h2d_solo_rank{rank}"
                    if active:
                        average = 15.0 if case == "h2d_both" else 10.0
                        records.append(dict(
                            case=case, direction="h2d", active=True, bytes=1024,
                            repeat=1, iterations=2, start_lateness_us=1.0,
                            started_ns=1000 + rank * 10, ended_ns=31000 + rank * 10,
                            host_total_us=average * 2, host_avg_us=average,
                            device_total_us=average * 2, device_avg_us=average,
                            host_gbps=0.1, device_gbps=0.1, correctness="PASS"))
                    else:
                        records.append(dict(
                            case=case, direction="idle", active=False, bytes=1024,
                            repeat=1, iterations=0, start_lateness_us=1.0,
                            started_ns=1000, ended_ns=1100,
                            correctness="NOT_APPLICABLE"))
                result = dict(
                    rank=rank, world_size=2, physical_device=rank + 2,
                    pinned_host=True, correctness=[
                        {"bytes": 1024, "direction": "h2d", "status": "PASS"},
                        {"bytes": 1024, "direction": "d2h", "status": "PASS"}],
                    records=records)
                (root / "results" / f"rank{rank}.json").write_text(json.dumps(result))
            subprocess.run([sys.executable, str(analyzer), "--run-dir", str(root)], check=True)
            summary = json.loads((root / "h2d_io_summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(len(summary["aggregate"]), 2)
            self.assertTrue(all(row["device_time_inflation_median"] == 1.5
                                for row in summary["aggregate"]))
            self.assertTrue((root / "h2d_io_inflation.csv").is_file())
            self.assertTrue((root / "h2d_io_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
