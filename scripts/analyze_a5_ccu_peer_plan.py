#!/usr/bin/env python3
"""Do not turn an exit status or incomplete rank set into correctness PASS."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def analyze(root):
    plan = json.loads((root / "plans/peer_plan.json").read_text())
    k = len(plan["devices"])
    rows, complete = [], True
    settings_file = root / "run_settings.json"
    if settings_file.exists():
        repeats = json.loads(settings_file.read_text())["repeats"]
        logs = [root / "cases" / f"peer_plan_r{r}.log" for r in range(1, repeats + 1)]
    else:
        logs = sorted((root / "cases").glob("*.log"))
    for log in logs:
        results = []
        for line in (log.read_text(errors="replace").splitlines() if log.exists() else []):
            if line.startswith("RESULT_JSON "):
                try:
                    result = json.loads(line[len("RESULT_JSON "):])
                    if isinstance(result, dict):
                        results.append(result)
                except json.JSONDecodeError:
                    pass
        status_file = log.with_suffix(".status")
        status = status_file.read_text().strip() if status_file.exists() else "INTERRUPTED"
        valid = (status == "0" and len(results) == k and
                 {r.get("rank") for r in results} == set(range(k)) and
                 all(r.get("correctness") == "PASS" and r.get("ranks") == k and
                     r.get("host_call_us") and len(r["host_call_us"]) == r.get("iterations") and
                     isinstance(r.get("host_avg_us"), (int, float)) and math.isfinite(r["host_avg_us"])
                     for r in results))
        complete &= valid
        rows.append(dict(case=log.stem, status=status, correctness="PASS" if valid else "INCOMPLETE_OR_FAIL",
            slowest_rank_host_avg_us=max(r["host_avg_us"] for r in results) if valid else None))
    complete &= bool(rows)
    with (root / "peer_plan_results.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("case", "status", "correctness", "slowest_rank_host_avg_us"), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    summary = dict(complete=complete, devices=plan["devices"],
                   relay_limit_per_peer=plan["relay_limit_per_peer"], cases=rows)
    (root / "peer_plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# CCU explicit peer-plan AllToAll", "", f"- complete: `{complete}`",
              f"- communicator physical devices: `{plan['devices']}`",
              f"- relay limit per peer: `{plan['relay_limit_per_peer']}`", "",
              "| case | status | correctness | slowest-rank host average (us) |",
              "|---|---|---|---:|"]
    for row in rows:
        report.append(f"| {row['case']} | {row['status']} | {row['correctness']} | {row['slowest_rank_host_avg_us']} |")
    times = [r["slowest_rank_host_avg_us"] for r in rows if r["correctness"] == "PASS"]
    if times:
        report.extend(["", f"Median across valid repeats: {statistics.median(times):.3f} us."])
    report.extend(["", "Timing includes host submission and synchronization, excludes resource preparation and warmup.",
        "Use profiling CCU task durations for device-only timing; this host value is not link bandwidth.",
        "A correct result does not independently identify the physical relay. Correlate topology ports with HCCN deltas."])
    (root / "peer_plan_report.md").write_text("\n".join(report) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--run-dir", required=True, type=Path)
    args = p.parse_args()
    analyze(args.run_dir)
    print(args.run_dir / "peer_plan_report.md")
