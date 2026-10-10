#!/usr/bin/env python3
"""Do not turn an exit status or incomplete rank set into correctness PASS."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def valid_result(result, k):
    def numeric(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool) and
                math.isfinite(value) and value >= 0)
    rank, count = result.get("rank"), result.get("iterations")
    samples = result.get("host_call_us")
    if not (isinstance(rank, int) and not isinstance(rank, bool) and 0 <= rank < k and
            result.get("ranks") == k and result.get("correctness") == "PASS" and
            isinstance(count, int) and not isinstance(count, bool) and count > 0):
        return False
    mode = result.get("sync_mode", "per-call")
    if mode == "batch":
        gate = result.get("batch_validation")
        return (isinstance(gate, dict) and gate.get("status") == "PASS" and
            type(gate.get("rounds")) is int and gate["rounds"] == 2 and
            type(gate.get("calls_per_round")) is int and
            gate["calls_per_round"] == min(max(count, 2), 20) and
            samples == [] and numeric(result.get("host_batch_total_us")) and
            numeric(result.get("host_avg_us")) and
            math.isclose(result["host_avg_us"], result["host_batch_total_us"] / count,
                         rel_tol=1e-6, abs_tol=1e-6))
    return (mode == "per-call" and isinstance(samples, list) and len(samples) == count and
            all(numeric(value) for value in samples) and numeric(result.get("host_avg_us")) and
            math.isclose(result["host_avg_us"], sum(samples) / count, rel_tol=1e-6, abs_tol=1e-6))


def analyze(root):
    plan = json.loads((root / "plans/peer_plan.json").read_text())
    k = len(plan["devices"])
    rows, complete = [], True
    settings_file = root / "run_settings.json"
    settings = {}
    if settings_file.exists():
        settings = json.loads(settings_file.read_text())
        repeats = settings["repeats"]
        logs = [root / "cases" / f"peer_plan_r{r}.log" for r in range(1, repeats + 1)]
    else:
        logs = sorted((root / "cases").glob("*.log"))
    for log in logs:
        results = []
        last_phase = {}
        for line in (log.read_text(errors="replace").splitlines() if log.exists() else []):
            if line.startswith("PEER_CASE_PHASE "):
                fields = dict(part.split("=", 1) for part in line.split()[1:] if "=" in part)
                if "rank" in fields and "phase" in fields:
                    last_phase[fields["rank"]] = fields["phase"]
            if line.startswith("RESULT_JSON "):
                try:
                    result = json.loads(line[len("RESULT_JSON "):])
                    if isinstance(result, dict):
                        results.append(result)
                except json.JSONDecodeError:
                    pass
        # New runs use atomic per-rank files; never let interleaved stdout or
        # a missing rank file become a successful batch result. Legacy logs
        # remain readable when no results directory exists.
        result_dir = root / log.stem.replace("peer_plan_", "") / "results"
        if result_dir.is_dir():
            results = []
            for rank in range(k):
                try:
                    result = json.loads((result_dir / f"rank{rank}.json").read_text())
                    if isinstance(result, dict) and result.get("rank") == rank:
                        results.append(result)
                except (OSError, ValueError):
                    pass
        status_file = log.with_suffix(".status")
        status = status_file.read_text().strip() if status_file.exists() else "INTERRUPTED"
        data_valid = (len(results) == k and
                 all(valid_result(r, k) for r in results) and
                 {r.get("rank") for r in results} == set(range(k)) and
                 len({r.get("sync_mode", "per-call") for r in results}) == 1 and
                 all(all(key not in settings or r.get(key, "per-call" if key == "sync_mode" else None) == settings[key]
                         for key in ("sync_mode", "warmup", "iterations", "bytes_per_peer", "graph_backend"))
                     for r in results))
        valid = status == "0" and data_valid
        complete &= valid
        rows.append(dict(case=log.stem, status=status,
            sync_mode=results[0].get("sync_mode", "per-call") if data_valid else settings.get("sync_mode", "UNKNOWN"),
            correctness="PASS" if data_valid else "INCOMPLETE_OR_FAIL",
            lifecycle=("PASS" if valid else "FAILED_OR_INTERRUPTED_AFTER_DATA_PASS"
                       if data_valid else "INCOMPLETE_OR_FAIL"),
            last_phase_by_rank=json.dumps(last_phase, sort_keys=True),
            slowest_rank_host_avg_us=max(r["host_avg_us"] for r in results) if data_valid else None))
    complete &= bool(rows)
    with (root / "peer_plan_results.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("case", "status", "sync_mode", "correctness", "lifecycle",
            "last_phase_by_rank", "slowest_rank_host_avg_us"), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    summary = dict(complete=complete, devices=plan["devices"],
                   relay_limit_per_peer=plan["relay_limit_per_peer"], cases=rows)
    (root / "peer_plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# CCU explicit peer-plan AllToAll", "", f"- complete: `{complete}`",
              f"- communicator physical devices: `{plan['devices']}`",
              f"- relay limit per peer: `{plan['relay_limit_per_peer']}`", "",
              "| case | status | sync mode | data correctness | lifecycle | slowest-rank host average (us) |",
              "|---|---|---|---|---|---:|"]
    for row in rows:
        report.append(f"| {row['case']} | {row['status']} | {row['sync_mode']} | {row['correctness']} | {row['lifecycle']} | {row['slowest_rank_host_avg_us']} |")
    report.extend(["", "## Last observed lifecycle phase per rank", ""])
    for row in rows:
        report.append(f"- {row['case']}: `{row['last_phase_by_rank']}`")
    report.extend(["", "## Shutdown observer outputs", ""])
    for scope_file in sorted(root.glob("r*/shutdown_trace/scope.json")):
        scope = json.loads(scope_file.read_text())
        pids = [process["pid"] for process in scope.get("processes", [])]
        report.append(f"- `{scope_file.relative_to(root)}`: still-matching process PIDs `{pids}`; "
                      "see proc snapshots and optional native_pid*.txt. Kernel waits alone do not identify a user-space destructor.")
    times = [r["slowest_rank_host_avg_us"] for r in rows if r["lifecycle"] == "PASS"]
    if times:
        report.extend(["", f"Median across valid repeats: {statistics.median(times):.3f} us."])
    report.extend(["", "Timing includes host submission and synchronization, excludes resource preparation and warmup.",
        "Data correctness PASS with nonzero status is not a complete run. Reported timings are retained as diagnostic values, not accepted performance samples.",
        "Use profiling CCU task durations for device-only timing; this host value is not link bandwidth.",
        "Batch host us is total batch time / iterations, not individual-call latency. Compare only matching sync modes.",
        "A batch result requires two queued changed-input/poisoned-output validation rounds with every snapshot checked before warmup/profiling.",
        "Passing this bounded preflight does not certify arbitrary concurrency, sizes, streams or graph replay. Graph batch is not enabled.",
        "New runs use r*/results/rank*.json as authoritative; old stdout-only runs remain supported.",
        "A correct result does not independently identify the physical relay. Correlate topology ports with HCCN deltas."])
    (root / "peer_plan_report.md").write_text("\n".join(report) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--run-dir", required=True, type=Path)
    args = p.parse_args()
    analyze(args.run_dir)
    print(args.run_dir / "peer_plan_report.md")
